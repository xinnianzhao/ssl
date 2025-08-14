#!/usr/bin/env python
# coding=utf-8
"""Fine-tuning WavLM model for CTC-based ASR using CGN dataset and custom BPE tokenizer"""

import functools
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple

import datasets
import evaluate
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from pathlib import Path
try:
    import yaml  # optional
except Exception:
    yaml = None

import transformers
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
    EarlyStoppingCallback,
    ProcessorMixin,
    PretrainedConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
import torch
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import wandb

# Import custom tokenizer
from utils.tokenizer.bpe_ctc_tokenizer import BPECTCTokenizer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.30.0")

require_version("datasets>=1.18.0", "To fix: pip install -r requirements.txt")

logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


def _is_espnet_ssl_checkpoint_dir(path_str: str) -> bool:
    """Return True if directory looks like an ESPnet SSL checkpoint folder."""
    p = Path(path_str)
    if not p.is_dir():
        return False
    has_yaml = (p / "config.yaml").is_file()
    has_pth = any(p.glob("*.pth"))
    return has_yaml and has_pth


def _load_yaml_or_empty(path: Path) -> Dict[str, Any]:
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _init_wav2vec2_base_model(tokenizer, base_model_name: str, cache_dir: Optional[str], token: Optional[str], trust_remote_code: bool) -> Tuple[Wav2Vec2Config, Wav2Vec2ForCTC]:
    """Create a HF Wav2Vec2ForCTC using a base HF checkpoint and adjust vocab-related fields."""
    config = Wav2Vec2Config.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    
    # Set conv_bias to False to match ESPnet training
    config.conv_bias = False  # Uncomment if you want to match ESPnet exactly

    model = Wav2Vec2ForCTC.from_pretrained(
        base_model_name,
        config=config,
        cache_dir=cache_dir,
        token=token,
        trust_remote_code=trust_remote_code,
        ignore_mismatched_sizes=True,
    )
    return config, model


def _load_partial_espnet_weights_into_hf(model: Wav2Vec2ForCTC, checkpoint_dir: str) -> Dict[str, Any]:
    """Attempt to load ESPnet .pth weights into a HF Wav2Vec2 model with best-effort key matching.

    Returns a stats dict with counts of loaded and total parameters.
    """
    ckpt_dir = Path(checkpoint_dir)
    pth_files = sorted(ckpt_dir.glob("*.pth"))
    if not pth_files:
        logger.warning(f"No .pth found under {checkpoint_dir}")
        return {"loaded_keys": 0, "total_hf_keys": 0}

    # Heuristic: prefer '5epoch.pth' if present
    pth_path = next((p for p in pth_files if p.name.startswith("5epoch")), pth_files[-1])
    logger.info(f"Loading ESPnet checkpoint: {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")

    # Try common containers
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            src_sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            src_sd = ckpt["state_dict"]
        else:
            # Maybe it's already a raw state dict
            src_sd = ckpt
    else:
        logger.warning("Unexpected checkpoint format; skip weight loading")
        return {"loaded_keys": 0, "total_hf_keys": 0}

    hf_sd = model.state_dict()
    new_sd = {}
    
    # Create mapping between ESPnet and HF keys
    def map_espnet_to_hf_key(espnet_key: str) -> Optional[str]:
        """Map ESPnet key to HuggingFace key format."""
        
        # Start with the original key
        hf_key = espnet_key
        
        # Step 1: Handle the main prefix transformations
        if hf_key.startswith("encoder.hubert_pretrain_model.wav2vec2."):
            hf_key = hf_key.replace("encoder.hubert_pretrain_model.wav2vec2.", "wav2vec2.")
        elif hf_key.startswith("encoder.hubert_pretrain_model."):
            hf_key = hf_key.replace("encoder.hubert_pretrain_model.", "wav2vec2.")
        elif hf_key.startswith("encoder.wav2vec2."):
            hf_key = hf_key.replace("encoder.wav2vec2.", "wav2vec2.")
        elif hf_key.startswith("encoder."):
            # Be careful with this one - only if it's not already handled
            if not hf_key.startswith("wav2vec2."):
                hf_key = hf_key.replace("encoder.", "wav2vec2.", 1)
        
        # Step 2: Handle encoder.transformer -> encoder renaming
        hf_key = hf_key.replace(".encoder.transformer.", ".encoder.")
        hf_key = hf_key.replace("wav2vec2.encoder.transformer.", "wav2vec2.encoder.")
        
        # Step 3: Handle specific component renaming
        # Feed forward layer renaming
        hf_key = hf_key.replace(".feed_forward.w_1.", ".feed_forward.intermediate_dense.")
        hf_key = hf_key.replace(".feed_forward.w_2.", ".feed_forward.output_dense.")
        
        # Layer norm renaming
        hf_key = hf_key.replace(".feed_forward_macaron.w_1.", ".feed_forward.intermediate_dense.")
        hf_key = hf_key.replace(".feed_forward_macaron.w_2.", ".feed_forward.output_dense.")
        hf_key = hf_key.replace(".norm1.", ".layer_norm.")
        hf_key = hf_key.replace(".norm2.", ".final_layer_norm.")
        hf_key = hf_key.replace(".self_attn_layer_norm.", ".layer_norm.")
        hf_key = hf_key.replace(".final_layer_norm.", ".final_layer_norm.")
        
        # Attention component renaming  
        hf_key = hf_key.replace(".self_attn.", ".attention.")
        
        # Positional convolution renaming
        if "pos_conv_embed.conv.weight_g" in hf_key:
            hf_key = hf_key.replace("pos_conv_embed.conv.weight_g", "pos_conv_embed.conv.parametrizations.weight.original0")
        elif "pos_conv_embed.conv.weight_v" in hf_key:
            hf_key = hf_key.replace("pos_conv_embed.conv.weight_v", "pos_conv_embed.conv.parametrizations.weight.original1")
        
        # Feature projection renaming
        hf_key = hf_key.replace("encoder.feature_projection.", "feature_projection.")
        
        # Masked spec embed renaming
        if hf_key == "wav2vec2.mask_emb":
            hf_key = "wav2vec2.masked_spec_embed"
        
        return hf_key if hf_key != espnet_key else None

    loaded = 0
    matched_keys = []
    
    # First try direct mapping from ESPnet to HF
    for espnet_key, espnet_val in src_sd.items():
        # Skip non-tensor values
        if not isinstance(espnet_val, torch.Tensor):
            continue
            
        # Try to map ESPnet key to HF key
        hf_key = map_espnet_to_hf_key(espnet_key)
        
        if hf_key and hf_key in hf_sd:
            if hf_sd[hf_key].shape == espnet_val.shape:
                new_sd[hf_key] = espnet_val
                loaded += 1
                matched_keys.append(hf_key)
                logger.debug(f"Matched: {espnet_key} -> {hf_key}")
    
    # Second pass: try to match remaining HF keys
    for hf_key, hf_val in hf_sd.items():
        if hf_key.startswith("lm_head."):
            continue  # skip CTC head; different vocab
        if hf_key in matched_keys:
            continue  # already matched
            
        # Generate possible ESPnet key candidates
        candidates = []
        
        # Basic patterns
        if hf_key.startswith("wav2vec2."):
            base_key = hf_key[len("wav2vec2."):]
            candidates.extend([
                f"encoder.hubert_pretrain_model.wav2vec2.{base_key}",
                f"encoder.hubert_pretrain_model.{base_key}",
                f"encoder.wav2vec2.{base_key}",
                f"encoder.{base_key}",
                f"model.{base_key}",
            ])
        
        # Try each candidate
        for cand in candidates:
            if cand in src_sd and src_sd[cand].shape == hf_val.shape:
                new_sd[hf_key] = src_sd[cand]
                loaded += 1
                matched_keys.append(hf_key)
                logger.debug(f"Matched (2nd pass): {cand} -> {hf_key}")
                break

    missing = [k for k in hf_sd.keys() if k not in new_sd and not k.startswith("lm_head.")]
    logger.info(
        f"ESPnet->HF weight load: matched {loaded} / {len(hf_sd)} HF tensors; "
        f"unmatched (first 10): {missing[:10]}"
    )
    
    # Log all missing keys for debugging
    if missing:
        logger.info("All unmatched HF keys (excluding lm_head):")
        for key in missing:
            logger.info(f"  - {key}")

    model.load_state_dict({**hf_sd, **new_sd}, strict=False)
    return {"loaded_keys": loaded, "total_hf_keys": len(hf_sd), "matched_keys": matched_keys}


class CustomProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "Wav2Vec2FeatureExtractor"
    tokenizer_class = "PreTrainedTokenizer"
    
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
    
    def __call__(self, audio=None, text=None, **kwargs):
        # Process audio
        if audio is not None:
            inputs = self.feature_extractor(audio, **kwargs)
        
        # Process text
        if text is not None:
            text_inputs = self.tokenizer(text, **kwargs)
            if audio is not None:
                inputs.update(text_inputs)
            else:
                inputs = text_inputs
        
        return inputs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        tokenizer = BPECTCTokenizer.from_pretrained(pretrained_model_name_or_path)
        return cls(feature_extractor, tokenizer)
    
    def save_pretrained(self, save_directory, **kwargs):
        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_vocabulary(save_directory)
        # generate custom config file
        preprocessor_config = PretrainedConfig()
        preprocessor_config.update({
            "processor_class": "CustomProcessor",
            "feature_extractor_type": self.feature_extractor.__class__.__name__,
            "tokenizer_type": self.tokenizer.__class__.__name__,
        })
        preprocessor_config.save_pretrained(save_directory)


def get_parameter_groups(
    model,
    base_lr=1e-4,          # 顶层 encoder 的 LR
    layer_decay=0.95,      # LLRD 衰减
    ctc_lr=3e-4,           # CTC 头 LR
    weight_decay=0.01,
    freeze_feature_extractor=True,  # 通常冻结 CNN
):
    """
    返回可直接给 AdamW 的 optimizer_grouped_parameters。
    规则：
      - CTC 头单独 lr
      - encoder.layers.* 做 LLRD（越底层 lr 越小）
      - encoder.pos_conv_embed & encoder.layer_norm 视为最底层（最小 lr）
      - feature_projection 给次低 lr
      - feature_extractor 可冻结或给极低 lr
      - bias/LayerNorm 参数 weight_decay=0
      - 仅包含 requires_grad=True 的参数（支持 warmup 阶段只训 CTC）
    """
    no_decay_keywords = ["bias", "LayerNorm.weight", "layer_norm.weight", "layer_norm.bias"]
    def is_no_decay(n):
        return any(k in n for k in no_decay_keywords)

    if freeze_feature_extractor:
        for p in model.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False

    # 统计层数
    num_layers = len(model.wav2vec2.encoder.layers)

    groups = []  # 会填充多个 {params, lr, weight_decay, name}

    # 1) CTC 头
    decay, no_decay = [], []
    for n, p in model.lm_head.named_parameters():
        if not p.requires_grad: 
            continue
        (no_decay if is_no_decay(n) else decay).append(p)
    if decay:
        groups.append({"params": decay, "lr": ctc_lr, "weight_decay": weight_decay, "name": "ctc_head.decay"})
    if no_decay:
        groups.append({"params": no_decay, "lr": ctc_lr, "weight_decay": 0.0, "name": "ctc_head.no_decay"})

    grouped_ids = set(id(p) for g in groups for p in g["params"])

    # 2) encoder 层做 LLRD
    for layer_idx, layer in enumerate(model.wav2vec2.encoder.layers):
        layer_lr = base_lr * (layer_decay ** (num_layers - 1 - layer_idx))
        decay, no_decay = [], []
        for n, p in layer.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if is_no_decay(n) else decay).append(p)
        if decay:
            groups.append({"params": decay, "lr": layer_lr, "weight_decay": weight_decay,
                           "name": f"encoder.layer{layer_idx}.decay"})
        if no_decay:
            groups.append({"params": no_decay, "lr": layer_lr, "weight_decay": 0.0,
                           "name": f"encoder.layer{layer_idx}.no_decay"})
        grouped_ids |= {id(p) for p in decay + no_decay}

    # 3) 其它 encoder 组件（pos_conv_embed / layer_norm）
    # 这些通常靠底部，给最小 lr
    min_lr = base_lr * (layer_decay ** (num_layers - 1))
    for mod_name in ["pos_conv_embed", "layer_norm"]:
        if hasattr(model.wav2vec2.encoder, mod_name):
            decay, no_decay = [], []
            module = getattr(model.wav2vec2.encoder, mod_name)
            for n, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if is_no_decay(n) else decay).append(p)
            if decay:
                groups.append({"params": decay, "lr": min_lr, "weight_decay": weight_decay,
                               "name": f"encoder.{mod_name}.decay"})
            if no_decay:
                groups.append({"params": no_decay, "lr": min_lr, "weight_decay": 0.0,
                               "name": f"encoder.{mod_name}.no_decay"})
            grouped_ids |= {id(p) for p in decay + no_decay}

    # 4) feature_projection（投影+LN），给次低 lr
    proj_lr = base_lr * (layer_decay ** (num_layers - 1))
    decay, no_decay = [], []
    for n, p in model.wav2vec2.feature_projection.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if is_no_decay(n) else decay).append(p)
    if decay:
        groups.append({"params": decay, "lr": proj_lr, "weight_decay": weight_decay,
                       "name": "feature_projection.decay"})
    if no_decay:
        groups.append({"params": no_decay, "lr": proj_lr, "weight_decay": 0.0,
                       "name": "feature_projection.no_decay"})
    grouped_ids |= {id(p) for p in decay + no_decay}

    # 5) feature_extractor（CNN）：通常冻结；若不冻结，给最小 lr 或更小
    if not freeze_feature_extractor:
        fe_lr = base_lr * (layer_decay ** (num_layers + 1))  # 更小一些
        decay, no_decay = [], []
        for n, p in model.wav2vec2.feature_extractor.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if is_no_decay(n) else decay).append(p)
        if decay:
            groups.append({"params": decay, "lr": fe_lr, "weight_decay": weight_decay,
                           "name": "feature_extractor.decay"})
        if no_decay:
            groups.append({"params": no_decay, "lr": fe_lr, "weight_decay": 0.0,
                           "name": "feature_extractor.no_decay"})
        grouped_ids |= {id(p) for p in decay + no_decay}

    # 6) 兜底：任何遗漏但可训练的参数（一般不会有）
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in grouped_ids:
            continue
        (no_decay if is_no_decay(n) else decay).append(p)
    if decay:
        groups.append({"params": decay, "lr": base_lr, "weight_decay": weight_decay, "name": "other.decay"})
    if no_decay:
        groups.append({"params": no_decay, "lr": base_lr, "weight_decay": 0.0, "name": "other.no_decay"})

    return groups


class FreezeEncoderCallback(TrainerCallback):
    """
    Callback to handle progressive unfreezing of encoder.
    First warmup the CTC layer for freeze_encoder_steps, then unfreeze encoder.
    """
    
    def __init__(self, freeze_encoder_steps: int, model, trainer=None):
        self.freeze_encoder_steps = freeze_encoder_steps
        self.model = model
        self.trainer = trainer
        self.encoder_frozen = freeze_encoder_steps > 0
        self.stage_b_start_step = None  # Track when Stage B starts
        
        if self.encoder_frozen:
            # Initially freeze encoder (Stage A: CTC warmup)
            for param in self.model.wav2vec2.parameters():
                param.requires_grad = False
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            logger.info("Stage A: Freezing encoder, training CTC head only")
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.encoder_frozen and state.global_step >= self.freeze_encoder_steps:
            # Stage B: Unfreeze encoder and rebuild optimizer
            logger.info(f"\nStage B: Unfreezing encoder at step {state.global_step}")
            
            # Record when Stage B starts
            self.stage_b_start_step = state.global_step
            
            # Unfreeze encoder layers
            for param in self.model.wav2vec2.parameters():
                param.requires_grad = True
            
            # Rebuild optimizer and scheduler
            if self.trainer and hasattr(kwargs.get('trainer', None), 'optimizer'):
                trainer = kwargs['trainer']
                # Store Stage B start step in trainer for scheduler creation
                trainer.stage_b_start_step = self.stage_b_start_step
                # Reset optimizer and scheduler to trigger recreation
                trainer.optimizer = None
                trainer.lr_scheduler = None
                # Create new optimizer with updated parameter groups
                trainer.create_optimizer()
                # Create new scheduler for Stage B
                trainer.create_scheduler(trainer.args.max_steps - self.stage_b_start_step, trainer.optimizer)
                logger.info("Optimizer and scheduler rebuilt for Stage B")
            
            self.encoder_frozen = False
        return control


class CTCTrainer(Trainer):
    """
    Custom trainer that logs predictions during evaluation and supports layer-wise learning rate decay.
    """
    
    def __init__(self, *args, num_examples_to_log=5, use_layerwise_lr_decay=True, freeze_encoder_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_examples_to_log = num_examples_to_log
        self.eval_predictions = None
        self.eval_labels = None
        self.use_layerwise_lr_decay = use_layerwise_lr_decay
        self.freeze_encoder_steps = freeze_encoder_steps
        self.stage_b_start_step = None  # Will be set by callback
    
    def compute_metrics(self, eval_pred):
        """Override to store predictions for logging."""
        if self.args.predict_with_generate:
            # Handle generation case if needed
            return super().compute_metrics(eval_pred)
        
        # Call the original compute_metrics
        metrics = self.compute_metrics_fn(eval_pred) if self.compute_metrics_fn is not None else {}
        
        # Extract predictions from function attributes if available
        if hasattr(self.compute_metrics_fn, 'pred_str') and hasattr(self.compute_metrics_fn, 'label_str'):
            self.eval_predictions = self.compute_metrics_fn.pred_str
            self.eval_labels = self.compute_metrics_fn.label_str
            
            # Log sample predictions
            if is_main_process(self.args.local_rank):
                logger.info("\n" + "="*80)
                logger.info("Sample Predictions from Evaluation:")
                logger.info("="*80)
                
                num_to_show = min(self.num_examples_to_log, len(self.eval_predictions))
                for i in range(num_to_show):
                    logger.info(f"\nExample {i+1}:")
                    logger.info(f"  Label: {self.eval_labels[i] if i < len(self.eval_labels) else 'N/A'}")
                    logger.info(f"  Pred:  {self.eval_predictions[i] if i < len(self.eval_predictions) else 'N/A'}")
                
                logger.info("="*80 + "\n")
                
                # Log to wandb if available
                if wandb.run is not None:
                    # Create a table for wandb
                    table_data = []
                    for i in range(num_to_show):
                        table_data.append([
                            i+1,
                            self.eval_labels[i] if i < len(self.eval_labels) else 'N/A',
                            self.eval_predictions[i] if i < len(self.eval_predictions) else 'N/A'
                        ])
                    
                    wandb.log({
                        "eval/predictions_table": wandb.Table(
                            columns=["Example", "Label", "Prediction"],
                            data=table_data
                        )
                    })
        
        return metrics
    
    def create_optimizer(self):
        """
        Override to create optimizer with layer-wise learning rate decay.
        """
        if self.optimizer is None:
            if self.use_layerwise_lr_decay:
                # Check if encoder is frozen (Stage A: CTC warmup)
                encoder_frozen = not any(p.requires_grad for p in self.model.wav2vec2.encoder.parameters())
                
                if encoder_frozen:
                    # Stage A: Only CTC head is trainable
                    logger.info("Creating optimizer for Stage A (CTC warmup only)")
                    parameter_groups = get_parameter_groups(
                        self.model,
                        base_lr=1e-4,
                        layer_decay=0.95,
                        ctc_lr=5e-4,  # Higher LR for CTC during warmup
                        weight_decay=0.01,
                        freeze_feature_extractor=True
                    )
                else:
                    # Stage B: Full model fine-tuning with LLRD
                    logger.info("Creating optimizer for Stage B (full model with LLRD)")
                    parameter_groups = get_parameter_groups(
                        self.model,
                        base_lr=1e-4,
                        layer_decay=0.95,
                        ctc_lr=3e-4,
                        weight_decay=0.01,
                        freeze_feature_extractor=True  # Keep CNN frozen
                    )
                
                # Log parameter groups
                logger.info("Parameter groups created:")
                for group in parameter_groups:
                    num_params = sum(p.numel() for p in group['params'])
                    logger.info(f"  {group['name']}: {num_params:,} params, lr={group['lr']:.2e}, wd={group['weight_decay']}")
                
                # Create AdamW optimizer
                self.optimizer = AdamW(
                    parameter_groups,
                    betas=(0.9, 0.98),
                    eps=1e-8
                )
            else:
                # Fall back to default optimizer creation
                return super().create_optimizer()
        
        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Override to create appropriate scheduler for each stage with independent warmup.
        """
        if self.lr_scheduler is None:
            if optimizer is None:
                optimizer = self.optimizer
            
            # Check if we're in Stage A or Stage B
            encoder_frozen = not any(p.requires_grad for p in self.model.wav2vec2.encoder.parameters())
            
            if encoder_frozen:
                # Stage A: CTC warmup
                # Use 10% of Stage A steps for warmup
                warmup_steps = int(self.freeze_encoder_steps * 0.1)
                logger.info(f"Creating Stage A scheduler: linear warmup for {warmup_steps} steps, total {self.freeze_encoder_steps} steps")
                
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=self.freeze_encoder_steps
                )
            else:
                # Stage B: Full model training
                # Calculate remaining steps after Stage A
                if hasattr(self, 'stage_b_start_step') and self.stage_b_start_step is not None:
                    stage_b_steps = self.args.max_steps - self.stage_b_start_step
                else:
                    # If called directly without Stage A
                    stage_b_steps = num_training_steps
                
                # Use 10% of Stage B steps for warmup
                warmup_steps = int(stage_b_steps * 0.1)
                logger.info(f"Creating Stage B scheduler: {'cosine' if self.args.lr_scheduler_type == 'cosine' else 'linear'} warmup for {warmup_steps} steps, total {stage_b_steps} steps")
                
                if self.args.lr_scheduler_type == "cosine":
                    self.lr_scheduler = get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=stage_b_steps
                    )
                else:
                    self.lr_scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=stage_b_steps
                    )
        
        return self.lr_scheduler


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/wav2vec2-large-xlsr-53",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_vocab_file: str = field(
        default="./utils/tokenizer/vocab.json",
        metadata={"help": "Path to tokenizer vocab.json file"},
    )
    tokenizer_merges_file: Optional[str] = field(
        default="./utils/tokenizer/merges.txt",
        metadata={"help": "Path to tokenizer merges.txt file"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    attention_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    feat_proj_dropout: float = field(default=0.0, metadata={"help": "The dropout ratio for the projected features."})
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector"
                "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
                "vectors will be masked along the time axis."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of each feature vector along the feature axis to be chosen as the start of the vector"
                "span to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature bins will be masked along the time axis."
            )
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: Optional[str] = field(
        default="sum", metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."}
    )
    ctc_zero_infinity: bool = field(
        default=False,
        metadata={
            "help": "Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly occur when the inputs are too short to be aligned to the targets."
        },
    )
    freeze_encoder_steps: int = field(
        default=0,
        metadata={"help": "Number of steps to warm up CTC layer before unfreezing encoder. 0 means no warmup."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    cgn_data_dir: str = field(
        default="/home/xinyu/data1/HuggingFace/datasets/CGN/data",
        metadata={"help": "Path to CGN dataset directory with parquet files"}
    )
    train_split_name: str = field(
        default="train",
        metadata={"help": "The name of the training data split to use."},
    )
    eval_split_name: str = field(
        default="validation",
        metadata={"help": "The name of the evaluation data split to use."},
    )
    test_split_name: str = field(
        default="test",
        metadata={"help": "The name of the test data split to use."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this value if set."
        },
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    wandb_project: str = field(
        default="ssl",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_run_name: Optional[str] = field(
        default="wavlm-cgn",
        metadata={"help": "Weights & Biases run name. If not specified, will be auto-generated."}
    )
    num_examples_to_log: int = field(
        default=5,
        metadata={"help": "Number of prediction examples to log during evaluation."}
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={"help": "Number of evaluation calls with no improvement after which training will be stopped."}
    )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """

    processor: CustomProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Use tokenizer's pad method instead of manual padding
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt"
        )
        
        batch["labels"] = labels_batch["input_ids"]
        # replace padding with -100 to ignore loss correctly
        batch["labels"][batch["labels"] == self.processor.tokenizer.pad_token_id] = -100
        
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch


def load_cgn_data(data_dir: str, split: str) -> Dataset:
    """Load CGN data from parquet files without materializing the whole split in RAM."""
    parquet_files = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found for split '{split}' in {data_dir}")
    
    # Use HuggingFace datasets parquet loader (memory-mapped Arrow) instead of pandas concat
    data_files = [str(p) for p in parquet_files]
    ds_dict = datasets.load_dataset("parquet", data_files={"data": data_files})
    dataset = ds_dict["data"]
    
    return dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_speech_recognition_ctc", model_args, data_args)

    # Setup logging
    os.environ["WANDB_PROJECT"] = data_args.wandb_project
    report_to_wandb = "wandb" in training_args.report_to
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize wandb if requested
    if report_to_wandb and is_main_process(training_args.local_rank):
        wandb.init(
            project=data_args.wandb_project,
            name=data_args.wandb_run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "dataset": "CGN",
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "num_epochs": training_args.num_train_epochs,
                "freeze_encoder_steps": model_args.freeze_encoder_steps,
            }
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load CGN datasets
    raw_datasets = DatasetDict()
    
    if training_args.do_train:
        raw_datasets["train"] = load_cgn_data(data_args.cgn_data_dir, data_args.train_split_name)
        
    if training_args.do_eval:
        raw_datasets["eval"] = load_cgn_data(data_args.cgn_data_dir, data_args.eval_split_name)
        
    if training_args.do_predict:
        raw_datasets["test"] = load_cgn_data(data_args.cgn_data_dir, data_args.test_split_name)

    # Initialize custom tokenizer
    # Check if we're loading from a HF model that already has tokenizer files
    if not _is_espnet_ssl_checkpoint_dir(model_args.model_name_or_path):
        # Try to load tokenizer from model path first
        hf_vocab_path = os.path.join(model_args.model_name_or_path, "vocab.json")
        hf_merges_path = os.path.join(model_args.model_name_or_path, "merges.txt")
        
        if os.path.exists(hf_vocab_path) and os.path.exists(hf_merges_path):
            # Use tokenizer from HF model directory
            tokenizer = BPECTCTokenizer(
                vocab_file=hf_vocab_path,
                merges_file=hf_merges_path,
            )
        else:
            # Fall back to provided paths
            tokenizer = BPECTCTokenizer(
                vocab_file=model_args.tokenizer_vocab_file,
                merges_file=model_args.tokenizer_merges_file,
            )
    else:
        # Use provided paths for ESPnet checkpoint
        tokenizer = BPECTCTokenizer(
            vocab_file=model_args.tokenizer_vocab_file,
            merges_file=model_args.tokenizer_merges_file,
        )

    # Load feature extractor (using Wav2Vec2FeatureExtractor)
    # If an ESPnet checkpoint dir is provided, load FE from a compatible HF base.
    fe_source = (
        "facebook/wav2vec2-large-xlsr-53" if _is_espnet_ssl_checkpoint_dir(model_args.model_name_or_path)
        else model_args.model_name_or_path
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        fe_source,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )
    
    # Create custom processor with both feature extractor and tokenizer
    processor = CustomProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Build model/config
    if _is_espnet_ssl_checkpoint_dir(model_args.model_name_or_path):
        logger.info(f"Detected ESPnet SSL checkpoint dir: {model_args.model_name_or_path}")
        # Initialize from a HF base (e.g., facebook/wav2vec2-large-xlsr-53), then load partial weights
        base_name = "facebook/wav2vec2-large-xlsr-53"
        config, model = _init_wav2vec2_base_model(
            tokenizer,
            base_model_name=base_name,
            cache_dir=model_args.cache_dir,
            token=data_args.token,
            trust_remote_code=data_args.trust_remote_code,
        )
        stats = _load_partial_espnet_weights_into_hf(model, model_args.model_name_or_path)
        logger.info(f"Loaded ESPnet weights: {stats}")
    else:
        # Standard HF init
        config = Wav2Vec2Config.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            token=data_args.token,
            trust_remote_code=data_args.trust_remote_code,
        )
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id

        model = Wav2Vec2ForCTC.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            token=data_args.token,
            trust_remote_code=data_args.trust_remote_code,
            ignore_mismatched_sizes=True,  # Important: Allow resizing of output layer
        )
    # Freeze feature encoder if requested
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()
    # Make CTCLoss robust against invalid alignment (avoid inf -> NaN)
    model.config.ctc_zero_infinity = True

    # Preprocessing the datasets
    def prepare_dataset(batch):
        # Load audio from bytes
        import soundfile as sf
        import io
        
        audio_bytes = batch["audio"]["bytes"]
        audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Extract input values
        inputs = feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="np"
        )
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(inputs.input_values[0])  # raw samples length
        
        # Encode labels using custom tokenizer
        batch["labels"] = tokenizer.encode(batch["transcription"])      
        
        return batch

    # Vectorize dataset
    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=raw_datasets[list(raw_datasets.keys())[0]].column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc="preprocess datasets",
        )

    # Filter dataset by duration
    def is_audio_in_length_range(length):
        return (
            length > data_args.min_duration_in_seconds * feature_extractor.sampling_rate
            and length < data_args.max_duration_in_seconds * feature_extractor.sampling_rate
        )

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
        num_proc=data_args.preprocessing_num_workers,
    )

    # Apply sample limits if specified
    if training_args.do_train and data_args.max_train_samples is not None:
        max_train_samples = min(len(vectorized_datasets["train"]), data_args.max_train_samples)
        vectorized_datasets["train"] = vectorized_datasets["train"].select(range(max_train_samples))

    if training_args.do_eval and data_args.max_eval_samples is not None:
        max_eval_samples = min(len(vectorized_datasets["eval"]), data_args.max_eval_samples)
        vectorized_datasets["eval"] = vectorized_datasets["eval"].select(range(max_eval_samples))

    if training_args.do_predict and data_args.max_test_samples is not None:
        max_test_samples = min(len(vectorized_datasets["test"]), data_args.max_test_samples)
        vectorized_datasets["test"] = vectorized_datasets["test"].select(range(max_test_samples))

    # Data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding="longest",
    )

    # Metrics
    wer_metric = evaluate.load("wer", cache_dir=model_args.cache_dir)

    def preprocess_logits_for_metrics(logits, labels):
        # logits can be a tuple when using some models (e.g., with past_key_values)
        if isinstance(logits, tuple):
            logits = logits[0]
        # Return argmax over vocabulary dimension to drastically reduce memory during accumulation
        return logits.argmax(dim=-1)

    def compute_metrics(pred):
        # pred.predictions can be either full logits (ndim==3) or already argmaxed ids (ndim==2)
        pred_array = pred.predictions
        if isinstance(pred_array, np.ndarray) and pred_array.ndim == 3:
            pred_ids = np.argmax(pred_array, axis=-1)
        else:
            pred_ids = pred_array

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        # Store predictions for logging by CTCTrainer
        # Don't return them in metrics to avoid logging format errors
        compute_metrics.pred_str = pred_str
        compute_metrics.label_str = label_str

        return {"wer": wer}

    # Set up callbacks
    callbacks = []
    freeze_callback = None
    if model_args.freeze_encoder_steps > 0:
        freeze_callback = FreezeEncoderCallback(model_args.freeze_encoder_steps, model)
        callbacks.append(freeze_callback)
    
    # Add early stopping callback if patience is specified
    if data_args.early_stopping_patience is not None and data_args.early_stopping_patience > 0:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=data_args.early_stopping_patience
        )
        callbacks.append(early_stopping_callback)
        logger.info(f"Early stopping enabled with patience={data_args.early_stopping_patience}")

    # Initialize our custom Trainer with prediction logging
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        tokenizer=feature_extractor,
        callbacks=callbacks,
        num_examples_to_log=data_args.num_examples_to_log,
        use_layerwise_lr_decay=True,  # Enable layer-wise learning rate decay
        freeze_encoder_steps=model_args.freeze_encoder_steps,  # Pass freeze steps to trainer
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    # Set trainer reference in callback for optimizer rebuilding
    if freeze_callback:
        freeze_callback.trainer = trainer

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        if data_args.max_train_samples:
            metrics["train_samples"] = data_args.max_train_samples

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        if data_args.max_eval_samples:
            metrics["eval_samples"] = data_args.max_eval_samples

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Test ***")
        test_dataset = vectorized_datasets["test"]
        
        predictions = trainer.predict(test_dataset, metric_key_prefix="test")
        metrics = predictions.metrics
        if data_args.max_test_samples:
            metrics["test_samples"] = data_args.max_test_samples

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # Write model configuration
    config_name = "config.json"
    if training_args.output_dir is not None:
        config_file = os.path.join(training_args.output_dir, config_name)
        with open(config_file, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    # Save tokenizer 
    if training_args.output_dir is not None:
        tokenizer.save_pretrained(training_args.output_dir)

    return results


if __name__ == "__main__":
    main()