#!/usr/bin/env python
# coding=utf-8
"""Fine-tuning Whisper model for Seq2Seq ASR using CGN dataset and custom BPE tokenizer"""

import functools
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import datasets
import evaluate
import torch
import torch.optim
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from pathlib import Path

import transformers
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperProcessor,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback,
    EarlyStoppingCallback,
    ProcessorMixin,
    PretrainedConfig,
    GenerationConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import wandb
from transformers.generation.utils import GenerationMixin

# Import custom tokenizer
from utils.tokenizer.bpe_aed_tokenizer import BPEAEDTokenizer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.30.0")

require_version("datasets>=1.18.0", "To fix: pip install -r requirements.txt")

logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


class CustomProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
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
        feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        tokenizer = BPEAEDTokenizer.from_pretrained(pretrained_model_name_or_path)
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


def get_whisper_parameter_groups(
    model,
    encoder_base_lr=3e-5,      # Encoder 顶层 LR
    decoder_base_lr=1e-4,      # Decoder 顶层 LR  
    lm_head_lr=3e-4,           # LM head LR
    layer_decay=0.95,          # LLRD 衰减
    weight_decay=0.01,
):
    """
    返回 Whisper 模型的分层学习率参数组。
    
    Whisper 结构：
    - model.encoder.layers[*]  # Encoder 层
    - model.decoder.layers[*]  # Decoder 层  
    - proj_out (lm_head)       # 输出投影层
    """
    no_decay_keywords = ["bias", "LayerNorm.weight", "layer_norm.weight", "layer_norm.bias"]
    def is_no_decay(n):
        return any(k in n for k in no_decay_keywords)

    groups = []
    grouped_ids = set()

    # 1) LM head (输出层)
    decay, no_decay = [], []
    for n, p in model.proj_out.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if is_no_decay(n) else decay).append(p)
    if decay:
        groups.append({"params": decay, "lr": lm_head_lr, "weight_decay": weight_decay, "name": "lm_head.decay"})
    if no_decay:
        groups.append({"params": no_decay, "lr": lm_head_lr, "weight_decay": 0.0, "name": "lm_head.no_decay"})
    grouped_ids |= {id(p) for p in decay + no_decay}

    # 2) Decoder 层 LLRD
    num_decoder_layers = len(model.model.decoder.layers)
    for layer_idx, layer in enumerate(model.model.decoder.layers):
        # 顶层是最后一层
        layer_lr = decoder_base_lr * (layer_decay ** (num_decoder_layers - 1 - layer_idx))
        decay, no_decay = [], []
        for n, p in layer.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if is_no_decay(n) else decay).append(p)
        if decay:
            groups.append({"params": decay, "lr": layer_lr, "weight_decay": weight_decay,
                           "name": f"decoder.layer{layer_idx}.decay"})
        if no_decay:
            groups.append({"params": no_decay, "lr": layer_lr, "weight_decay": 0.0,
                           "name": f"decoder.layer{layer_idx}.no_decay"})
        grouped_ids |= {id(p) for p in decay + no_decay}

    # 3) Decoder 其他组件 (embed_tokens, embed_positions, layer_norm)
    # Note: embed_tokens might be shared with proj_out, so skip if already grouped
    min_decoder_lr = decoder_base_lr * (layer_decay ** (num_decoder_layers - 1))
    for comp_name in ["embed_tokens", "embed_positions", "layer_norm"]:
        if hasattr(model.model.decoder, comp_name):
            comp = getattr(model.model.decoder, comp_name)
            if comp is None:
                continue
            decay, no_decay = [], []
            for n, p in comp.named_parameters():
                if not p.requires_grad or id(p) in grouped_ids:
                    continue
                (no_decay if is_no_decay(n) else decay).append(p)
            if decay:
                groups.append({"params": decay, "lr": min_decoder_lr, "weight_decay": weight_decay,
                               "name": f"decoder.{comp_name}.decay"})
            if no_decay:
                groups.append({"params": no_decay, "lr": min_decoder_lr, "weight_decay": 0.0,
                               "name": f"decoder.{comp_name}.no_decay"})
            grouped_ids |= {id(p) for p in decay + no_decay}

    # 4) Encoder 层 LLRD (只在 Stage B 中使用)
    num_encoder_layers = len(model.model.encoder.layers)
    for layer_idx, layer in enumerate(model.model.encoder.layers):
        layer_lr = encoder_base_lr * (layer_decay ** (num_encoder_layers - 1 - layer_idx))
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

    # 5) Encoder 其他组件 (embed_positions, conv1, conv2, layer_norm)
    min_encoder_lr = encoder_base_lr * (layer_decay ** (num_encoder_layers - 1))
    for comp_name in ["embed_positions", "conv1", "conv2", "layer_norm"]:
        if hasattr(model.model.encoder, comp_name):
            comp = getattr(model.model.encoder, comp_name)
            if comp is None:
                continue
            decay, no_decay = [], []
            for n, p in comp.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if is_no_decay(n) else decay).append(p)
            if decay:
                groups.append({"params": decay, "lr": min_encoder_lr, "weight_decay": weight_decay,
                               "name": f"encoder.{comp_name}.decay"})
            if no_decay:
                groups.append({"params": no_decay, "lr": min_encoder_lr, "weight_decay": 0.0,
                               "name": f"encoder.{comp_name}.no_decay"})
            grouped_ids |= {id(p) for p in decay + no_decay}

    # 6) 兜底：任何遗漏的可训练参数
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in grouped_ids:
            continue
        (no_decay if is_no_decay(n) else decay).append(p)
    if decay:
        groups.append({"params": decay, "lr": decoder_base_lr, "weight_decay": weight_decay, "name": "other.decay"})
    if no_decay:
        groups.append({"params": no_decay, "lr": decoder_base_lr, "weight_decay": 0.0, "name": "other.no_decay"})

    return groups


class FreezeEncoderCallback(TrainerCallback):
    """
    Callback to handle progressive unfreezing of encoder.
    Stage A: Decoder warmup - freeze encoder, train decoder + lm_head
    Stage B: Full model finetune - unfreeze encoder  
    """
    
    def __init__(self, freeze_encoder_steps: int, model, trainer=None):
        self.freeze_encoder_steps = freeze_encoder_steps
        self.model = model
        self.trainer = trainer
        self.encoder_frozen = freeze_encoder_steps > 0
        self.stage_b_start_step = None  # Track when Stage B starts
        
        if self.encoder_frozen:
            # Stage A: Decoder warmup - freeze encoder
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
            # Enable decoder and lm_head
            for param in self.model.model.decoder.parameters():
                param.requires_grad = True
            for param in self.model.proj_out.parameters():
                param.requires_grad = True
            logger.info("Stage A: Decoder warmup - freezing encoder, training decoder + lm_head only")
            self.model.model.encoder.gradient_checkpointing = False
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.encoder_frozen and state.global_step >= self.freeze_encoder_steps:
            # Stage B: Unfreeze encoder and rebuild optimizer
            logger.info(f"\nStage B: Unfreezing encoder at step {state.global_step}")
            
            # Record when Stage B starts
            self.stage_b_start_step = state.global_step
            
            # Unfreeze encoder
            for param in self.model.model.encoder.parameters():
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


class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Seq2Seq trainer that logs predictions during evaluation and supports layer-wise learning rate decay.
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
        Override to create optimizer with layer-wise learning rate decay for Whisper.
        """
        if self.optimizer is None:
            if self.use_layerwise_lr_decay:
                # Check if encoder is frozen (Stage A: Decoder warmup)
                encoder_frozen = not any(p.requires_grad for p in self.model.model.encoder.parameters())
                
                if encoder_frozen:
                    # Stage A: Decoder warmup - only decoder + lm_head trainable
                    logger.info("Creating optimizer for Stage A (Decoder warmup)")
                    parameter_groups = get_whisper_parameter_groups(
                        self.model,
                        encoder_base_lr=3e-5,  # Won't be used since encoder is frozen
                        decoder_base_lr=3e-4,  # Higher LR for decoder during warmup
                        lm_head_lr=3e-4,       # Same high LR for lm_head
                        layer_decay=0.95,
                        weight_decay=0.01
                    )
                else:
                    # Stage B: Full model fine-tuning with LLRD
                    logger.info("Creating optimizer for Stage B (full model with LLRD)")
                    parameter_groups = get_whisper_parameter_groups(
                        self.model,
                        encoder_base_lr=3e-5,  # Lower LR for encoder
                        decoder_base_lr=1e-4,  # Normal LR for decoder
                        lm_head_lr=3e-4,       # Higher LR for lm_head
                        layer_decay=0.95,
                        weight_decay=0.01
                    )
                
                # Log parameter groups
                logger.info("Parameter groups created:")
                for group in parameter_groups:
                    num_params = sum(p.numel() for p in group['params'])
                    if num_params > 0:  # Only log non-empty groups
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
            encoder_frozen = not any(p.requires_grad for p in self.model.model.encoder.parameters())
            
            if encoder_frozen:
                # Stage A: Decoder warmup
                # Use 10% of Stage A steps for warmup (300 steps for 3000 total)
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
        default="openai/whisper-medium",
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
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    subfolder: str = field(
        default="",
        metadata={
            "help": "In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can specify the folder name here."
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
    freeze_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the encoder layers of the model."}
    )
    freeze_decoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the decoder layers of the model."}
    )
    freeze_encoder_steps: int = field(
        default=0,
        metadata={"help": "Number of steps to warm up decoder/CTC before unfreezing encoder. 0 means no warmup."}
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                "will always be a token of index 123."
            )
        },
    )
    suppress_tokens: List[int] = field(
        default=None, metadata={"help": "A list of tokens that will be suppressed at generation."}
    )
    apply_spec_augment: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply *SpecAugment* data augmentation to the input features. This is currently only relevant for Wav2Vec2, HuBERT, WavLM and Whisper models."
        },
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
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="transcription",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=30.0,
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
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    do_lower_case: bool = field(
        default=False,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    language: str = field(
        default="nl",
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
    wandb_project: str = field(
        default="ssl",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_run_name: Optional[str] = field(
        default="whisper-cgn",
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
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`CustomProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: CustomProcessor
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = "input_features"
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        # Use tokenizer's pad method instead of manual padding
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt"
        )
        
        labels = labels_batch["input_ids"]
        
        # replace padding with -100 to ignore loss correctly
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

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


def patch_whisper_generation_to_basic(model: WhisperForConditionalGeneration) -> WhisperForConditionalGeneration:
    """Override Whisper's custom generation with the base GenerationMixin implementation.
    Disables Whisper-specific forced/suppressed tokens so decoding becomes a plain beam search
    starting from decoder_start_token_id using input_features only.
    """
    # Disable Whisper-specific constraints on both config and generation_config
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = None
        if getattr(model.generation_config, "decoder_start_token_id", None) is None:
            model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = None

    # Bind base GenerationMixin methods to bypass Whisper's overridden generate/logits-processor
    model.generate = GenerationMixin.generate.__get__(model, model.__class__)
    model._get_logits_processor = GenerationMixin._get_logits_processor.__get__(model, model.__class__)

    return model


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_speech_recognition_seq2seq", model_args, data_args)

    # 2. Setup loggin
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

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
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

    # 4. Load dataset
    raw_datasets = DatasetDict()
    
    if training_args.do_train:
        raw_datasets["train"] = load_cgn_data(data_args.cgn_data_dir, data_args.train_split_name)
        
    if training_args.do_eval:
        raw_datasets["eval"] = load_cgn_data(data_args.cgn_data_dir, data_args.eval_split_name)
        
    if training_args.do_predict:
        raw_datasets["test"] = load_cgn_data(data_args.cgn_data_dir, data_args.test_split_name)

    # 5. Load pretrained model and components
    
    # Initialize custom tokenizer
    tokenizer = BPEAEDTokenizer(
        vocab_file=model_args.tokenizer_vocab_file,
        merges_file=model_args.tokenizer_merges_file,
    )
    
    # Load feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Load model config
    config = WhisperConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Update config with custom tokenizer settings
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.decoder_start_token_id = tokenizer.bos_token_id
    
    # Set task and language
    config.forced_decoder_ids = None
    config.suppress_tokens = model_args.suppress_tokens
    config.use_cache = False if training_args.gradient_checkpointing else True
    
    # Apply SpecAugment if specified
    if model_args.apply_spec_augment:
        config.apply_spec_augment = model_args.apply_spec_augment
    
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        subfolder=model_args.subfolder,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=True,  # Important: Allow resizing of embeddings
    )
    
    # Freeze encoder/decoder if requested
    if model_args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False
    
    if model_args.freeze_decoder:
        for param in model.model.decoder.parameters():
            param.requires_grad = False
        model.model.decoder.gradient_checkpointing = False
    
    # Ensure we use basic, task/language-agnostic generation
    patch_whisper_generation_to_basic(model)

    model.generation_config = GenerationConfig(
        forced_decoder_ids=None,
        suppress_tokens=None,
        decoder_start_token_id=model.config.decoder_start_token_id,
        num_beams=training_args.generation_num_beams,
        max_length=training_args.generation_max_length,
        early_stopping=True,
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=3,
    )
        # Create custom processor with both feature extractor and tokenizer
    processor = CustomProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # 6. Preprocessing the datasets
    def prepare_dataset(batch):
        # Load audio from bytes
        import soundfile as sf
        import io
        
        audio_bytes = batch[data_args.audio_column_name]["bytes"]
        audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Extract input features
        inputs = feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="np"
        )

        batch["input_features"] = inputs.input_features[0]
        batch["input_length"] = len(audio_array)
        
        # Encode labels using custom tokenizer with special tokens
        transcription = batch[data_args.text_column_name]
        if data_args.do_lower_case:
            transcription = transcription.lower()
        
        batch["labels"] = tokenizer.encode(transcription, add_special_tokens=True)
        
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
    
    if training_args.do_train:
        vectorized_datasets["train"] = vectorized_datasets["train"].filter(
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

    # 7. Define data collator
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and training_args.do_train
    )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )
    # 8. Metrics
    metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        
        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = metric.compute(predictions=pred_str, references=label_str)
        
        # Store predictions for logging by WhisperSeq2SeqTrainer
        # Don't return them in metrics to avoid logging format errors
        compute_metrics.pred_str = pred_str
        compute_metrics.label_str = label_str
        
        return {"wer": wer}
    
    # 9. Initialize our trainer
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
    
    # Initialize our custom Seq2SeqTrainer with prediction logging
    trainer = WhisperSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=callbacks,
        num_examples_to_log=data_args.num_examples_to_log,
        use_layerwise_lr_decay=True,  # Enable layer-wise learning rate decay
        freeze_encoder_steps=model_args.freeze_encoder_steps,  # Pass freeze steps to trainer
    )
    
    # Set trainer reference in callback for optimizer rebuilding
    if freeze_callback:
        freeze_callback.trainer = trainer
    
    # 10. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the feature extractor too
        
        metrics = train_result.metrics
        if data_args.max_train_samples:
            metrics["train_samples"] = data_args.max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # 11. Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        if data_args.max_eval_samples:
            metrics["eval_samples"] = data_args.max_eval_samples
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # 12. Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(
            vectorized_datasets["test"],
            metric_key_prefix="test",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        metrics = predict_results.metrics
        if data_args.max_test_samples:
            metrics["test_samples"] = data_args.max_test_samples
        
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        
        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))
    
    # 13. Write model card and save tokenizer vocabulary
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "tags": "whisper-event",
    }
    if data_args.language is not None:
        kwargs["language"] = data_args.language
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    
    # Save tokenizer vocabulary
    if training_args.output_dir is not None:
        tokenizer.save_vocabulary(training_args.output_dir)
    
    return results


if __name__ == "__main__":
    main()