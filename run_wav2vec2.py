#!/usr/bin/env python
# coding=utf-8
"""Fine-tuning Wav2Vec2 model for CTC-based ASR using CGN dataset and custom BPE tokenizer"""

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
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from pathlib import Path

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
    ProcessorMixin,
    PretrainedConfig,
)
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


class FreezeEncoderCallback(TrainerCallback):
    """
    Callback to handle progressive unfreezing of encoder.
    First warmup the CTC layer for freeze_encoder_steps, then unfreeze encoder.
    """
    
    def __init__(self, freeze_encoder_steps: int, model):
        self.freeze_encoder_steps = freeze_encoder_steps
        self.model = model
        self.encoder_frozen = freeze_encoder_steps > 0
        
        if self.encoder_frozen:
            # Initially freeze encoder
            self.model.freeze_feature_encoder()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.encoder_frozen and state.global_step >= self.freeze_encoder_steps:
            # Unfreeze encoder after warmup steps
            for param in self.model.wav2vec2.feature_extractor.parameters():
                param.requires_grad = True
            for param in self.model.wav2vec2.feature_projection.parameters():
                param.requires_grad = True
            for param in self.model.wav2vec2.encoder.parameters():
                param.requires_grad = True
            self.encoder_frozen = False
            print(f"\nUnfreezing encoder at step {state.global_step}")
        return control


class CTCTrainer(Trainer):
    """
    Custom trainer that logs predictions during evaluation.
    """
    
    def __init__(self, *args, num_examples_to_log=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_examples_to_log = num_examples_to_log
        self.eval_predictions = None
        self.eval_labels = None
    
    def compute_metrics(self, eval_pred):
        """Override to store predictions for logging."""
        if self.args.predict_with_generate:
            # Handle generation case if needed
            return super().compute_metrics(eval_pred)
        
        # Call the original compute_metrics
        metrics = self.compute_metrics_fn(eval_pred) if self.compute_metrics_fn is not None else {}
        
        # Extract and log predictions
        if "pred_str" in metrics and "label_str" in metrics:
            self.eval_predictions = metrics.pop("pred_str")
            self.eval_labels = metrics.pop("label_str")
            
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
        default="wav2vec2-cgn",
        metadata={"help": "Weights & Biases run name. If not specified, will be auto-generated."}
    )
    num_examples_to_log: int = field(
        default=5,
        metadata={"help": "Number of prediction examples to log during evaluation."}
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
    """Load CGN data from parquet files."""
    parquet_files = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found for split '{split}' in {data_dir}")
    
    # Read all parquet files for this split
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)
    
    # Concatenate all dataframes
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(full_df)
    
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
    tokenizer = BPECTCTokenizer(
        vocab_file=model_args.tokenizer_vocab_file,
        merges_file=model_args.tokenizer_merges_file,
    )

    # Load feature extractor (using Wav2Vec2FeatureExtractor)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )
    
    # Create custom processor with both feature extractor and tokenizer
    processor = CustomProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load model config
    config = Wav2Vec2Config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )
 
    # Update config with tokenizer-related settings
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    
    # # Update config with model-specific settings
    # config.attention_dropout = model_args.attention_dropout
    # config.activation_dropout = model_args.activation_dropout
    # config.feat_proj_dropout = model_args.feat_proj_dropout
    # config.hidden_dropout = model_args.hidden_dropout
    # config.final_dropout = model_args.final_dropout
    # config.mask_time_prob = model_args.mask_time_prob
    # config.mask_time_length = model_args.mask_time_length
    # config.mask_feature_prob = model_args.mask_feature_prob
    # config.mask_feature_length = model_args.mask_feature_length
    # config.layerdrop = model_args.layerdrop
    # config.ctc_loss_reduction = model_args.ctc_loss_reduction
    # config.ctc_zero_infinity = model_args.ctc_zero_infinity

    # Load model
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
        batch["input_length"] = len(inputs.input_values[0])
        
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

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        # Return metrics - note that pred_str and label_str will be logged by callback
        return {"wer": wer, "pred_str": pred_str, "label_str": label_str}

    # Set up callbacks
    callbacks = []
    if model_args.freeze_encoder_steps > 0:
        callbacks.append(FreezeEncoderCallback(model_args.freeze_encoder_steps, model))
    
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
    )

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

    # Save tokenizer vocabulary
    if training_args.output_dir is not None:
        tokenizer.save_vocabulary(training_args.output_dir)

    return results


if __name__ == "__main__":
    main()