#!/usr/bin/env python3
"""
Generate ASR predictions using finetuned HuBERT model on CGN test dataset.
"""

import os
import sys
import torch
import numpy as np
import datasets
import argparse
import soundfile as sf
import io
from pathlib import Path
from tqdm import tqdm
from transformers import (
    HubertForCTC,
    Wav2Vec2FeatureExtractor,
    ProcessorMixin
)

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from run_hubert import load_cgn_data
sys.path.append(str(Path(__file__).parent.parent))
from tokenizer.bpe_ctc_tokenizer import BPECTCTokenizer


class CustomProcessor(ProcessorMixin):
    """Custom processor combining Wav2Vec2FeatureExtractor and BPECTCTokenizer."""
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "Wav2Vec2FeatureExtractor"
    tokenizer_class = "PreTrainedTokenizer"
    
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
    
    def __call__(self, *args, **kwargs):
        if "audio" in kwargs or (args and not isinstance(args[0], str)):
            return self.feature_extractor(*args, **kwargs)
        else:
            return self.tokenizer(*args, **kwargs)
    
    def batch_decode(self, *args, **kwargs):
        # Add group_tokens=True for CTC models to merge repeated tokens
        if 'group_tokens' not in kwargs:
            kwargs['group_tokens'] = True
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        if 'group_tokens' not in kwargs:
            kwargs['group_tokens'] = True
        return self.tokenizer.decode(*args, **kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        tokenizer = BPECTCTokenizer.from_pretrained(pretrained_model_name_or_path)
        return cls(feature_extractor, tokenizer)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ASR predictions using finetuned HuBERT model on CGN test dataset")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/xinyu/data1/HuggingFace/datasets/CGN/data",
        help="Path to CGN dataset directory containing parquet files"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to finetuned HuBERT model directory"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for predictions and labels files"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent / "outputs"
    else:
        args.output_dir = Path(args.output_dir)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output files
    predictions_file = args.output_dir / "pred.txt"
    labels_file = args.output_dir / "label.txt"
    
    print(f"Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Model: {args.model_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    print(f"Loading model and processor from: {args.model_dir}")
    
    # Load finetuned model
    model = HubertForCTC.from_pretrained(args.model_dir)
    
    # Load processor (feature extractor + custom tokenizer)
    # Check if vocab.json is in the model directory
    if os.path.exists(os.path.join(args.model_dir, "vocab.json")):
        processor = CustomProcessor.from_pretrained(args.model_dir)
    else:
        processor = CustomProcessor.from_pretrained(os.path.dirname(args.model_dir))
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    
    # Load CGN test dataset
    print("Loading CGN test dataset...")
    test_dataset = load_cgn_data(args.data_dir, "test")
    print(f"Loaded {len(test_dataset)} test samples")
    
    predictions = []
    labels = []
    
    print("Generating predictions...")
    
    # Process in batches for efficiency
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), args.batch_size)):
            batch_end = min(i + args.batch_size, len(test_dataset))
            batch_indices = range(i, batch_end)
            
            # Process audio from bytes
            audio_arrays = []
            sampling_rates = []
            speech_ids = []
            transcriptions = []
            
            for idx in batch_indices:
                sample = test_dataset[idx]
                audio_bytes = sample["audio"]["bytes"]
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
                audio_arrays.append(audio_array)
                sampling_rates.append(sampling_rate)
                speech_ids.append(sample["id"])
                transcriptions.append(sample["transcription"])
            
            # Process audio with feature extractor
            inputs = processor.feature_extractor(
                audio_arrays, 
                sampling_rate=sampling_rates[0],  # Assuming all have same rate
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') else None
            
            # Get model outputs
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask
            )
            
            # Get predictions from logits
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode predictions
            batch_predictions = processor.batch_decode(predicted_ids.cpu().numpy(), skip_special_tokens=True)
            
            # Get speech IDs and transcriptions
            for j in range(len(batch_predictions)):
                speech_id = speech_ids[j]
                prediction = batch_predictions[j].strip()
                label = transcriptions[j].strip()
                
                predictions.append(f"{speech_id} {prediction}")
                labels.append(f"{speech_id} {label}")
    
    # Save predictions to file
    print(f"Saving predictions to {predictions_file}")
    with open(predictions_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")
    
    # Save labels to file
    print(f"Saving labels to {labels_file}")
    with open(labels_file, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(label + "\n")
    
    print("Generation complete!")
    print(f"Predictions saved to: {predictions_file}")
    print(f"Labels saved to: {labels_file}")
    
    # Calculate and print basic statistics
    print(f"\nStatistics:")
    print(f"Total samples processed: {len(predictions)}")
    
    # Sample output for verification
    print("\nSample outputs (first 3):")
    for i in range(min(3, len(predictions))):
        print(f"Prediction: {predictions[i]}")
        print(f"Label: {labels[i]}")
        print("-" * 50)


if __name__ == "__main__":
    main()