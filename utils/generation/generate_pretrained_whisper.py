#!/usr/bin/env python3
"""
Generate ASR predictions using official Whisper pretrained model on CGN test dataset.
"""

import os
import sys
import torch
import datasets
import argparse
import soundfile as sf
import io
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Add parent directory to path to import load_cgn_data
sys.path.append(str(Path(__file__).parent.parent.parent))
from run_whisper import load_cgn_data


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Whisper ASR predictions on CGN test dataset")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/xinyu/data1/HuggingFace/datasets/CGN/data",
        help="Path to CGN dataset directory containing parquet files"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default="openai/whisper-medium",
        help="Path to Whisper model or HuggingFace model name (e.g., openai/whisper-small)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--num_beams",
        type=int,
        default=3,
        help="Number of beams for beam search"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=225,
        help="Maximum length of generated sequences"
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
    print(f"  Num beams: {args.num_beams}")
    print(f"  Max length: {args.max_length}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    print(f"Loading model and processor from: {args.model_dir}")
    
    # Load official Whisper processor and model
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    
    # Load CGN test dataset
    print("Loading CGN test dataset...")
    test_dataset = load_cgn_data(args.data_dir, "test")
    print(f"Loaded {len(test_dataset)} test samples")
    
    # Prepare for generation
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="nl", task="transcribe")
    
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
            
            # Ensure all samples have the same sampling rate (16000 for Whisper)
            inputs = processor(
                audio_arrays, 
                sampling_rate=sampling_rates[0],  # Assuming all have same rate
                return_tensors="pt"
            )
            # Move inputs to device
            input_features = inputs.input_features.to(device)
            
            # Generate predictions
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=args.max_length,
                num_beams=args.num_beams,
                temperature=0.0,  # Deterministic generation
                do_sample=False,
                return_timestamps=False
            )
            
            # Decode predictions
            batch_predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            # Get speech IDs and transcriptions
            for j in range(len(batch_predictions)):
                speech_id = speech_ids[j]
                prediction = batch_predictions[j].strip()
                label = transcriptions[j].strip()
                # print(f"Speech ID: {speech_id} {prediction}")
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