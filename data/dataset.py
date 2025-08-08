import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import sentencepiece as spm



class CGNDataset(Dataset):
    """
    Dataset class for CGN (Corpus Gesproken Nederlands) Dutch speech corpus
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_path: Optional[str] = None,
        sampling_rate: int = 16000,
        max_duration: float = 20.0,
        model_type: str = "wav2vec2",
        use_whisper_tokenizer: bool = False
    ):
        """
        Args:
            data_path: Path to text file with format: audio_path|transcript
            tokenizer_path: Path to SentencePiece model file
            sampling_rate: Target sampling rate for audio
            max_duration: Maximum audio duration in seconds
            model_type: Type of model (affects preprocessing)
            use_whisper_tokenizer: Use Whisper's tokenizer instead of custom
        """
        self.sampling_rate = sampling_rate
        self.max_duration = max_duration
        self.model_type = model_type
        self.use_whisper_tokenizer = use_whisper_tokenizer
        
        # Load data
        self.data = self._load_data(data_path)
        
        # Initialize tokenizer
        if use_whisper_tokenizer:
            from transformers import WhisperTokenizer
            self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium")
        elif tokenizer_path:
            self.sp_model = spm.SentencePieceProcessor(model_file=tokenizer_path)
            self.vocab_size = self.sp_model.get_piece_size()
        else:
            self.sp_model = None
            self.vocab_size = None
            
    def _load_data(self, data_path: str) -> List[Dict[str, str]]:
        """Load data from text file"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    audio_path, transcript = line.split('|', 1)
                    data.append({
                        'audio_path': audio_path.strip(),
                        'transcript': transcript.strip()
                    })
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio"""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sampling_rate
            )
            waveform = resampler(waveform)
        
        # Truncate or pad to max duration
        max_samples = int(self.max_duration * self.sampling_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        
        return waveform.squeeze(0)
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text using appropriate tokenizer"""
        if self.use_whisper_tokenizer:
            # Whisper tokenizer
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True
            ).input_ids.squeeze(0)
        elif self.sp_model:
            # SentencePiece tokenizer
            token_ids = self.sp_model.encode(text, out_type=int)
            tokens = torch.tensor(token_ids, dtype=torch.long)
        else:
            # Return empty tensor if no tokenizer
            tokens = torch.tensor([], dtype=torch.long)
        
        return tokens
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        item = self.data[idx]
        
        # Load audio
        waveform = self._load_audio(item['audio_path'])
        
        # Tokenize text
        labels = self._tokenize_text(item['transcript'])
        
        return {
            'input_values': waveform,
            'labels': labels,
            'transcript': item['transcript'],
            'audio_path': item['audio_path']
        }


class DataCollatorCTC:
    """
    Data collator for CTC models (Wav2Vec2, HuBERT, WavLM)
    """
    
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate inputs and labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        # Pad inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        
        # Pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )
        
        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        
        return batch


class DataCollatorWhisper:
    """
    Data collator for Whisper model
    """
    
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract features
        input_features = [{"input_features": self.processor.feature_extractor(
            f["input_values"], 
            sampling_rate=16000
        ).input_features[0]} for f in features]
        
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        # Pad inputs
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )
        
        # Replace padding with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Add decoder_start_token_id
        if (labels[:, 0] != self.decoder_start_token_id).all().cpu().item():
            labels = torch.cat(
                [torch.full((labels.shape[0], 1), self.decoder_start_token_id), labels],
                dim=1
            )
        
        batch["labels"] = labels
        
        return batch


def create_dataloader(
    data_path: str,
    tokenizer_path: Optional[str],
    batch_size: int,
    model_type: str,
    processor=None,
    sampling_rate: int = 16000,
    max_duration: float = 20.0,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for the CGN dataset
    """
    
    # Create dataset
    dataset = CGNDataset(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        sampling_rate=sampling_rate,
        max_duration=max_duration,
        model_type=model_type,
        use_whisper_tokenizer=(model_type == "whisper")
    )
    
    # Create appropriate collator
    if model_type == "whisper":
        collator = DataCollatorWhisper(
            processor=processor,
            decoder_start_token_id=processor.tokenizer.decoder_start_token_id
        )
    else:
        collator = DataCollatorCTC(
            processor=processor,
            padding=True
        )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader