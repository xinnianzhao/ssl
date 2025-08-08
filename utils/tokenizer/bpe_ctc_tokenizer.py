#!/usr/bin/env python3
"""BPE CTC Tokenizer for speech recognition models."""

import json
import os
import re
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import unicodedata
from transformers import PreTrainedTokenizer

class BPECTCTokenizer(PreTrainedTokenizer):
    """
    A tokenizer for CTC models that uses BPE tokens.
    
    This tokenizer combines BPE tokenization with CTC decoding, where:
    - BPE tokens are used as the vocabulary
    - CTC decoding removes repeated tokens
    - BPE decoding converts token sequences back to text
    
    Args:
        vocab_file (str): Path to the vocab.json file containing BPE token mappings.
        pad_token (str, optional): Padding token. Defaults to "<pad>".
        unk_token (str, optional): Unknown token. Defaults to "<unk>".
        bos_token (str, optional): Beginning of sentence token. Defaults to "<s>".
        eos_token (str, optional): End of sentence token. Defaults to "</s>".
    """
    
    def __init__(
        self,
        vocab_file: str,
        merges_file: Optional[str] = None,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        **kwargs
    ):
        # Store file paths for saving
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        
        # Special tokens to add at the beginning of vocab
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]
        
        # Load BPE merges if provided
        if merges_file and os.path.exists(merges_file):
            with open(merges_file, encoding="utf-8") as f:
                merges = f.read().split("\n")
                # Skip the header if present
                if merges[0].startswith("#"):
                    merges = merges[1:]
                # Filter empty lines
                merges = [m for m in merges if m.strip()]
                # Parse merges
                self.bpe_ranks = {tuple(merge.split()): i for i, merge in enumerate(merges)}
        else:
            self.bpe_ranks = {}
        
        # Cache for BPE encoding
        self.cache = {}
        
        # Load vocab from file
        with open(vocab_file, encoding="utf-8") as f:
            original_vocab = json.load(f)
        
        # Create encoder with special tokens at the beginning
        self.encoder = {}
        
        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.encoder[token] = i
        
        # Add original vocab tokens, shifting their IDs
        offset = len(self.special_tokens)
        for token, idx in original_vocab.items():
            self.encoder[token] = idx + offset
        
        # Create decoder (inverse mapping)
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # Get special token IDs
        self.pad_token_id = self.encoder[pad_token]
        self.unk_token_id = self.encoder[unk_token]
        self.bos_token_id = self.encoder[bos_token]
        self.eos_token_id = self.encoder[eos_token]
        
        # BPE pattern for detokenization
        self.bpe_pattern = re.compile(r'Ġ')
        
        # Pattern for tokenization (similar to GPT2/Whisper)
        # This pattern splits text into words and punctuation
        # Using a simpler pattern that works with standard re module
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-ZÀ-ÿĀ-žА-я]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""")
        
        # Initialize parent class
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs
        )
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.encoder)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the full vocabulary dictionary."""
        return dict(self.encoder)
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its corresponding ID."""
        return self.encoder.get(token, self.unk_token_id)
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its corresponding token."""
        return self.decoder.get(index, self.unk_token)
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert token(s) to their corresponding IDs."""
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert ID(s) to their corresponding tokens."""
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(idx) for idx in ids]
    
    def _ctc_decode(self, token_ids: List[int]) -> List[int]:
        """
        Perform CTC decoding by removing consecutive duplicate tokens.
        Also removes pad tokens which are used as CTC blank tokens.
        
        Args:
            token_ids: List of token IDs from model output.
            
        Returns:
            List of token IDs with duplicates removed.
        """
        if len(token_ids) == 0:
            return []
        
        # Group consecutive identical tokens
        grouped = []
        for token_id, _ in groupby(token_ids):
            # Skip pad tokens (used as CTC blank)
            if token_id != self.pad_token_id:
                grouped.append(token_id)
        
        return grouped
    
    def _bpe_decode(self, tokens: List[str]) -> str:
        """
        Convert BPE tokens back to text.
        
        BPE tokens use 'Ġ' to represent spaces at the beginning of words.
        This method converts the tokens back to proper text.
        
        Args:
            tokens: List of BPE tokens.
            
        Returns:
            Decoded text string.
        """
        if not tokens:
            return ""
        
        # Join tokens
        text = "".join(tokens)
        
        # Replace BPE space marker with actual space
        text = self.bpe_pattern.sub(" ", text)
        
        # Clean up any leading/trailing spaces
        text = text.strip()
        
        return text
    
    def decode(
        self,
        token_ids: Union[int, List[int], np.ndarray],
        skip_special_tokens: bool = True,
        group_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        This method performs:
        1. CTC decoding (removing duplicate consecutive tokens) if group_tokens=True
        2. Converts IDs to tokens
        3. Filters special tokens if skip_special_tokens=True
        4. BPE decoding to convert tokens to text
        
        Args:
            token_ids: Token IDs to decode. Can be int, list of ints, or numpy array.
            skip_special_tokens: Whether to remove special tokens from output.
            group_tokens: Whether to perform CTC decoding (remove duplicates).
            
        Returns:
            Decoded text string.
        """
        # Convert to list if necessary
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # CTC decode if requested
        if group_tokens:
            token_ids = self._ctc_decode(token_ids)
        
        # Convert IDs to tokens
        tokens = self.convert_ids_to_tokens(token_ids)
        
        # Filter special tokens if requested
        if skip_special_tokens:
            special_tokens_set = set(self.special_tokens)
            tokens = [token for token in tokens if token not in special_tokens_set]
        
        # BPE decode to get final text
        text = self._bpe_decode(tokens)
        
        return text
    
    def batch_decode(
        self,
        sequences: Union[List[List[int]], np.ndarray],
        skip_special_tokens: bool = True,
        group_tokens: bool = True,
    ) -> List[str]:
        """
        Decode multiple sequences of token IDs.
        
        Args:
            sequences: List of sequences or 2D numpy array.
            skip_special_tokens: Whether to remove special tokens from output.
            group_tokens: Whether to perform CTC decoding (remove duplicates).
            
        Returns:
            List of decoded text strings.
        """
        if isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()
        
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens, group_tokens=group_tokens)
            for seq in sequences
        ]
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary to a file.
        
        Args:
            save_directory: Directory to save the vocabulary file.
            filename_prefix: Optional prefix for the filename.
            
        Returns:
            Tuple containing the path to the saved vocabulary file.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        
        vocab_filename = "vocab.json"
        if filename_prefix:
            vocab_filename = f"{filename_prefix}-{vocab_filename}"
        
        vocab_file = os.path.join(save_directory, vocab_filename)
        
        # Save the vocabulary (without special tokens at the beginning for compatibility)
        # Create original vocab format
        original_vocab = {}
        offset = len(self.special_tokens)
        for token, idx in self.encoder.items():
            if token not in self.special_tokens:
                original_vocab[token] = idx - offset
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(original_vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load tokenizer from a pretrained model."""
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
        merges_file = os.path.join(pretrained_model_name_or_path, "merges.txt")
        
        if not os.path.exists(merges_file):
            merges_file = None
            
        return cls(vocab_file=vocab_file, merges_file=merges_file, **kwargs)
    
    def get_pairs(self, word: Tuple[str, ...]) -> set:
        """Return set of symbol pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def bpe(self, token: str) -> str:
        """Apply BPE to a token."""
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = self.get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            # Find the highest priority merge
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        
        word = ' '.join(word)
        self.cache[token] = word
        return word
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into BPE tokens."""
        bpe_tokens = []
        
        # Find all words/tokens using regex pattern
        for token in re.findall(self.pat, text):
            # For each token, apply BPE
            # Note: In the original Llama tokenizer, space at the beginning of words is represented as Ġ
            token = token.replace(' ', 'Ġ')
            
            # Apply BPE if we have merges
            if self.bpe_ranks:
                bpe_output = self.bpe(token)
                # Split the BPE output and add to tokens
                bpe_tokens.extend(bpe_output.split(' '))
            else:
                # If no BPE merges, just add the token as is
                bpe_tokens.append(token)
        
        return bpe_tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)
    
    def __call__(self, text: Union[str, List[str]], padding: Union[bool, str] = False, return_tensors: Optional[str] = None, **kwargs) -> Dict[str, Union[List[int], 'torch.Tensor']]:
        """
        Tokenize text to token IDs.
        
        Args:
            text: Text to tokenize.
            padding: Whether to pad the sequences.
            return_tensors: 'pt' for PyTorch tensors, None for lists.
            
        Returns:
            Dictionary with 'input_ids' containing token IDs.
        """
        if isinstance(text, str):
            text = [text]
        
        input_ids = []
        for t in text:
            ids = self.encode(t)
            input_ids.append(ids)
        
        batch_output = {"input_ids": input_ids}
        
        # Use the parent's pad method if padding is requested
        if padding:
            batch_output = self.pad(batch_output, padding=padding, return_tensors=return_tensors, **kwargs)
        elif return_tensors == 'pt':
            import torch
            batch_output = {k: torch.tensor(v, dtype=torch.long) for k, v in batch_output.items()}
        
        return batch_output


if __name__ == "__main__":
    # Example usage
    tokenizer = BPECTCTokenizer("vocab.json", "merges.txt")
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"Unk token ID: {tokenizer.unk_token_id}")
    print(f"Number of BPE merges: {len(tokenizer.bpe_ranks)}")
    
    # Test encoding
    test_text = "en dus nu een beetje uw werk meer uw hobby geworden ook"
    print(f"\nTest text: {test_text}")
    
    # Tokenize the text
    tokens = tokenizer.tokenize(test_text)
    print(f"Tokens: {tokens}")
    
    # Encode to IDs
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    
    # Simulate CTC output with repeated tokens
    ctc_output = []
    for token_id in token_ids:
        # Add some repetitions and blanks
        ctc_output.extend([0, token_id, token_id, token_id])
    ctc_output.append(0)  # End with blank
    
    print(f"\nSimulated CTC output (with repeats): {ctc_output}")
    
    # Decode back
    decoded = tokenizer.decode(ctc_output)
    print(f"CTC decoded: {decoded}")
    
    # Test without CTC grouping
    decoded_no_ctc = tokenizer.decode(ctc_output, group_tokens=False)
    print(f"Decoded without CTC: {decoded_no_ctc}")