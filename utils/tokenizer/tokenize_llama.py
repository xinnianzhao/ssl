#!/usr/bin/env python3
"""
Tokenize text files using Llama-3.1-8B-Instruct tokenizer and create vocabulary files.
"""

import json
import glob
from collections import Counter
from transformers import AutoTokenizer
import os

def main():
    # Load Llama-3.1-8B-Instruct tokenizer
    print("Loading Llama-3.1-8B-Instruct tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    # Find all txt files in data directory
    data_dir = "/home/xinyu/xinnian/tasks/ssl/data"
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not txt_files:
        print(f"No .txt files found in {data_dir} directory")
        return
    
    print(f"Found {len(txt_files)} text files to process")
    
    # Counter for token frequencies
    token_counter = Counter()
    
    # Process each text file
    for txt_file in txt_files:
        print(f"Processing {txt_file}...")
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split line into speech_id and text
                parts = line.split(' ', 1)
                if len(parts) < 2:
                    continue
                
                speech_id, text = parts
                
                # Tokenize only the text part
                tokens = tokenizer.tokenize(text)
                
                # Count token frequencies
                token_counter.update(tokens)
    
    print(f"Total unique tokens: {len(token_counter)}")
    
    # Sort tokens by frequency (descending) and create vocab dict
    # Format: {token: rank} where rank starts from 1
    sorted_tokens = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
    vocab = {token: rank for rank, (token, freq) in enumerate(sorted_tokens, start=0)}
    
    # Save vocab.json
    print("Saving vocab.json...")
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Create vocab_mapper.json
    # Map from token_id (our vocab rank) to Llama tokenizer's token ID
    print("Creating vocab_mapper.json...")
    vocab_mapper = {}
    
    for token, rank in vocab.items():
        # Get token ID from Llama tokenizer
        # Handle special tokens and regular tokens
        try:
            # Try to get ID directly
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 1:
                vocab_mapper[str(rank)] = token_ids[0]
            else:
                # If token is split into multiple IDs, use convert_tokens_to_ids
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    vocab_mapper[str(rank)] = token_id
        except:
            # If encoding fails, try convert_tokens_to_ids directly
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                vocab_mapper[str(rank)] = token_id
    
    # Save vocab_mapper.json
    print("Saving vocab_mapper.json...")
    with open('vocab_mapper.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_mapper, f, ensure_ascii=False, indent=2)
    
    print("Done!")
    print(f"- vocab.json: {len(vocab)} tokens with ranks")
    print(f"- vocab_mapper.json: {len(vocab_mapper)} token ID mappings")

if __name__ == "__main__":
    main()