#!/usr/bin/env python3
"""
Compare tokenizers from different Llama models to analyze their differences.
"""

import json
import glob
import os
from collections import Counter, defaultdict
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Set

def load_tokenizers():
    """Load all three tokenizers."""
    models = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "ReBatch/Llama-3-8B-dutch", 
        "ChocoLlama/Llama-3-ChocoLlama-8B-instruct"
    ]
    
    tokenizers = {}
    for model in models:
        print(f"Loading tokenizer for {model}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
            tokenizers[model] = tokenizer
            print(f"✓ Successfully loaded {model}")
        except Exception as e:
            print(f"✗ Failed to load {model}: {e}")
            return None
    
    return tokenizers

def compare_vocab_sizes(tokenizers: Dict):
    """Compare vocabulary sizes of different tokenizers."""
    print("\n" + "="*80)
    print("VOCABULARY SIZE COMPARISON")
    print("="*80)
    
    for name, tokenizer in tokenizers.items():
        vocab_size = len(tokenizer.get_vocab())
        print(f"{name.split('/')[-1]:40} : {vocab_size:,} tokens")
    
    return {name: len(tok.get_vocab()) for name, tok in tokenizers.items()}

def compare_special_tokens(tokenizers: Dict):
    """Compare special tokens across tokenizers."""
    print("\n" + "="*80)
    print("SPECIAL TOKENS COMPARISON")
    print("="*80)
    
    special_tokens = {}
    for name, tokenizer in tokenizers.items():
        model_name = name.split('/')[-1]
        special = {
            'bos_token': tokenizer.bos_token,
            'eos_token': tokenizer.eos_token,
            'unk_token': tokenizer.unk_token,
            'pad_token': tokenizer.pad_token,
            'cls_token': getattr(tokenizer, 'cls_token', None),
            'sep_token': getattr(tokenizer, 'sep_token', None),
            'mask_token': getattr(tokenizer, 'mask_token', None),
        }
        special_tokens[model_name] = special
        
        print(f"\n{model_name}:")
        for token_type, token in special.items():
            if token is not None:
                token_id = tokenizer.convert_tokens_to_ids(token) if token else None
                print(f"  {token_type:12} : {token:20} (ID: {token_id})")
    
    return special_tokens

def compare_vocab_overlap(tokenizers: Dict):
    """Compare vocabulary overlap between tokenizers."""
    print("\n" + "="*80)
    print("VOCABULARY OVERLAP ANALYSIS")
    print("="*80)
    
    vocabs = {}
    for name, tokenizer in tokenizers.items():
        model_name = name.split('/')[-1]
        vocabs[model_name] = set(tokenizer.get_vocab().keys())
    
    model_names = list(vocabs.keys())
    
    # Pairwise comparison
    print("\nPairwise vocabulary overlap:")
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:
                common = len(vocabs[model1] & vocabs[model2])
                total = len(vocabs[model1] | vocabs[model2])
                overlap_pct = (common / total) * 100 if total > 0 else 0
                print(f"{model1:30} ∩ {model2:30} : {common:,} tokens ({overlap_pct:.2f}%)")
    
    # Three-way intersection
    common_all = vocabs[model_names[0]] & vocabs[model_names[1]] & vocabs[model_names[2]]
    print(f"\nTokens common to all three models: {len(common_all):,}")
    
    # Unique tokens per model
    print("\nUnique tokens per model:")
    for model in model_names:
        unique = vocabs[model] - (vocabs[model_names[(model_names.index(model)+1)%3]] | 
                                  vocabs[model_names[(model_names.index(model)+2)%3]])
        print(f"{model:40} : {len(unique):,} unique tokens")
        
        # Show examples of unique tokens (first 10)
        if len(unique) > 0:
            examples = list(unique)[:10]
            print(f"  Examples: {examples}")
    
    return vocabs

def tokenize_sample_texts(tokenizers: Dict, num_samples: int = 5):
    """Tokenize sample texts from data files and compare results."""
    print("\n" + "="*80)
    print("TOKENIZATION COMPARISON ON REAL DATA")
    print("="*80)
    
    # Read sample texts
    data_files = [
        "/home/xinyu/xinnian/tasks/ssl/data/test.txt",
        "/home/xinyu/xinnian/tasks/ssl/data/train.txt",
        "/home/xinyu/xinnian/tasks/ssl/data/validation.txt"
    ]
    
    sample_texts = []
    for data_file in data_files:
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            sample_texts.append(parts[1])
            if sample_texts:
                break
    
    if not sample_texts:
        print("No sample texts found in data files")
        return
    
    # Compare tokenization results
    differences = []
    for i, text in enumerate(sample_texts[:num_samples], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        tokenized = {}
        token_ids = {}
        
        for name, tokenizer in tokenizers.items():
            model_name = name.split('/')[-1]
            
            # Tokenize text
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.encode(text, add_special_tokens=False)
            
            tokenized[model_name] = tokens
            token_ids[model_name] = ids
            
            print(f"\n{model_name}:")
            print(f"  Num tokens: {len(tokens)}")
            print(f"  First 10 tokens: {tokens[:10]}")
            print(f"  First 10 IDs: {ids[:10]}")
        
        # Check if tokenization differs
        model_names = list(tokenized.keys())
        if not (tokenized[model_names[0]] == tokenized[model_names[1]] == tokenized[model_names[2]]):
            differences.append({
                'text': text,
                'tokenizations': tokenized,
                'token_ids': token_ids
            })
    
    return differences

def analyze_tokenization_differences(differences: List[Dict]):
    """Analyze and summarize tokenization differences."""
    if not differences:
        print("\n✓ All models produce identical tokenization for the sample texts!")
        return
    
    print("\n" + "="*80)
    print("DETAILED DIFFERENCE ANALYSIS")
    print("="*80)
    
    print(f"\nFound {len(differences)} texts with different tokenizations:")
    
    for i, diff in enumerate(differences, 1):
        print(f"\n--- Difference {i} ---")
        print(f"Text: {diff['text'][:100]}{'...' if len(diff['text']) > 100 else ''}")
        
        # Compare token counts
        print("\nToken count comparison:")
        for model, tokens in diff['tokenizations'].items():
            print(f"  {model:40} : {len(tokens)} tokens")
        
        # Find where tokenizations differ
        model_names = list(diff['tokenizations'].keys())
        tokens_lists = [diff['tokenizations'][m] for m in model_names]
        
        # Find first divergence point
        min_len = min(len(t) for t in tokens_lists)
        divergence_idx = None
        
        for idx in range(min_len):
            if not all(tokens_lists[0][idx] == t[idx] for t in tokens_lists[1:]):
                divergence_idx = idx
                break
        
        if divergence_idx is not None:
            print(f"\nFirst divergence at token position {divergence_idx}:")
            for model in model_names:
                tokens = diff['tokenizations'][model]
                if divergence_idx < len(tokens):
                    context_start = max(0, divergence_idx - 2)
                    context_end = min(len(tokens), divergence_idx + 3)
                    context = tokens[context_start:context_end]
                    print(f"  {model}: {context}")

def compare_token_encoding_consistency(tokenizers: Dict):
    """Check if the same token strings map to the same IDs across models."""
    print("\n" + "="*80)
    print("TOKEN ENCODING CONSISTENCY CHECK")
    print("="*80)
    
    # Get common tokens
    vocabs = {}
    for name, tokenizer in tokenizers.items():
        model_name = name.split('/')[-1]
        vocabs[model_name] = tokenizer.get_vocab()
    
    model_names = list(vocabs.keys())
    common_tokens = set(vocabs[model_names[0]].keys())
    for model in model_names[1:]:
        common_tokens &= set(vocabs[model].keys())
    
    print(f"Checking {min(100, len(common_tokens))} common tokens for ID consistency...")
    
    inconsistent = []
    for i, token in enumerate(list(common_tokens)[:100]):
        ids = {model: vocabs[model][token] for model in model_names}
        
        if len(set(ids.values())) > 1:
            inconsistent.append((token, ids))
    
    if inconsistent:
        print(f"\n✗ Found {len(inconsistent)} tokens with different IDs across models:")
        for token, ids in inconsistent[:10]:
            print(f"  Token '{token}':")
            for model, token_id in ids.items():
                print(f"    {model:40} : ID {token_id}")
    else:
        print("\n✓ All common tokens have consistent IDs across models!")

def main():
    print("="*80)
    print("LLAMA TOKENIZER COMPARISON TOOL")
    print("="*80)
    
    # Load tokenizers
    tokenizers = load_tokenizers()
    if not tokenizers:
        print("Failed to load tokenizers. Exiting.")
        return
    
    # Compare various aspects
    vocab_sizes = compare_vocab_sizes(tokenizers)
    special_tokens = compare_special_tokens(tokenizers)
    vocabs = compare_vocab_overlap(tokenizers)
    
    # Tokenize sample texts
    differences = tokenize_sample_texts(tokenizers, num_samples=10)
    
    # Analyze differences
    if differences:
        analyze_tokenization_differences(differences)
    
    # Check token encoding consistency
    compare_token_encoding_consistency(tokenizers)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    model_names = list(tokenizers.keys())
    if all(vocab_sizes[model_names[0]] == vocab_sizes[m] for m in model_names):
        print("✓ All models have the same vocabulary size")
    else:
        print("✗ Models have different vocabulary sizes")
    
    if differences:
        print(f"✗ Found {len(differences)} texts with different tokenizations")
    else:
        print("✓ All sample texts are tokenized identically")
    
    # Check if tokenizers are essentially the same
    if (all(vocab_sizes[model_names[0]] == vocab_sizes[m] for m in model_names) and 
        not differences and
        all(len(vocabs[m.split('/')[-1]] - set.union(*[vocabs[n.split('/')[-1]] for n in model_names if n != m])) == 0 
            for m in model_names)):
        print("\n🎯 CONCLUSION: The tokenizers are IDENTICAL")
    else:
        print("\n🎯 CONCLUSION: The tokenizers have DIFFERENCES")
        print("   See detailed analysis above for specific differences")

if __name__ == "__main__":
    main()