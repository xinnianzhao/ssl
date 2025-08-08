#!/usr/bin/env python3
"""Comprehensive test script for BPECTCTokenizer with BPE encoding."""

import numpy as np
from bpe_ctc_tokenizer import BPECTCTokenizer

def main():
    # Initialize tokenizer with merges file
    print("Loading BPECTCTokenizer with BPE merges...")
    tokenizer = BPECTCTokenizer("vocab.json", "merges.txt")
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Number of BPE merges: {len(tokenizer.bpe_ranks)}")
    print(f"Special tokens:")
    print(f"  - Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"  - Unk token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    print(f"  - BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"  - EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print()
    
    # Test 1: Dutch sentence encoding and decoding
    print("=" * 60)
    print("Test 1: Dutch sentence encoding/decoding with CTC simulation")
    print("=" * 60)
    
    test_text = "en dus nu een beetje uw werk meer uw hobby geworden ook"
    print(f"Original text: '{test_text}'")
    print()
    
    # Tokenize
    tokens = tokenizer.tokenize(test_text)
    print(f"BPE tokens ({len(tokens)}): {tokens}")
    print()
    
    # Encode to IDs
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs ({len(token_ids)}): {token_ids}")
    print()
    
    # Verify tokens can be decoded correctly (without CTC)
    simple_decoded = tokenizer.decode(token_ids, group_tokens=False)
    print(f"Simple decode (no CTC): '{simple_decoded}'")
    print()
    
    # Simulate CTC output with various patterns
    print("Simulating CTC outputs with different patterns:")
    print("-" * 40)
    
    # Pattern 1: Regular repetitions (3x each token)
    ctc_regular = []
    for tid in token_ids:
        ctc_regular.extend([tid, tid, tid])
    print(f"1. Regular (3x): {ctc_regular[:20]}...")
    decoded1 = tokenizer.decode(ctc_regular, group_tokens=True)
    print(f"   Decoded: '{decoded1}'")
    print()
    
    # Pattern 2: With blanks between tokens
    ctc_blanks = []
    for tid in token_ids:
        ctc_blanks.extend([0, tid, tid, tid])
    ctc_blanks.append(0)
    print(f"2. With blanks: {ctc_blanks[:20]}...")
    decoded2 = tokenizer.decode(ctc_blanks, group_tokens=True)
    print(f"   Decoded: '{decoded2}'")
    print()
    
    # Pattern 3: Variable repetitions
    import random
    random.seed(42)
    ctc_variable = []
    for tid in token_ids:
        # Random number of repetitions (1-5)
        reps = random.randint(1, 5)
        ctc_variable.extend([tid] * reps)
        # Sometimes add blanks
        if random.random() > 0.5:
            ctc_variable.append(0)
    print(f"3. Variable reps: {ctc_variable[:20]}...")
    decoded3 = tokenizer.decode(ctc_variable, group_tokens=True)
    print(f"   Decoded: '{decoded3}'")
    print()
    
    # Test 2: Multiple sentences
    print("=" * 60)
    print("Test 2: Batch processing with multiple sentences")
    print("=" * 60)
    
    sentences = [
        "het is een mooie dag",
        "ik werk aan een project",
        "de zon schijnt vandaag"
    ]
    
    print("Original sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. '{sent}'")
    print()
    
    # Encode all sentences
    encoded_sentences = []
    for sent in sentences:
        ids = tokenizer.encode(sent)
        encoded_sentences.append(ids)
        tokens = tokenizer.tokenize(sent)
        print(f"Sentence {sentences.index(sent)+1} tokens: {tokens}")
        print(f"  IDs: {ids}")
    print()
    
    # Simulate CTC outputs for batch
    ctc_batch = []
    for ids in encoded_sentences:
        ctc_seq = []
        for tid in ids:
            ctc_seq.extend([0, tid, tid])  # Simple pattern with blanks
        ctc_batch.append(ctc_seq)
    
    # Batch decode
    batch_decoded = tokenizer.batch_decode(ctc_batch, group_tokens=True)
    print("Batch decoded results:")
    for i, decoded in enumerate(batch_decoded, 1):
        print(f"  {i}. '{decoded}'")
    print()
    
    # Test 3: Edge cases
    print("=" * 60)
    print("Test 3: Edge cases")
    print("=" * 60)
    
    # Empty input
    empty_ids = []
    decoded_empty = tokenizer.decode(empty_ids)
    print(f"Empty input decoded: '{decoded_empty}'")
    
    # Only blanks
    only_blanks = [0, 0, 0, 0, 0]
    decoded_blanks = tokenizer.decode(only_blanks)
    print(f"Only blanks decoded: '{decoded_blanks}'")
    
    # Single token repeated
    single_token = [6, 6, 6, 6]  # "een" repeated
    decoded_single = tokenizer.decode(single_token)
    print(f"Single token repeated decoded: '{decoded_single}'")
    
    # Mixed with special tokens
    with_special = [2, 6, 6, 67, 67, 3]  # BOS + tokens + EOS
    decoded_special = tokenizer.decode(with_special, skip_special_tokens=False)
    print(f"With special tokens (not skipped): '{decoded_special}'")
    decoded_special_skip = tokenizer.decode(with_special, skip_special_tokens=True)
    print(f"With special tokens (skipped): '{decoded_special_skip}'")
    print()
    
    # Test 4: Save and reload vocabulary
    print("=" * 60)
    print("Test 4: Save and reload vocabulary")
    print("=" * 60)
    
    save_dir = "./test_save_v2"
    saved_files = tokenizer.save_vocabulary(save_dir, filename_prefix="test")
    print(f"Vocabulary saved to: {saved_files[0]}")
    
    # Load with the saved vocabulary
    new_tokenizer = BPECTCTokenizer(saved_files[0], "merges.txt")
    print(f"New tokenizer vocab size: {new_tokenizer.vocab_size}")
    
    # Test that it works the same
    test_reload = "test herlaad functionaliteit"
    ids1 = tokenizer.encode(test_reload)
    ids2 = new_tokenizer.encode(test_reload)
    print(f"Original tokenizer IDs: {ids1}")
    print(f"Reloaded tokenizer IDs: {ids2}")
    print(f"Match: {ids1 == ids2}")
    
    # Clean up
    import os
    import shutil
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()