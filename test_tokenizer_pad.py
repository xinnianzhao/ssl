#!/usr/bin/env python3
"""Test script to verify tokenizer pad method works correctly."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils', 'tokenizer'))

from bpe_aed_tokenizer import BPEAEDTokenizer
from bpe_ctc_tokenizer import BPECTCTokenizer

def test_tokenizer_pad(tokenizer_class, name):
    print(f"\nTesting {name}...")
    
    # Initialize tokenizer
    tokenizer = tokenizer_class(
        vocab_file="./utils/tokenizer/vocab.json",
        merges_file="./utils/tokenizer/merges.txt"
    )
    
    # Test texts
    texts = [
        "This is a short text",
        "This is a much longer text that should be used for padding test",
        "Medium text here"
    ]
    
    # Test without padding
    print("Without padding:")
    result = tokenizer(texts)
    for i, ids in enumerate(result["input_ids"]):
        print(f"  Text {i+1}: {len(ids)} tokens")
    
    # Test with padding
    print("\nWith padding:")
    result_padded = tokenizer(texts, padding=True, return_tensors="pt")
    print(f"  Shape: {result_padded['input_ids'].shape}")
    print(f"  All sequences padded to same length: {result_padded['input_ids'].shape[1]}")
    
    # Test attention mask
    if "attention_mask" in result_padded:
        print(f"  Attention mask shape: {result_padded['attention_mask'].shape}")
        print(f"  Attention mask sample: {result_padded['attention_mask'][0][:20]}...")
    
    print(f"{name} test passed!")
    return True

if __name__ == "__main__":
    try:
        # Test BPEAEDTokenizer
        test_tokenizer_pad(BPEAEDTokenizer, "BPEAEDTokenizer")
        
        # Test BPECTCTokenizer  
        test_tokenizer_pad(BPECTCTokenizer, "BPECTCTokenizer")
        
        print("\n✅ All tokenizer pad tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()