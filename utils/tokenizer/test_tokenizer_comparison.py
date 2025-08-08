#!/usr/bin/env python3
"""Test script to compare BPEAEDTokenizer with Llama Tokenizer and verify vocab mapping."""

import json
from transformers import AutoTokenizer
from bpe_aed_tokenizer import BPEAEDTokenizer

def main():
    # Test sentence
    test_text = "en dus nu een beetje uw werk meer uw hobby geworden ook"
    print("=" * 70)
    print("TOKENIZER COMPARISON TEST")
    print("=" * 70)
    print(f"Test sentence: '{test_text}'")
    print()
    
    # 1. Initialize BPEAEDTokenizer
    print("1. Loading BPEAEDTokenizer...")
    bpe_tokenizer = BPEAEDTokenizer("vocab.json", "merges.txt")
    print(f"   Vocabulary size: {bpe_tokenizer.vocab_size}")
    
    # 2. Initialize Llama Tokenizer
    print("\n2. Loading Llama-3.1-8B-Instruct tokenizer...")
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    print(f"   Vocabulary size: {llama_tokenizer.vocab_size}")
    
    # 3. Load vocab_mapper.json
    print("\n3. Loading vocab_mapper.json...")
    with open("vocab_mapper.json", "r", encoding="utf-8") as f:
        vocab_mapper = json.load(f)
    print(f"   Mapper contains {len(vocab_mapper)} mappings")
    
    print("\n" + "=" * 70)
    print("ENCODING TEST")
    print("=" * 70)
    
    # 4. Encode with BPEAEDTokenizer
    print("\n4. Encoding with BPEAEDTokenizer:")
    bpe_tokens = bpe_tokenizer.tokenize(test_text)
    print(f"   BPE tokens: {bpe_tokens}")
    ids_1 = bpe_tokenizer.encode(test_text)
    print(f"   ids_1 (BPEAEDTokenizer): {ids_1}")
    print(f"   Length: {len(ids_1)}")
    
    # 5. Encode with Llama Tokenizer
    print("\n5. Encoding with Llama Tokenizer:")
    llama_tokens = llama_tokenizer.tokenize(test_text)
    print(f"   Llama tokens: {llama_tokens}")
    ids_2 = llama_tokenizer.encode(test_text, add_special_tokens=False)
    print(f"   ids_2 (Llama): {ids_2}")
    print(f"   Length: {len(ids_2)}")
    
    # 6. Map ids_1 to ids_3 using vocab_mapper
    print("\n6. Mapping ids_1 to ids_3 using vocab_mapper:")
    ids_3 = []
    unmapped_tokens = []
    
    # We need to account for the special token offset in our vocab
    special_token_offset = len(bpe_tokenizer.special_tokens)
    
    for id_val in ids_1:
        # Get the token for this ID
        token = bpe_tokenizer.decoder.get(id_val)
        
        # Skip special tokens in mapping
        if token in bpe_tokenizer.special_tokens:
            continue
            
        # Adjust ID to match the original vocab index (before special tokens were added)
        original_id = id_val - special_token_offset
        
        # Look up in vocab_mapper
        mapper_key = str(original_id)
        if mapper_key in vocab_mapper:
            ids_3.append(vocab_mapper[mapper_key])
        else:
            unmapped_tokens.append((id_val, token))
            # Try to get the ID directly from Llama tokenizer as fallback
            llama_id = llama_tokenizer.convert_tokens_to_ids(token)
            if llama_id != llama_tokenizer.unk_token_id:
                ids_3.append(llama_id)
            else:
                ids_3.append(llama_tokenizer.unk_token_id)
    
    print(f"   ids_3 (mapped from ids_1): {ids_3}")
    print(f"   Length: {len(ids_3)}")
    
    if unmapped_tokens:
        print(f"   Warning: {len(unmapped_tokens)} tokens couldn't be mapped:")
        for id_val, token in unmapped_tokens[:5]:  # Show first 5
            print(f"     - ID {id_val}: '{token}'")
    
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    # 7. Compare ids_2 and ids_3
    print("\n7. Comparing ids_2 (Llama) and ids_3 (mapped):")
    if ids_2 == ids_3:
        print("   ✓ SUCCESS: ids_2 and ids_3 are IDENTICAL!")
    else:
        print("   ✗ MISMATCH: ids_2 and ids_3 are different")
        print(f"   ids_2: {ids_2}")
        print(f"   ids_3: {ids_3}")
        
        # Find differences
        min_len = min(len(ids_2), len(ids_3))
        for i in range(min_len):
            if ids_2[i] != ids_3[i]:
                token_2 = llama_tokenizer.decode([ids_2[i]])
                token_3 = llama_tokenizer.decode([ids_3[i]]) if ids_3[i] != llama_tokenizer.unk_token_id else "<UNK>"
                print(f"   Difference at position {i}: {ids_2[i]} ('{token_2}') vs {ids_3[i]} ('{token_3}')")
        
        if len(ids_2) != len(ids_3):
            print(f"   Length difference: {len(ids_2)} vs {len(ids_3)}")
    
    print("\n" + "=" * 70)
    print("DECODING TEST")
    print("=" * 70)
    
    # 8. Decode ids_1 with BPEAEDTokenizer
    print("\n8. Decoding with BPEAEDTokenizer:")
    str_1 = bpe_tokenizer.decode(ids_1)
    print(f"   str_1 (from ids_1): '{str_1}'")
    
    # 9. Decode ids_2 with Llama Tokenizer
    print("\n9. Decoding with Llama Tokenizer:")
    str_2 = llama_tokenizer.decode(ids_2, skip_special_tokens=True)
    print(f"   str_2 (from ids_2): '{str_2}'")
    
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)
    
    # 10. Compare decoded strings with original
    print("\n10. Comparing decoded strings with original:")
    print(f"   Original:  '{test_text}'")
    print(f"   str_1:     '{str_1}'")
    print(f"   str_2:     '{str_2}'")
    print()
    
    if str_1 == test_text:
        print("   ✓ str_1 matches original text perfectly!")
    else:
        print(f"   ✗ str_1 differs from original")
        
    if str_2 == test_text:
        print("   ✓ str_2 matches original text perfectly!")
    else:
        print(f"   ✗ str_2 differs from original")
    
    if str_1 == str_2:
        print("   ✓ str_1 and str_2 are identical!")
    else:
        print("   ✗ str_1 and str_2 are different")
    
    # Additional debug info
    print("\n" + "=" * 70)
    print("DEBUG INFORMATION")
    print("=" * 70)
    
    print("\nToken-by-token comparison:")
    print(f"{'Position':<10} {'BPE Token':<20} {'BPE ID':<10} {'Llama Token':<20} {'Llama ID':<10}")
    print("-" * 80)
    
    max_len = max(len(bpe_tokens), len(llama_tokens))
    for i in range(max_len):
        bpe_tok = bpe_tokens[i] if i < len(bpe_tokens) else "-"
        bpe_id = ids_1[i] if i < len(ids_1) else "-"
        llama_tok = llama_tokens[i] if i < len(llama_tokens) else "-"
        llama_id = ids_2[i] if i < len(ids_2) else "-"
        print(f"{i:<10} {bpe_tok:<20} {bpe_id:<10} {llama_tok:<20} {llama_id:<10}")

if __name__ == "__main__":
    main()