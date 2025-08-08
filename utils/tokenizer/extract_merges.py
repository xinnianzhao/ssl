#!/usr/bin/env python3
"""Extract BPE merges from Llama tokenizer and create merges.txt."""

import json

def extract_merges_from_llama():
    """Extract BPE merges from Llama tokenizer."""
    
    # Path to the Llama tokenizer.json
    tokenizer_json_path = "/home/xinyu/data1/HuggingFace/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/tokenizer.json"
    
    print(f"Loading tokenizer.json from: {tokenizer_json_path}")
    
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_config = json.load(f)
    
    # Extract merges from the config
    if 'model' in tokenizer_config and 'merges' in tokenizer_config['model']:
        merges = tokenizer_config['model']['merges']
        print(f"Found {len(merges)} total merges in tokenizer.json")
        
        # Load our vocabulary to filter relevant merges
        with open('vocab.json', 'r', encoding='utf-8') as f:
            our_vocab = json.load(f)
        
        print(f"Our vocabulary has {len(our_vocab)} tokens")
        
        # Create a set of all tokens in our vocab
        our_tokens = set(our_vocab.keys())
        
        # Also create a set of all possible substrings that could be formed from merges
        # This helps us keep merges that might be used to form our tokens
        all_substrings = set()
        for token in our_tokens:
            # Add the token itself
            all_substrings.add(token)
            # Add all substrings
            for i in range(len(token)):
                for j in range(i+1, len(token)+1):
                    all_substrings.add(token[i:j])
        
        print(f"Total substrings to consider: {len(all_substrings)}")
        
        # Filter merges that are relevant to our vocabulary
        relevant_merges = []
        for merge in merges:
            parts = merge.split()
            if len(parts) == 2:
                left, right = parts
                merged = left + right
                
                # Keep this merge if:
                # 1. The merged result is in our vocab or its substrings
                # 2. OR either part is in our vocab or its substrings
                if (merged in all_substrings or 
                    left in all_substrings or 
                    right in all_substrings):
                    relevant_merges.append(merge)
        
        print(f"Filtered to {len(relevant_merges)} relevant merges")
        
        # Save all merges (we might want to keep all for proper BPE encoding)
        # But mark how many are directly relevant
        with open('merges.txt', 'w', encoding='utf-8') as f:
            # Write header
            f.write("#version: 0.2\n")
            # Write all merges (keeping original order is important for BPE)
            for merge in merges:
                f.write(f"{merge}\n")
        
        print(f"Saved all {len(merges)} merges to merges.txt (for proper BPE encoding)")
        
        # Also save a filtered version for reference
        with open('merges_filtered.txt', 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
            for merge in relevant_merges:
                f.write(f"{merge}\n")
        
        print(f"Saved {len(relevant_merges)} filtered merges to merges_filtered.txt")
        
        return True
    else:
        print("Could not find merges in tokenizer.json")
        return False

if __name__ == "__main__":
    extract_merges_from_llama()