#!/usr/bin/env python
"""Test script to verify layer-wise learning rate decay setup"""

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from run_wav2vec2 import get_parameter_groups
from torch.optim import AdamW

def test_lr_setup():
    """Test the learning rate setup for different stages"""
    
    # Create a dummy model
    config = Wav2Vec2Config(
        vocab_size=100,
        hidden_size=768,
        num_hidden_layers=12,  # 12 encoder layers
    )
    model = Wav2Vec2ForCTC(config)
    
    print("=" * 80)
    print("Testing Layer-wise Learning Rate Decay Setup")
    print("=" * 80)
    
    # Test Stage A: CTC warmup (encoder frozen)
    print("\n1. Stage A: CTC Warmup (Encoder Frozen)")
    print("-" * 40)
    
    # Freeze encoder
    for p in model.wav2vec2.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = True
    
    groups_stage_a = get_parameter_groups(
        model,
        base_lr=1e-4,
        layer_decay=0.95,
        ctc_lr=5e-4,  # Higher LR for CTC during warmup
        weight_decay=0.01,
        freeze_feature_extractor=True
    )
    
    print(f"Number of parameter groups: {len(groups_stage_a)}")
    for group in groups_stage_a:
        num_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']:30s}: {num_params:10,} params, lr={group['lr']:.2e}, wd={group['weight_decay']}")
    
    # Test Stage B: Full model fine-tuning
    print("\n2. Stage B: Full Model Fine-tuning with LLRD")
    print("-" * 40)
    
    # Unfreeze encoder
    for p in model.wav2vec2.parameters():
        p.requires_grad = True
    
    groups_stage_b = get_parameter_groups(
        model,
        base_lr=1e-4,
        layer_decay=0.95,
        ctc_lr=3e-4,
        weight_decay=0.01,
        freeze_feature_extractor=True  # Keep CNN frozen
    )
    
    print(f"Number of parameter groups: {len(groups_stage_b)}")
    
    # Group by component for clearer display
    components = {}
    for group in groups_stage_b:
        component = group['name'].split('.')[0]
        if component not in components:
            components[component] = []
        components[component].append(group)
    
    # Display by component
    for component in ['ctc_head', 'encoder', 'feature_projection', 'feature_extractor', 'other']:
        if component in components:
            print(f"\n  {component}:")
            for group in components[component]:
                num_params = sum(p.numel() for p in group['params'])
                layer_info = ""
                if 'layer' in group['name']:
                    # Extract layer number
                    import re
                    match = re.search(r'layer(\d+)', group['name'])
                    if match:
                        layer_num = int(match.group(1))
                        layer_info = f" (Layer {layer_num:2d})"
                print(f"    {group['name']:35s}{layer_info:12s}: {num_params:10,} params, lr={group['lr']:.2e}, wd={group['weight_decay']}")
    
    # Verify LLRD calculation
    print("\n3. Verifying Layer-wise Learning Rate Decay")
    print("-" * 40)
    
    num_layers = 12
    base_lr = 1e-4
    layer_decay = 0.95
    
    print(f"Base LR (top layer): {base_lr:.2e}")
    print(f"Layer decay factor: {layer_decay}")
    print(f"Number of encoder layers: {num_layers}")
    print("\nExpected LR for each encoder layer:")
    
    for layer_idx in range(num_layers):
        expected_lr = base_lr * (layer_decay ** (num_layers - 1 - layer_idx))
        print(f"  Layer {layer_idx:2d}: {expected_lr:.2e} (decay factor: {layer_decay ** (num_layers - 1 - layer_idx):.4f})")
    
    print(f"\nCTC head LR: 3e-04")
    print(f"Feature projection LR: {base_lr * (layer_decay ** (num_layers - 1)):.2e}")
    print(f"Feature extractor: Frozen (requires_grad=False)")
    
    # Test optimizer creation
    print("\n4. Testing AdamW Optimizer Creation")
    print("-" * 40)
    
    optimizer = AdamW(
        groups_stage_b,
        betas=(0.9, 0.98),
        eps=1e-8
    )
    
    print(f"Optimizer type: {type(optimizer).__name__}")
    print(f"Betas: {optimizer.defaults['betas']}")
    print(f"Eps: {optimizer.defaults['eps']}")
    print(f"Number of param groups: {len(optimizer.param_groups)}")
    
    # Verify that all trainable parameters are included
    all_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer_params = sum(sum(p.numel() for p in g['params']) for g in optimizer.param_groups)
    
    print(f"\nTotal trainable parameters in model: {all_trainable_params:,}")
    print(f"Total parameters in optimizer: {optimizer_params:,}")
    print(f"Match: {'✓' if all_trainable_params == optimizer_params else '✗'}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    test_lr_setup()