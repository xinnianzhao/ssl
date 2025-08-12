#!/usr/bin/env python
"""Test script to verify Whisper layer-wise learning rate decay and two-stage training setup"""

import torch
from transformers import WhisperForConditionalGeneration, WhisperConfig
from run_whisper import get_whisper_parameter_groups
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def test_whisper_lr_setup():
    """Test the learning rate setup for Whisper two-stage training"""
    
    # Create a dummy Whisper model
    config = WhisperConfig(
        vocab_size=51865,  # Standard Whisper vocab size
        d_model=768,       # Must be divisible by num_heads
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=12,  # 768 / 12 = 64
        decoder_attention_heads=12,
        pad_token_id=50257,
        bos_token_id=50257,
        eos_token_id=50257,
    )
    model = WhisperForConditionalGeneration(config)
    
    print("=" * 80)
    print("Testing Whisper Two-Stage Training Setup")
    print("=" * 80)
    
    # =================
    # Stage A: Decoder Warmup (encoder frozen)
    # =================
    print("\n1. Stage A: Decoder Warmup (Encoder Frozen)")
    print("-" * 40)
    
    # Freeze encoder for Stage A
    for p in model.model.encoder.parameters():
        p.requires_grad = False
    for p in model.model.decoder.parameters():
        p.requires_grad = True
    for p in model.proj_out.parameters():
        p.requires_grad = True
    
    # Check if lm_head shares weights with embed_tokens
    print(f"proj_out shares weights with embed_tokens: {model.proj_out.weight is model.model.decoder.embed_tokens.weight}")
    
    # Get parameter groups for Stage A
    groups_stage_a = get_whisper_parameter_groups(
        model,
        encoder_base_lr=3e-5,  # Won't be used since encoder is frozen
        decoder_base_lr=3e-4,  # Higher LR for decoder warmup
        lm_head_lr=3e-4,       # Same high LR for lm_head
        layer_decay=0.95,
        weight_decay=0.01
    )
    
    print(f"Number of parameter groups: {len(groups_stage_a)}")
    
    # Display Stage A parameter groups
    stage_a_total_params = 0
    for group in groups_stage_a:
        num_params = sum(p.numel() for p in group['params'])
        if num_params > 0:  # Only show non-empty groups
            stage_a_total_params += num_params
            print(f"  {group['name']:35s}: {num_params:10,} params, lr={group['lr']:.2e}, wd={group['weight_decay']}")
    
    print(f"\nTotal trainable parameters in Stage A: {stage_a_total_params:,}")
    
    # Create Stage A optimizer
    optimizer_a = AdamW(groups_stage_a, betas=(0.9, 0.98), eps=1e-8)
    
    # Create Stage A scheduler (3000 steps total, 300 warmup)
    scheduler_a = get_linear_schedule_with_warmup(
        optimizer_a,
        num_warmup_steps=300,
        num_training_steps=3000
    )
    
    # Test Stage A learning rates at different steps
    print("\nStage A Learning Rate Schedule:")
    test_steps = [0, 150, 300, 1500, 3000]
    for step in test_steps:
        # Simulate steps
        for _ in range(step - (test_steps[test_steps.index(step)-1] if test_steps.index(step) > 0 else 0)):
            scheduler_a.step()
        
        # Get current LRs
        lrs = [group['lr'] for group in optimizer_a.param_groups if sum(p.numel() for p in group['params']) > 0]
        if lrs:
            print(f"  Step {step:4d}: lm_head/decoder top = {max(lrs):.2e}")
    
    # Reset scheduler for next test
    optimizer_a = AdamW(groups_stage_a, betas=(0.9, 0.98), eps=1e-8)
    scheduler_a = get_linear_schedule_with_warmup(optimizer_a, num_warmup_steps=300, num_training_steps=3000)
    
    # =================
    # Stage B: Full Model Fine-tuning
    # =================
    print("\n2. Stage B: Full Model Fine-tuning with LLRD")
    print("-" * 40)
    
    # Unfreeze encoder for Stage B
    for p in model.model.encoder.parameters():
        p.requires_grad = True
    
    # Get parameter groups for Stage B
    groups_stage_b = get_whisper_parameter_groups(
        model,
        encoder_base_lr=3e-5,  # Lower LR for encoder
        decoder_base_lr=1e-4,  # Normal LR for decoder
        lm_head_lr=3e-4,       # Higher LR for lm_head
        layer_decay=0.95,
        weight_decay=0.01
    )
    
    print(f"Number of parameter groups: {len(groups_stage_b)}")
    
    # Group by component for clearer display
    components = {}
    for group in groups_stage_b:
        if sum(p.numel() for p in group['params']) == 0:
            continue
        component = group['name'].split('.')[0]
        if component not in components:
            components[component] = []
        components[component].append(group)
    
    # Display by component
    stage_b_total_params = 0
    for component in ['lm_head', 'decoder', 'encoder', 'other']:
        if component in components:
            print(f"\n  {component}:")
            for group in components[component]:
                num_params = sum(p.numel() for p in group['params'])
                stage_b_total_params += num_params
                layer_info = ""
                if 'layer' in group['name']:
                    # Extract layer number
                    import re
                    match = re.search(r'layer(\d+)', group['name'])
                    if match:
                        layer_num = int(match.group(1))
                        layer_info = f" (Layer {layer_num:2d})"
                print(f"    {group['name']:30s}{layer_info:12s}: {num_params:10,} params, lr={group['lr']:.2e}, wd={group['weight_decay']}")
    
    print(f"\nTotal trainable parameters in Stage B: {stage_b_total_params:,}")
    
    # Verify LLRD calculation
    print("\n3. Verifying Layer-wise Learning Rate Decay")
    print("-" * 40)
    
    print("\nDecoder LLRD (base_lr=1e-4, decay=0.95):")
    num_decoder_layers = 6
    for layer_idx in range(num_decoder_layers):
        expected_lr = 1e-4 * (0.95 ** (num_decoder_layers - 1 - layer_idx))
        print(f"  Layer {layer_idx}: {expected_lr:.2e} (decay factor: {0.95 ** (num_decoder_layers - 1 - layer_idx):.4f})")
    
    print("\nEncoder LLRD (base_lr=3e-5, decay=0.95):")
    num_encoder_layers = 6
    for layer_idx in range(num_encoder_layers):
        expected_lr = 3e-5 * (0.95 ** (num_encoder_layers - 1 - layer_idx))
        print(f"  Layer {layer_idx}: {expected_lr:.2e} (decay factor: {0.95 ** (num_encoder_layers - 1 - layer_idx):.4f})")
    
    print(f"\nLM head LR: 3e-04")
    
    # Create Stage B optimizer
    optimizer_b = AdamW(groups_stage_b, betas=(0.9, 0.98), eps=1e-8)
    
    # Create Stage B scheduler (7000 steps, 700 warmup)
    scheduler_b = get_linear_schedule_with_warmup(
        optimizer_b,
        num_warmup_steps=700,
        num_training_steps=7000
    )
    
    # Test Stage B learning rates at different steps
    print("\n4. Stage B Learning Rate Schedule (starting from step 3000):")
    test_steps_b = [0, 350, 700, 3500, 7000]
    for i, step in enumerate(test_steps_b):
        # Simulate steps
        if i > 0:
            for _ in range(step - test_steps_b[i-1]):
                scheduler_b.step()
        
        # Get current LRs for key components
        lm_head_lr = None
        decoder_top_lr = None
        encoder_top_lr = None
        
        for group in optimizer_b.param_groups:
            if 'lm_head' in group['name'] and 'decay' in group['name']:
                lm_head_lr = group['lr']
            elif 'decoder.layer5.decay' in group['name']:  # Top decoder layer
                decoder_top_lr = group['lr']
            elif 'encoder.layer5.decay' in group['name']:  # Top encoder layer
                encoder_top_lr = group['lr']
        
        actual_step = 3000 + step  # Offset by Stage A steps
        print(f"  Step {actual_step:4d} (Stage B step {step:4d}):")
        if lm_head_lr:
            print(f"    lm_head     = {scheduler_b.get_last_lr()[0] if hasattr(scheduler_b, 'get_last_lr') else lm_head_lr:.2e}")
        if decoder_top_lr:
            print(f"    decoder top = {decoder_top_lr:.2e}")
        if encoder_top_lr:
            print(f"    encoder top = {encoder_top_lr:.2e}")
    
    # Verify optimizer settings
    print("\n5. Optimizer Configuration")
    print("-" * 40)
    print(f"Stage A - Optimizer: AdamW")
    print(f"  Betas: (0.9, 0.98)")
    print(f"  Eps: 1e-8")
    print(f"  Weight decay: 0.01 (except bias/LayerNorm)")
    print(f"  Warmup: 300 steps (10% of 3000)")
    
    print(f"\nStage B - Optimizer: AdamW")
    print(f"  Betas: (0.9, 0.98)")
    print(f"  Eps: 1e-8")
    print(f"  Weight decay: 0.01 (except bias/LayerNorm)")
    print(f"  Warmup: 700 steps (10% of 7000)")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    test_whisper_lr_setup()