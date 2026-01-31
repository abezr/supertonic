"""
Diagnostic script to analyze the trained TTS model and identify quality issues.
"""
import torch
import os
import json

def analyze_checkpoint(checkpoint_path):
    """Analyze a checkpoint file for training diagnostics."""
    print(f"\n{'='*70}")
    print(f"ANALYZING CHECKPOINT: {checkpoint_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Basic checkpoint info
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"\n1. TRAINING PROGRESS:")
    print(f"   - Trained for {epoch} epochs")
    
    # Model state analysis
    model_state = checkpoint['model_state_dict']
    print(f"\n2. MODEL ARCHITECTURE COMPONENTS:")
    
    # Count parameters per component
    components = {
        'text_encoder': [],
        'duration_predictor': [],
        'vector_estimator': []
    }
    
    for name, param in model_state.items():
        if 'text_encoder' in name:
            components['text_encoder'].append((name, param.numel()))
        elif 'duration_predictor' in name:
            components['duration_predictor'].append((name, param.numel()))
        elif 'vector_estimator' in name:
            components['vector_estimator'].append((name, param.numel()))
    
    total_params = 0
    for comp_name, params in components.items():
        comp_total = sum(p[1] for p in params)
        total_params += comp_total
        print(f"   - {comp_name}: {comp_total:,} parameters ({len(params)} tensors)")
    
    print(f"   - TOTAL: {total_params:,} parameters")
    
    # Weight statistics - check for vanishing/exploding gradients
    print(f"\n3. WEIGHT STATISTICS (checking for training issues):")
    
    issues_found = []
    
    for name, param in model_state.items():
        if 'weight' in name and param.dim() > 1:
            mean = param.mean().item()
            std = param.std().item()
            
            # Check for potential issues
            if std < 0.001:
                issues_found.append(f"   [WARN] {name}: std={std:.6f} (potentially collapsed/vanished)")
            elif std > 10:
                issues_found.append(f"   [WARN] {name}: std={std:.4f} (potentially exploded)")
            
            # Check for dead neurons (all zeros)
            zero_ratio = (param == 0).float().mean().item()
            if zero_ratio > 0.5:
                issues_found.append(f"   [WARN] {name}: {zero_ratio*100:.1f}% zeros (dead neurons)")
    
    if issues_found:
        print("   Issues detected:")
        for issue in issues_found[:10]:  # Show first 10
            print(issue)
        if len(issues_found) > 10:
            print(f"   ... and {len(issues_found) - 10} more")
    else:
        print("   [OK] No obvious weight issues detected")
    
    # Check gradient statistics if available
    print(f"\n4. OPTIMIZER STATE:")
    if 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        print(f"   - Optimizer type: {opt_state.get('param_groups', [{}])[0].get('__class__', 'Unknown')}")
        if 'param_groups' in opt_state:
            pg = opt_state['param_groups'][0]
            print(f"   - Learning rate: {pg.get('lr', 'unknown')}")
            print(f"   - Betas: {pg.get('betas', 'unknown')}")
            print(f"   - Weight decay: {pg.get('weight_decay', 'unknown')}")
    
    return checkpoint

def analyze_dataset_quality():
    """Analyze the dataset for quality issues."""
    print(f"\n{'='*70}")
    print(f"DATASET QUALITY ANALYSIS")
    print(f"{'='*70}")
    
    filelist_path = "dataset_marina/filelist.txt"
    if not os.path.exists(filelist_path):
        print("ERROR: filelist.txt not found")
        return
    
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\n1. DATASET SIZE:")
    print(f"   - Total samples: {len(lines)}")
    
    # Check for duplicate text
    texts = [line.strip().split('|')[1] if '|' in line else '' for line in lines]
    unique_texts = set(texts)
    print(f"   - Unique texts: {len(unique_texts)}")
    print(f"   - Duplicates: {len(texts) - len(unique_texts)}")
    
    # Check text lengths
    text_lengths = [len(t) for t in texts]
    avg_len = sum(text_lengths) / len(text_lengths)
    min_len = min(text_lengths)
    max_len = max(text_lengths)
    
    print(f"\n2. TEXT LENGTH DISTRIBUTION:")
    print(f"   - Average: {avg_len:.1f} characters")
    print(f"   - Min: {min_len}, Max: {max_len}")
    
    # Check for very short or very long samples
    short_samples = sum(1 for l in text_lengths if l < 20)
    long_samples = sum(1 for l in text_lengths if l > 200)
    
    if short_samples > 0:
        print(f"   [WARN] {short_samples} samples with < 20 chars (may be insufficient for learning)")
    if long_samples > 0:
        print(f"   [WARN] {long_samples} samples with > 200 chars (may cause memory issues)")
    
    # Check for path consistency
    print(f"\n3. PATH CONSISTENCY:")
    windows_paths = sum(1 for line in lines if ':\\' in line or 'D:\\' in line)
    unix_paths = sum(1 for line in lines if '/mnt/' in line)
    print(f"   - Windows-style paths: {windows_paths}")
    print(f"   - Unix-style paths: {unix_paths}")
    
    if windows_paths > 0 and unix_paths > 0:
        print(f"   [WARNING] MIXED PATH FORMATS DETECTED! This can cause training issues.")
        print(f"      Run tools/migrate_dataset_to_wsl.py to fix this.")

def analyze_training_config():
    """Analyze training configuration for issues."""
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION ANALYSIS")
    print(f"{'='*70}")
    
    # Read the training script
    with open('training/train_high_vram.py', 'r', encoding='utf-8') as f:
        train_code = f.read()
    
    print(f"\n1. HIGH VRAM TRAINING SCRIPT:")
    
    # Extract key hyperparameters
    import re
    
    batch_size = re.search(r'BATCH_SIZE\s*=\s*(\d+)', train_code)
    lr = re.search(r'LEARNING_RATE\s*=\s*([\d.e-]+)', train_code)
    epochs = re.search(r'EPOCHS\s*=\s*(\d+)', train_code)
    
    if batch_size:
        print(f"   - Batch Size: {batch_size.group(1)}")
    if lr:
        print(f"   - Learning Rate: {lr.group(1)}")
    if epochs:
        print(f"   - Max Epochs: {epochs.group(1)}")
    
    # Check model initialization
    vocab_match = re.search(r'vocab_size=(\d+)', train_code)
    embed_match = re.search(r'embed_dim=(\d+)', train_code)
    hidden_match = re.search(r'hidden_dim=(\d+)', train_code)
    
    print(f"\n2. MODEL HYPERPARAMETERS:")
    if vocab_match:
        print(f"   - Vocab size: {vocab_match.group(1)}")
    if embed_match:
        print(f"   - Embedding dim: {embed_match.group(1)}")
    if hidden_match:
        print(f"   - Hidden dim: {hidden_match.group(1)}")
    
    # Check for pretrained weights usage
    if 'PRETRAINED_WEIGHTS' in train_code and 'pretrained_weights.pt' in train_code:
        print(f"\n3. PRETRAINED WEIGHTS:")
        if os.path.exists('pretrained_weights.pt'):
            print(f"   [OK] pretrained_weights.pt exists")
            # Check file size
            size_mb = os.path.getsize('pretrained_weights.pt') / (1024 * 1024)
            print(f"   - Size: {size_mb:.2f} MB")
        else:
            print(f"   [WARN] pretrained_weights.pt NOT FOUND!")
            print(f"      Training from scratch - this explains poor quality!")

def identify_root_causes():
    """Identify the most likely root causes of poor quality."""
    print(f"\n{'='*70}")
    print(f"ROOT CAUSE ANALYSIS")
    print(f"{'='*70}")
    
    causes = []
    
    # Check 1: Pretrained weights
    if not os.path.exists('pretrained_weights.pt'):
        causes.append(("CRITICAL", "No pretrained weights found", 
                      "Training from random initialization. TTS models need pretrained weights for good quality."))
    
    # Check 2: Dataset size
    filelist_path = "dataset_marina/filelist.txt"
    if os.path.exists(filelist_path):
        with open(filelist_path, 'r', encoding='utf-8') as f:
            num_samples = len(f.readlines())
        if num_samples < 1000:
            causes.append(("HIGH", f"Small dataset ({num_samples} samples)", 
                          "TTS typically needs 1000+ hours of audio for high quality. Fine-tuning needs at least 10+ minutes."))
    
    # Check 3: Mixed paths
    if os.path.exists(filelist_path):
        with open(filelist_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if 'D:\\' in content and '/mnt/' in content:
            causes.append(("HIGH", "Mixed Windows/Unix paths in dataset", 
                          "Some samples may not load correctly, causing training on partial data."))
    
    # Check 4: Model architecture
    with open('training/model.py', 'r', encoding='utf-8') as f:
        model_code = f.read()
    
    if 'n_blocks=6' in model_code or 'n_blocks=8' in model_code:
        causes.append(("MEDIUM", "Small ConvNeXt encoder (6-8 blocks)", 
                      "Modern TTS uses 12-24 blocks. More capacity needed for complex phoneme-to-speech mapping."))
    
    # Check 5: Training duration
    checkpoint_path = "checkpoint_high_vram.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        epoch = checkpoint.get('epoch', 0)
        if epoch < 50:
            causes.append(("MEDIUM", f"Training stopped early (epoch {epoch})", 
                          "TTS models typically need 100-500 epochs to converge."))
    
    # Check 6: Inference settings
    with open('run_inference.py', 'r', encoding='utf-8') as f:
        inference_code = f.read()
    
    if 'n_steps=10' in inference_code or 'n_steps=20' in inference_code:
        causes.append(("LOW", "Low flow matching steps (10-20)", 
                      "More steps (50-100) improve quality but are slower."))
    
    print(f"\nISSUES RANKED BY SEVERITY:\n")
    for severity, issue, explanation in causes:
        icon = "[CRITICAL]" if severity == "CRITICAL" else "[HIGH]" if severity == "HIGH" else "[MEDIUM]" if severity == "MEDIUM" else "[LOW]"
        print(f"{icon} {issue}")
        print(f"   -> {explanation}\n")
    
    return causes

def suggest_fixes(causes):
    """Suggest fixes based on identified issues."""
    print(f"\n{'='*70}")
    print(f"RECOMMENDED FIXES")
    print(f"{'='*70}")
    
    print(f"\nIMMEDIATE ACTIONS (High Impact):\n")
    
    if any(c[0] == "CRITICAL" for c in causes):
        print("1. SETUP PRETRAINED WEIGHTS (CRITICAL):")
        print("   -> Run: ./setup_pretrained_weights.sh")
        print("   -> Or manually download and convert ONNX weights")
        print("   -> Without pretrained weights, quality will always be poor\n")
    
    if any("Mixed Windows/Unix paths" in c[1] for c in causes):
        print("2. FIX DATASET PATHS:")
        print("   -> Run: python tools/migrate_dataset_to_wsl.py")
        print("   -> This ensures all samples load correctly\n")
    
    print(f"\nTRAINING IMPROVEMENTS:\n")
    
    print("3. INCREASE MODEL CAPACITY:")
    print("   -> Edit training/train_high_vram.py:")
    print("      model = Supertonic(")
    print("          vocab_size=2000,")
    print("          embed_dim=384,     # Increase from 256")
    print("          hidden_dim=384,    # Increase from 256")
    print("          style_dim=128")
    print("      )")
    print("   -> Or set environment variable: CONVNEXT_BLOCKS=12\n")
    
    print("4. IMPROVE TRAINING REGIME:")
    print("   -> Use phased training from phases_loss_gate.json")
    print("   -> Start with LR=2e-5, warmup steps=2000")
    print("   -> Train for at least 100 epochs\n")
    
    print("5. ADD MORE DATA:")
    print("   -> Current: ~2100 samples (likely ~30-60 minutes)")
    print("   -> Target: 5000+ samples for fine-tuning")
    print("   -> More diverse text = better generalization\n")
    
    print(f"\nINFERENCE IMPROVEMENTS:\n")
    
    print("6. IMPROVE INFERENCE QUALITY:")
    print("   -> Increase flow matching steps: n_steps=50 or n_steps=100")
    print("   -> Use learned duration predictor instead of fixed_duration")
    print("   -> Add vocoder fine-tuning (HiFiGAN instead of GriffinLim)\n")

if __name__ == "__main__":
    print("="*70)
    print("SUPERTONIC TTS TRAINING DIAGNOSTIC")
    print("="*70)
    
    # Analyze checkpoints
    for ckpt in ['checkpoint_high_vram.pt', 'checkpoint_convnext.pt', 'checkpoint_low_vram.pt']:
        if os.path.exists(ckpt):
            analyze_checkpoint(ckpt)
            break
    else:
        print("\n[WARN] No checkpoint files found!")
    
    # Analyze dataset
    analyze_dataset_quality()
    
    # Analyze config
    analyze_training_config()
    
    # Identify root causes
    causes = identify_root_causes()
    
    # Suggest fixes
    suggest_fixes(causes)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
