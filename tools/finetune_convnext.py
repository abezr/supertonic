#!/usr/bin/env python3
"""
ConvNeXt fine-tuning script for Supertonic TTS model.

Usage (WSL):
    DEV_STRESS=1 ./venv/bin/python tools/finetune_convnext.py

For module execution:
    DEV_STRESS=1 ./venv/bin/python -m tools.finetune_convnext

Environment variables:
    DEV_STRESS=1          - Run GPU stress test before training
    DIAG_GPU=1            - Enable CUDA event timing for profiling
    COMPILE_MODEL=1       - Enable torch.compile for faster training (Ampere+)
    BATCH_SIZE=4          - Batch size (recommended: 4 for RTX 3050)
    GRAD_ACCUM_STEPS=6    - Gradient accumulation steps (recommended: 6 for RTX 3050)
    EPOCHS=100            - Number of training epochs
    CONVNEXT_BLOCKS=6     - Number of ConvNeXt blocks

Recommended settings for RTX 3050 (4GB VRAM):
    COMPILE_MODEL=1 BATCH_SIZE=4 GRAD_ACCUM_STEPS=6 ./venv/bin/python tools/finetune_convnext.py
"""
import sys
from pathlib import Path

# Add project root to Python path for imports when running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Repo imports
from training.model import Supertonic
from training.dataset import TTSDataset, collate_fn
from training.tokenizer_builder import build_extended_indexer
from training.utils import get_best_device, maximum_path
from training.onnx_to_pytorch import load_pretrained_into_model


def main():
    device = get_best_device()
    print(f"Training on {device} (ConvNeXt fine-tune)")

    if device.type == 'cpu':
        print("\n" + "="*60)
        print("WARNING: CUDA not detected! Training will be extremely slow.")
        print("Please install PyTorch with CUDA support:")
        print("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("="*60 + "\n")

    # Enable TF32 for speed on Ampere+ when available
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # Configuration (env overridable)
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
    GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 6))
    COMPILE_MODEL = os.environ.get("COMPILE_MODEL", "0") == "1"
    LR_MAIN = float(os.environ.get("LEARNING_RATE", 3e-5))
    LR_BASE = float(os.environ.get("LR_BASE", 5e-6))
    LR_NEW = float(os.environ.get("LR_NEW", 5e-5))
    EPOCHS = int(os.environ.get("EPOCHS", 100))
    CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoint_convnext.pt")
    MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 1.0))
    PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "pretrained_weights.pt")
    WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 1000))
    FREEZE_LOWER_BLOCKS = int(os.environ.get("FREEZE_LOWER_BLOCKS", 0))
    UNFREEZE_AFTER_STEPS = int(os.environ.get("UNFREEZE_AFTER_STEPS", 0))
    CONVNEXT_BLOCKS = int(os.environ.get("CONVNEXT_BLOCKS", 6))
    LOG_EVERY_N_BATCHES = int(os.environ.get("LOG_EVERY_N_BATCHES", 10))

    # GPU diagnostics flags
    DIAG_GPU = os.environ.get('DIAG_GPU', '0') == '1'
    DEV_STRESS = os.environ.get('DEV_STRESS', '0') == '1'

    if DIAG_GPU:
        print("[DIAG_GPU] Enabled: CUDA Event timing will be used for accurate GPU profiling")
    if DEV_STRESS:
        print("[DEV_STRESS] Enabled: Will run GPU stress test before training")
    if COMPILE_MODEL:
        print("="*60)
        print("[COMPILE_MODEL] Enabled: Will use torch.compile for optimized model execution")
        print("⚠️  WARNING: torch.compile may cause hangs on some systems!")
        print("   If training freezes or hangs, disable with COMPILE_MODEL=0")
        print("="*60)

    if not os.path.exists("unicode_indexer_expanded.json"):
        print("Building tokenizer...")
        build_extended_indexer()

    # Model (force ConvNeXt encoder)
    os.environ["ENCODER_TYPE"] = "convnext"
    os.environ["CONVNEXT_BLOCKS"] = str(CONVNEXT_BLOCKS)
    model = Supertonic(
        vocab_size=2000,
        embed_dim=192,   # ignored in convnext path
        hidden_dim=192,  # ignored in convnext path
        style_dim=128,
        encoder_type="convnext",
        n_langs=10
    ).to(device)

    print(f"Encoder type: {getattr(model, 'encoder_type', 'transformer')}")
    try:
        print(f"ConvNeXt blocks: {len(model.text_encoder.convnext)} (set via CONVNEXT_BLOCKS={CONVNEXT_BLOCKS})")
    except Exception:
        pass

    # Load pretrained weights if available (BEFORE compiling)
    if os.path.exists(PRETRAINED_WEIGHTS) and not os.path.exists(CHECKPOINT_PATH):
        print(f"\n{'='*60}")
        print(f"🔄 Loading pretrained ONNX weights from {PRETRAINED_WEIGHTS}")
        print(f"{'='*60}")
        load_pretrained_into_model(model, PRETRAINED_WEIGHTS, freeze_encoder=False)
        print("✅ Pretrained weights loaded - starting fine-tuning\n")
    elif not os.path.exists(PRETRAINED_WEIGHTS) and not os.path.exists(CHECKPOINT_PATH):
        print(f"\n{'!'*60}")
        print("⚠️  WARNING: No pretrained weights found! Training from random initialization.")
        print(f"{'!'*60}\n")

    # Optimizer with param groups
    param_groups = []
    enc_base, enc_new = [], []
    for n, p in model.text_encoder.named_parameters():
        if any(n.startswith(k) for k in ["embedding", "lang_embedding", "style_proj"]):
            enc_new.append(p)
        else:
            enc_base.append(p)
    if enc_base:
        param_groups.append({"params": enc_base, "lr": LR_BASE})
    if enc_new:
        param_groups.append({"params": enc_new, "lr": LR_NEW})
    param_groups.append({"params": list(model.duration_predictor.parameters()), "lr": LR_MAIN})
    param_groups.append({"params": list(model.vector_estimator.parameters()), "lr": LR_MAIN})
    optimizer = optim.AdamW(param_groups, lr=LR_MAIN)

    print(f"LR_BASE={LR_BASE}, LR_NEW={LR_NEW}, LR_MAIN={LR_MAIN}")

    # Optional freeze
    if FREEZE_LOWER_BLOCKS > 0:
        for i, blk in enumerate(model.text_encoder.convnext):
            if i < FREEZE_LOWER_BLOCKS:
                for p in blk.parameters():
                    p.requires_grad = False
        print(f"Froze lower ConvNeXt blocks: [0..{FREEZE_LOWER_BLOCKS-1}]")
    unfroze = False

    # Warmup scheduler
    scheduler = None
    if WARMUP_STEPS > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: min(1.0, s / max(1, WARMUP_STEPS))
        )
        print(f"Warmup enabled: WARMUP_STEPS={WARMUP_STEPS}")

    # AMP scaler
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Resume (load checkpoint BEFORE compiling)
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        state_dict = checkpoint['model_state_dict']

        # Handle torch.compile prefix mismatch in both directions:
        # 1. Checkpoint has _orig_mod. prefix but model doesn't expect it -> strip prefix
        # 2. Checkpoint doesn't have prefix but model expects it -> add prefix
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())

        has_orig_mod_in_checkpoint = any(k.startswith('_orig_mod.') for k in checkpoint_keys)
        needs_orig_mod_in_model = any(k.startswith('_orig_mod.') for k in model_keys)

        if has_orig_mod_in_checkpoint and not needs_orig_mod_in_model:
            # Case 1: Strip _orig_mod. prefix from checkpoint
            print("  Converting: stripping '_orig_mod.' prefix from checkpoint keys")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        elif not has_orig_mod_in_checkpoint and needs_orig_mod_in_model:
            # Case 2: Add _orig_mod. prefix to checkpoint
            print("  Converting: adding '_orig_mod.' prefix to checkpoint keys")
            state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Apply torch.compile if enabled (Ampere+ GPUs like RTX 3050)
    # MUST be done AFTER loading checkpoint to avoid _orig_mod. prefix mismatch
    if COMPILE_MODEL and device.type == 'cuda':
        print("\n" + "="*60)
        print("🔧 Compiling model with torch.compile...")
        print("="*60)
        try:
            # Compile the model for optimized execution
            # Using default mode which works well for most models
            model = torch.compile(model, mode="default")
            print("✅ Model compiled successfully\n")
        except Exception as e:
            print(f"⚠️  torch.compile failed: {e}")
            print("   Continuing with uncompiled model\n")

    # Dataset
    if not os.path.exists("dataset_marina/filelist.txt"):
        print("Prepare 'dataset_marina/filelist.txt' by running generate_dataset.py")
        return

    dataset = TTSDataset("dataset_marina/filelist.txt", "unicode_indexer_expanded.json")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model.train()
    print("Beginning training (ConvNeXt fine-tune).\n")
    global_step = 0

    # Rolling average timing instrumentation
    timing_window_size = LOG_EVERY_N_BATCHES
    data_times = []
    fwd_bwd_times = []
    total_times = []

    # DEV_STRESS mode: Run GPU stress test to verify GPU can hit high utilization
    if DEV_STRESS and device.type == 'cuda':
        print("\n" + "="*60)
        print("[DEV_STRESS] Running GPU stress test...")
        print("="*60)
        _run_gpu_stress_test(device, duration_sec=15)
        print("="*60)
        print("[DEV_STRESS] Stress test complete. Starting training...\n")

    for epoch in range(start_epoch, EPOCHS):
        data_start = time.time()
        for batch_idx, (text_padded, text_lens, mel_padded, mel_lens, paths, lang_ids) in enumerate(loader):
            batch_start = time.time()
            data_time = batch_start - data_start

            text_padded = text_padded.to(device, non_blocking=True)
            mel_padded = mel_padded.to(device, non_blocking=True)
            lang_ids = lang_ids.to(device, non_blocking=True)

            # CUDA placement assertions
            assert text_padded.is_cuda, f"text_padded not on CUDA: {text_padded.device}"
            assert mel_padded.is_cuda, f"mel_padded not on CUDA: {mel_padded.device}"

            style = torch.randn(text_padded.size(0), 128, device=device)
            optimizer.zero_grad()

            device_type = 'cuda' if device.type == 'cuda' else 'cpu'

            # CUDA Event timing for accurate GPU profiling
            if DIAG_GPU and device.type == 'cuda':
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            fwd_bwd_start = time.time()
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                # 1) Text encode
                text_emb = model.text_encoder(text_padded, style, lang_ids=lang_ids)
                assert text_emb.is_cuda, f"text_emb not on CUDA: {text_emb.device}"

                # 2) MAS alignment (using vector_estimator final proj as proxy)
                # Build masks on device to avoid H2D transfers
                T_text = text_padded.size(1)
                T_mel = mel_padded.size(2)
                idx_text = torch.arange(T_text, device=device)[None, :]
                idx_mel = torch.arange(T_mel, device=device)[None, :]
                text_mask = idx_text < text_lens[:, None].to(device, non_blocking=True)
                mel_mask = idx_mel < mel_lens[:, None].to(device, non_blocking=True)

                with torch.no_grad():
                    text_proj = model.vector_estimator.final_proj(text_emb)  # already on device
                    log_p = -torch.cdist(text_proj, mel_padded.transpose(1, 2))  # stays on device
                    attn_mask = text_mask.unsqueeze(2) * mel_mask.unsqueeze(1)
                    attn = maximum_path(log_p, attn_mask)
                    target_dur = attn.sum(2)
                    log_target_dur = torch.log(target_dur + 1e-6)

                # 3) Duration loss
                log_dur_pred = model.duration_predictor(text_emb, style)
                loss_dur = F.mse_loss(log_dur_pred * text_mask, log_target_dur * text_mask, reduction='sum') / text_mask.sum()

                # 4) Upsample by MAS
                text_emb_upsampled = torch.bmm(attn.transpose(1, 2), text_emb)

                # 5) Flow Matching loss
                b = mel_padded.size(0)
                t = torch.rand(b, device=device)
                x1 = mel_padded
                x0 = torch.randn_like(x1)
                t_expanded = t[:, None, None].expand_as(x1)
                xt = t_expanded * x1 + (1 - t_expanded) * x0
                vt_target = x1 - x0
                vt_pred = model.vector_estimator(xt.transpose(1, 2), text_emb_upsampled, style, t)
                loss_fm = torch.mean((vt_target.transpose(1, 2) - vt_pred) ** 2)

                loss = (loss_fm + loss_dur) / GRAD_ACCUM_STEPS

            # Backward + step
            scaler.scale(loss).backward()

            took_step = False
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                took_step = True

            # Record CUDA event timing
            if DIAG_GPU and device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                # Print every 20 batches
                if batch_idx % 20 == 0:
                    print(f"[PerfDiag] fwd_bwd_ms={elapsed_ms:.2f}, batches={batch_idx}")

            step_end = time.time()
            fwd_bwd_time = step_end - fwd_bwd_start
            total_time = step_end - batch_start

            # Update rolling averages
            data_times.append(data_time)
            fwd_bwd_times.append(fwd_bwd_time)
            total_times.append(total_time)
            if len(data_times) > timing_window_size:
                data_times.pop(0)
                fwd_bwd_times.pop(0)
                total_times.pop(0)

            # Log every N batches
            if batch_idx % LOG_EVERY_N_BATCHES == 0:
                avg_data = sum(data_times) / len(data_times) if data_times else 0.0
                avg_fwd_bwd = sum(fwd_bwd_times) / len(fwd_bwd_times) if fwd_bwd_times else 0.0
                avg_total = sum(total_times) / len(total_times) if total_times else 0.0

                # Guard samples_per_sec to avoid negatives when timings are tiny
                sps = BATCH_SIZE / avg_total if avg_total > 0 else 0.0

                # GPU utilization hint: ratio of compute time to total time
                # Use CUDA event timing when DIAG_GPU is enabled
                if DIAG_GPU and device.type == 'cuda' and 'elapsed_ms' in locals():
                    gpu_util_hint = (elapsed_ms / 1000.0) / avg_total if avg_total > 0 else 0.0
                else:
                    gpu_util_hint = avg_fwd_bwd / avg_total if avg_total > 0 else 0.0

                print(f"Epoch {epoch} | Batch {batch_idx:3d} | Loss {loss.item()*GRAD_ACCUM_STEPS:.4f} | "
                      f"Speed: {sps:.2f} items/s | Time: {total_time:.2f}s (GPU {fwd_bwd_time:.2f}s)")

                # Print perf summary every N batches
                if batch_idx > 0:
                    print(f"[Perf] data={avg_data:.3f}s, fwd_bwd={avg_fwd_bwd:.3f}s, total={avg_total:.3f}s, "
                          f"gpu_util_hint={gpu_util_hint:.2%}")

                sys.stdout.flush()

            # Unfreeze schedule
            if took_step:
                global_step += 1
                if UNFREEZE_AFTER_STEPS > 0 and (not unfroze) and global_step >= UNFREEZE_AFTER_STEPS and FREEZE_LOWER_BLOCKS > 0:
                    for i in range(FREEZE_LOWER_BLOCKS):
                        for p in model.text_encoder.convnext[i].parameters():
                            p.requires_grad = True
                    unfroze = True
                    print(f"Unfroze lower ConvNeXt blocks at global_step={global_step}")
                    sys.stdout.flush()

            # Reset data_start for next iteration
            data_start = time.time()

        # Save checkpoint each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"Saved checkpoint to {CHECKPOINT_PATH}")


def _run_gpu_stress_test(device, duration_sec=15):
    """Run GPU stress test to verify GPU can hit high utilization."""
    import time

    print(f"[DEV_STRESS] Running {duration_sec}s GPU stress test on {device}...")

    # Create large tensors for matrix multiplication
    size = 4096
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(5):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()

    start_time = time.time()
    iterations = 0

    while time.time() - start_time < duration_sec:
        # Heavy matrix multiplication
        _ = torch.mm(a, b)
        # Add some 1D conv operations
        d = torch.randn(64, 128, 1024, device=device)
        e = torch.randn(128, 128, 3, device=device)
        _ = torch.nn.functional.conv1d(d, e, padding=1)
        # Force synchronization occasionally to prevent driver timeout
        if iterations % 10 == 0:
            torch.cuda.synchronize()
        iterations += 1

    elapsed = time.time() - start_time
    print(f"[DEV_STRESS] Completed {iterations} iterations in {elapsed:.1f}s "
          f"({iterations/elapsed:.1f} iter/s)")

    # Check GPU memory and utilization info
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"[DEV_STRESS] GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")


if __name__ == "__main__":
    main()
