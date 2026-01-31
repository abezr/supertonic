#!/usr/bin/env python3
import os
import time
import sys
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
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2))
    GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 12))
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

    # Load pretrained weights if available
    if os.path.exists(PRETRAINED_WEIGHTS) and not os.path.exists(CHECKPOINT_PATH):
        print(f"\n{'='*60}")
        print(f"🔄 Loading pretrained ONNX weights from {PRETRAINED_WEIGHTS}")
        print(f"{'='*60}")
        load_pretrained_into_model(model, PRETRAINED_WEIGHTS, freeze_encoder=False)
        print(f"✅ Pretrained weights loaded - starting fine-tuning\n")
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

    # Resume
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

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
        num_workers=0
    )

    model.train()
    print("Beginning training (ConvNeXt fine-tune).\n")
    global_step = 0

    for epoch in range(start_epoch, EPOCHS):
        for batch_idx, (text_padded, text_lens, mel_padded, mel_lens, paths, lang_ids) in enumerate(loader):
            batch_start = time.time()
            text_padded = text_padded.to(device)
            mel_padded = mel_padded.to(device)
            lang_ids = lang_ids.to(device)

            style = torch.randn(text_padded.size(0), 128).to(device)
            optimizer.zero_grad()

            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                # 1) Text encode
                text_emb = model.text_encoder(text_padded, style, lang_ids=lang_ids)

                # 2) MAS alignment (using vector_estimator final proj as proxy)
                with torch.no_grad():
                    text_proj = model.vector_estimator.final_proj(text_emb)  # [B, T_text, 80]
                    dist = torch.cdist(text_proj, mel_padded.transpose(1, 2))
                    log_p = -dist
                    text_mask = torch.arange(text_padded.size(1), device=device)[None, :] < text_lens[:, None].to(device)
                    mel_mask = torch.arange(mel_padded.size(2), device=device)[None, :] < mel_lens[:, None].to(device)
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
            fp_bp_start = time.time()
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

            step_end = time.time()

            if (batch_idx % 1) == 0:
                total_time = step_end - batch_start
                gpu_time = step_end - fp_bp_start
                sps = BATCH_SIZE / total_time if total_time > 0 else 0.0
                print(f"Epoch {epoch} | Batch {batch_idx:3d} | Loss {loss.item()*GRAD_ACCUM_STEPS:.4f} | "
                      f"Speed: {sps:.2f} items/s | Time: {total_time:.2f}s (GPU {gpu_time:.2f}s)")
                sys.stdout.flush()

            # Unfreeze schedule
            if UNFREEZE_AFTER_STEPS > 0 and (not unfroze):
                global_step += 1
                if took_step and global_step >= UNFREEZE_AFTER_STEPS and FREEZE_LOWER_BLOCKS > 0:
                    for i in range(FREEZE_LOWER_BLOCKS):
                        for p in model.text_encoder.convnext[i].parameters():
                            p.requires_grad = True
                    unfroze = True
                    print(f"Unfroze lower ConvNeXt blocks at global_step={global_step}")
                    sys.stdout.flush()

        # Save checkpoint each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
