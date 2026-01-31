import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import Supertonic
from .dataset import TTSDataset, collate_fn
from .tokenizer_builder import build_extended_indexer
from .utils import get_best_device, maximum_path
import os
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import time
import sys

def train():
    device = get_best_device()
    print(f"Training on {device} (Low VRAM Mode)")
    
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
    
    # Configuration for 4GB VRAM (env overridable)
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2)) # Start small
    GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 8)) # Simulate batch size 16
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-4))
    EPOCHS = int(os.environ.get("EPOCHS", 100))
    CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoint_low_vram.pt")
    MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 1.0))
    LOG_EVERY_N_BATCHES = int(os.environ.get("LOG_EVERY_N_BATCHES", 10))
    
    if not os.path.exists("unicode_indexer_expanded.json"):
        print("Building tokenizer...")
        build_extended_indexer()
    
    # 2. Config & Data
    # Assuming standard hyperparameters
    model = Supertonic(
        vocab_size=2000, # Expanded
        embed_dim=192,
        hidden_dim=192,
        style_dim=128,
        encoder_type=os.environ.get("ENCODER_TYPE", "transformer"),
        n_langs=10
    ).to(device)
    
    # Safe pretrained loading guard: only load ONNX-derived weights if embed_dim is 256
    PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "pretrained_weights.pt")
    embed_dim_current = getattr(model.text_encoder.embedding, 'embedding_dim', None)
    if os.path.exists(PRETRAINED_WEIGHTS) and not os.path.exists(CHECKPOINT_PATH):
        if embed_dim_current == 256:
            print(f"\n{'='*60}")
            print(f"🔄 Loading pretrained ONNX weights from {PRETRAINED_WEIGHTS}")
            print(f"{'='*60}")
            from training.onnx_to_pytorch import load_pretrained_into_model
            model = load_pretrained_into_model(model, PRETRAINED_WEIGHTS, freeze_encoder=False)
            print(f"✅ Pretrained weights loaded - starting fine-tuning\n")
        else:
            print(f"\n{'!'*60}")
            print("⛔ Skipping ONNX pretrained load: model embed_dim != 256 (Transformer path).")
            print("   To reuse ONNX English weights, reimplement text encoder as ConvNeXt (embed_dim=256).")
            print(f"{'!'*60}\n")
    elif not os.path.exists(PRETRAINED_WEIGHTS) and not os.path.exists(CHECKPOINT_PATH):
        print(f"\n{'!'*60}")
        print(f"⚠️  WARNING: No pretrained weights found!")
        print(f"   Training from random initialization (not recommended)")
        print(f"   Run: ./venv/bin/python training/onnx_to_pytorch.py --onnx-dir <path>")
        print(f"{'!'*60}\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Scaler for AMP
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Load Checkpoint if exists (overrides pretrained weights)
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Dummy filelist check
    if not os.path.exists("dataset_marina/filelist.txt"):
        print("Prepare 'dataset_marina/filelist.txt' by running generate_dataset.py")
        return

    print(f"DEBUG: Creating TTSDataset...")
    sys.stdout.flush()
    dataset = TTSDataset("dataset_marina/filelist.txt", "unicode_indexer_expanded.json")
    print(f"DEBUG: Dataset created with {len(dataset)} items")
    sys.stdout.flush()
    
    print(f"DEBUG: Creating DataLoader with batch_size={BATCH_SIZE}, num_workers=4...")
    sys.stdout.flush()
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True, 
        num_workers=0,  # DEBUG: Set to 0 to avoid multiprocessing issues
        pin_memory=False  # DEBUG: Disable pin_memory for debugging
    )
    print(f"DEBUG: DataLoader created")
    sys.stdout.flush()
    
    # 3. Training Loop
    model.train()
    print(f"Beginning training. Mel caching enabled in dataset.")
    print(f"Watch 'Iter Time' to verify processing speed.\n")
    
    # Rolling average timing instrumentation
    timing_window_size = LOG_EVERY_N_BATCHES
    data_times = []
    fwd_bwd_times = []
    total_times = []
    
    print(f"DEBUG: Starting training loop for {EPOCHS} epochs, starting from epoch {start_epoch}")
    sys.stdout.flush()
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"DEBUG: Starting epoch {epoch}")
        sys.stdout.flush()
        running_loss = 0.0
        data_start = time.time()
        
        print(f"DEBUG: About to iterate DataLoader (num_workers={loader.num_workers}, pin_memory={loader.pin_memory})")
        sys.stdout.flush()
        
        for batch_idx, (text_padded, text_lens, mel_padded, mel_lens, paths, lang_ids) in enumerate(loader):
            if batch_idx == 0:
                print(f"DEBUG: First batch received! batch_idx={batch_idx}")
                sys.stdout.flush()
            if batch_idx % LOG_EVERY_N_BATCHES == 0:
                print(f"DEBUG: Processing batch {batch_idx}")
                sys.stdout.flush()
            batch_start = time.time()
            data_time = batch_start - data_start
            
            text_padded = text_padded.to(device, non_blocking=True)
            mel_padded = mel_padded.to(device, non_blocking=True)
            lang_ids = lang_ids.to(device, non_blocking=True)
            
            style = torch.randn(text_padded.size(0), 128, device=device)
            optimizer.zero_grad()
            
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            fwd_bwd_start = time.time()
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                # 1. Text Encoder
                text_emb = model.text_encoder(text_padded, style, lang_ids=lang_ids)
                
                # 2. Monotonic Alignment Search (FIXED) - Build masks on device
                T_text = text_padded.size(1)
                T_mel = mel_padded.size(2)
                idx_text = torch.arange(T_text, device=device)[None, :]
                idx_mel = torch.arange(T_mel, device=device)[None, :]
                text_mask = idx_text < text_lens[:, None].to(device, non_blocking=True)
                mel_mask = idx_mel < mel_lens[:, None].to(device, non_blocking=True)
                
                with torch.no_grad():
                    # Project text embeddings to mel space for similarity
                    text_proj = model.vector_estimator.final_proj(text_emb)  # [B, T_text, 80]
                    
                    # Compute similarity matrix (negative distance = log-likelihood) - stays on device
                    log_p = -torch.cdist(text_proj, mel_padded.transpose(1, 2))  # [B, T_text, T_mel]
                    
                    # Create attention mask and find optimal alignment path - both on device
                    attn_mask = text_mask.unsqueeze(2) * mel_mask.unsqueeze(1)  # [B, T_text, T_mel]
                    attn = maximum_path(log_p, attn_mask)  # [B, T_text, T_mel]
                    
                    # Extract duration targets from alignment
                    target_dur = attn.sum(2)  # [B, T_text]
                    log_target_dur = torch.log(target_dur + 1e-6)
                
                # 3. Duration Loss
                log_dur_pred = model.duration_predictor(text_emb, style)
                loss_dur = F.mse_loss(log_dur_pred * text_mask, log_target_dur * text_mask, reduction='sum') / text_mask.sum()
                
                # 4. Upsample using MAS attention matrix
                text_emb_upsampled = torch.bmm(attn.transpose(1, 2), text_emb)  # [B, T_mel, D]
                
                # 5. Flow Matching
                b = mel_padded.size(0)
                t = torch.rand(b, device=device)
                
                x1 = mel_padded
                x0 = torch.randn_like(x1)
                t_expanded = t[:, None, None].expand_as(x1)
                xt = t_expanded * x1 + (1 - t_expanded) * x0
                
                vt_target = x1 - x0
                vt_pred = model.vector_estimator(xt.transpose(1, 2), text_emb_upsampled, style, t)
                
                loss_fm = torch.mean((vt_target.transpose(1, 2) - vt_pred) ** 2)
                loss = loss_fm + loss_dur
                loss = loss / GRAD_ACCUM_STEPS

            # Backward / Step
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                # Unscale and clip before stepping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            step_end = time.time()
            fwd_bwd_time = step_end - fwd_bwd_start
            total_time = step_end - batch_start
            
            running_loss += loss.item() * GRAD_ACCUM_STEPS
            
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
                gpu_util_hint = avg_fwd_bwd / avg_total if avg_total > 0 else 0.0
                
                print(f"Epoch {epoch} | Batch {batch_idx:3d} | Loss {loss.item()*GRAD_ACCUM_STEPS:.4f} | "
                      f"Speed: {sps:.2f} items/s | "
                      f"Time: {total_time:.2f}s (GPU {fwd_bwd_time:.2f}s)")
                
                # Print perf summary every N batches
                if batch_idx > 0:
                    print(f"[Perf] data={avg_data:.3f}s, fwd_bwd={avg_fwd_bwd:.3f}s, total={avg_total:.3f}s, "
                          f"gpu_util_hint={gpu_util_hint:.2%}")
                
                sys.stdout.flush()
            
            # Reset data_start for next iteration
            data_start = time.time()
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"Saved checkpoint to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()
