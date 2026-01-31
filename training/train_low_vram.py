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
        print(f"   Run: python training/onnx_to_pytorch.py --onnx-dir <path>")
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

    dataset = TTSDataset("dataset_marina/filelist.txt", "unicode_indexer_expanded.json")
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True, 
        num_workers=0 # Windows/WSL often has issues with multiple workers + cuda
    )
    
    # 3. Training Loop
    # 3. Training Loop
    model.train()
    print(f"Beginning training. Mel caching enabled in dataset.")
    print(f"Watch 'Iter Time' to verify processing speed.\n")
    
    for epoch in range(start_epoch, EPOCHS):
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (text_padded, text_lens, mel_padded, mel_lens, paths, lang_ids) in enumerate(loader):
            batch_start = time.time()
            text_padded = text_padded.to(device)
            mel_padded = mel_padded.to(device)
            lang_ids = lang_ids.to(device)
            
            # Record Data Loading time (approx)
            data_time = batch_start - start_time if batch_idx == 0 else batch_start - step_end
            
            style = torch.randn(text_padded.size(0), 128).to(device)
            optimizer.zero_grad()
            
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                # 1. Text Encoder
                text_emb = model.text_encoder(text_padded, style, lang_ids=lang_ids)
                
                # 2. Monotonic Alignment Search (FIXED)
                with torch.no_grad():
                    # Project text embeddings to mel space for similarity
                    text_proj = model.vector_estimator.final_proj(text_emb)  # [B, T_text, 80]
                    
                    # Compute similarity matrix (negative distance = log-likelihood)
                    dist = torch.cdist(text_proj, mel_padded.transpose(1, 2))  # [B, T_text, T_mel]
                    log_p = -dist
                    
                    # Create masks
                    text_mask = torch.arange(text_padded.size(1), device=device)[None, :] < text_lens[:, None].to(device)
                    mel_mask = torch.arange(mel_padded.size(2), device=device)[None, :] < mel_lens[:, None].to(device)
                    attn_mask = text_mask.unsqueeze(2) * mel_mask.unsqueeze(1)  # [B, T_text, T_mel]
                    
                    # Find optimal alignment path
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
            fp_bp_start = time.time()
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                # Unscale and clip before stepping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            step_end = time.time()
            
            running_loss += loss.item() * GRAD_ACCUM_STEPS
            
            if batch_idx % 1 == 0: # Print every batch for low vram feedback
                total_time = step_end - batch_start
                gpu_time = step_end - fp_bp_start
                samples_per_sec = BATCH_SIZE / total_time if total_time > 0 else 0.0
                
                print(f"Epoch {epoch} | Batch {batch_idx:3d} | Loss {loss.item()*GRAD_ACCUM_STEPS:.4f} | " 
                      f"Speed: {samples_per_sec:.2f} items/s | "
                      f"Time: {total_time:.2f}s (GPU {gpu_time:.2f}s)")
                sys.stdout.flush()
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"Saved checkpoint to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()
