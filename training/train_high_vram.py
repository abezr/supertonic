import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import Supertonic
from .dataset import TTSDataset, collate_fn
from .tokenizer_builder import build_extended_indexer
from .utils import get_best_device, maximum_path
import os
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

def train():
    device = get_best_device()
    print(f"Training on {device} (High VRAM Mode - RTX 5080)")
    
    if device.type == 'cpu':
        print("\n" + "="*60)
        print("WARNING: CUDA not detected! Training will be extremely slow.")
        print("Please install PyTorch with CUDA support:")
        print("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("="*60 + "\n")
    
    # Configuration for 16GB VRAM
    BATCH_SIZE = 32         # Increased from 2
    GRAD_ACCUM_STEPS = 1    # No accumulation needed if BS=32 fits
    LEARNING_RATE = 2e-4    # Slightly higher LR allows faster convergence with larger batch
    EPOCHS = 100
    CHECKPOINT_PATH = "checkpoint_high_vram.pt"
    
    # Enable performance optimizations
    torch.backends.cudnn.benchmark = True
    
    if not os.path.exists("unicode_indexer_expanded.json"):
        print("Building tokenizer...")
        build_extended_indexer()
    
    model = Supertonic(
        vocab_size=2000, 
        embed_dim=192,
        hidden_dim=192,
        style_dim=128
    ).to(device)
    
    # **CRITICAL FIX**: Load pretrained English weights from ONNX
    PRETRAINED_WEIGHTS = "pretrained_weights.pt"
    if os.path.exists(PRETRAINED_WEIGHTS) and not os.path.exists(CHECKPOINT_PATH):
        print(f"\n{'='*60}")
        print(f"🔄 Loading pretrained ONNX weights from {PRETRAINED_WEIGHTS}")
        print(f"{'='*60}")
        from training.onnx_to_pytorch import load_pretrained_into_model
        model = load_pretrained_into_model(model, PRETRAINED_WEIGHTS, freeze_encoder=False)
        print(f"✅ Pretrained weights loaded - starting fine-tuning\n")
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

    # Load Checkpoint
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    if not os.path.exists("dataset_marina/filelist.txt"):
        print("Prepare 'dataset_marina/filelist.txt' by running generate_dataset.py")
        return

    dataset = TTSDataset("dataset_marina/filelist.txt", "unicode_indexer_expanded.json")
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True, 
        num_workers=4, # More workers for faster loading
        pin_memory=True
    )
    
    print(f"Starting training with Batch Size {BATCH_SIZE}...")
    model.train()
    
    for epoch in range(start_epoch, EPOCHS):
        running_loss = 0.0
        
        for batch_idx, (text_padded, text_lens, mel_padded, mel_lens, paths, lang_ids) in enumerate(loader):
            text_padded = text_padded.to(device)
            mel_padded = mel_padded.to(device)
            lang_ids = lang_ids.to(device)
            style = torch.randn(text_padded.size(0), 128).to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                # 1. Text Encoder
                text_emb = model.text_encoder(text_padded, style, lang_ids=lang_ids)
                
                # 2. MAS Alignment
                with torch.no_grad():
                    text_proj = model.vector_estimator.final_proj(text_emb)
                    dist = torch.cdist(text_proj, mel_padded.transpose(1, 2))
                    log_p = -dist
                    
                    text_mask = torch.arange(text_padded.size(1), device=device)[None, :] < text_lens[:, None].to(device)
                    mel_mask = torch.arange(mel_padded.size(2), device=device)[None, :] < mel_lens[:, None].to(device)
                    attn_mask = text_mask.unsqueeze(2) * mel_mask.unsqueeze(1)
                    
                    attn = maximum_path(log_p, attn_mask)
                    target_dur = attn.sum(2)
                    log_target_dur = torch.log(target_dur + 1e-6)

                # 3. Duration Predictor
                log_dur_pred = model.duration_predictor(text_emb, style)
                loss_dur = F.mse_loss(log_dur_pred * text_mask, log_target_dur * text_mask, reduction='sum') / text_mask.sum()
                
                # 4. Upsample
                text_emb_upsampled = torch.bmm(attn.transpose(1, 2), text_emb)
                
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
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")
        
        # Epoch Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"Epoch {epoch} Completed. Avg Loss: {running_loss / len(loader):.4f}")

if __name__ == "__main__":
    train()
