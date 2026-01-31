import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import Supertonic
from .dataset import TTSDataset, collate_fn
from .tokenizer_builder import build_extended_indexer
import os
from torch.cuda.amp import GradScaler, autocast

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Configuration for 4GB VRAM
    BATCH_SIZE = 2 # Start small
    GRAD_ACCUM_STEPS = 8 # Simulate batch size 16
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    CHECKPOINT_PATH = "checkpoint_latest.pt"
    if not os.path.exists("unicode_indexer_expanded.json"):
        print("Building tokenizer...")
        build_extended_indexer()
    
    # 2. Config & Data
    # Assuming standard hyperparameters
    model = Supertonic(
        vocab_size=2000, # Expanded
        embed_dim=192,
        hidden_dim=192,
        style_dim=128
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler() # For Mixed Precision

    # Load Checkpoint if exists
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Dummy filelist check
    if not os.path.exists("filelist_ru.txt"):
        print("Prepare 'filelist_ru.txt' with format: path/to/wav|Text content")
        return

    dataset = TTSDataset("dataset_marina/filelist.txt", "unicode_indexer_expanded.json")
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True, 
        num_workers=4 # Windows/WSL often has issues with multiple workers + cuda
    )
    
    # 3. Training Loop
    model.train()
    
    for epoch in range(start_epoch, EPOCHS):
        for batch_idx, (text_padded, text_lens, mel_padded, mel_lens, paths) in enumerate(loader):
            text_padded = text_padded.to(device)
            mel_padded = mel_padded.to(device)
            # Dummy style vector [B, 128]
            style = torch.randn(text_padded.size(0), 128).to(device)
            
            # dummy style vector [B, 128] - In reality, use style from reference audio
            style = torch.randn(text_padded.size(0), 128).to(device)
            
            # --- Forward Pass (Mixed Precision) ---
            with autocast():
                # 1. Text Encoder
                text_emb = model.text_encoder(text_padded, style)
                
                # 2. Alignment / Duration
                # Simplified: Assuming 1-to-1 or already aligned for this test
                # In real fine-tuning, you often freeze alignment or use external durations
                
                # 3. Flow Matching (Vector Estimator)
                b = mel_padded.size(0)
                t = torch.rand(b, device=device)
                
                x1 = mel_padded.transpose(1, 2) # [B, T, D]
                x0 = torch.randn_like(x1)
                
                t_expand = t.view(b, 1, 1)
                xt = (1 - t_expand) * x0 + t_expand * x1
                
                vt_target = x1 - x0
                
                # Align text_emb to Mel length
                if text_emb.size(1) != x1.size(1):
                    text_emb_upsampled = torch.nn.functional.interpolate(
                        text_emb.transpose(1, 2), size=x1.size(1), mode='linear'
                    ).transpose(1, 2)
                else:
                    text_emb_upsampled = text_emb

                vt_pred = model.vector_estimator(xt, text_emb_upsampled, style, t)
                
                loss = torch.mean((vt_target - vt_pred) ** 2)
                loss = loss / GRAD_ACCUM_STEPS # Scale loss for accumulation

            # Backward
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            if batch_idx % 10 == 0:
                # Multiply loss back for display
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item() * GRAD_ACCUM_STEPS:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"Saved checkpoint to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()
