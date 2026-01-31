import torch
import soundfile as sf
from training.model import Supertonic
from training.utils import UnicodeProcessor, get_best_device
import torchaudio

def synthesize():
    device = get_best_device()
    checkpoint_path = "checkpoint_low_vram.pt"
    
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # DIAGNOSTIC: Show which epoch we're loading
    epoch_num = checkpoint.get('epoch', 'unknown')
    print(f"\n=== DIAGNOSTIC INFO ===")
    print(f"Loaded checkpoint from EPOCH: {epoch_num}")
    
    model = Supertonic(
        vocab_size=2000, 
        embed_dim=192,
        hidden_dim=192,
        style_dim=128
    ).to(device)
    
    # DIAGNOSTIC: Check if model weights are actually different from random init
    # This assumes 'text_encoder' and 'embedding' exist in the model structure
    try:
        sample_weight_before = model.text_encoder.embedding.weight.data.mean().item()
    except AttributeError:
        print("Warning: Could not access model.text_encoder.embedding.weight.data.mean() before loading state_dict.")
        sample_weight_before = float('nan') # Indicate it wasn't captured
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Text
    processor = UnicodeProcessor("unicode_indexer_expanded.json")
    text = "Привіт, це тест синтезу мовлення."
    print(f"Synthesizing: {text}")
    
    text_ids = processor(text, "uk") # Using 'uk' for Ukrainian
    text_tensor = torch.from_numpy(text_ids).long().unsqueeze(0).to(device)
    
    # Style (Random for now, or could load from dataset)
    # Ideally should be an embedding from the same distribution as training
    style = torch.randn(1, 128).to(device)
    
    # Inference
    # fixed_duration=15 -> ~160ms/char (Slower, clearer)
    # 256 samples/frame / 24000 sr = 10.6ms per frame. 
    # 15 frames = 160ms.
    duration_val = 15
    print(f"Generating with fixed_duration={duration_val} frames/token...")
    
    # Lang ID (uk=0)
    lang_ids = torch.zeros(1, dtype=torch.long, device=device)
    
    print(f"\n=== GENERATION DIAGNOSTICS ===")
    mel = model.inference(text_tensor, style, lang_ids=lang_ids, n_steps=20, fixed_duration=duration_val)
    
    # DIAGNOSTIC: Check mel statistics
    print(f"Generated Mel shape: {mel.shape}")
    print(f"Mel stats: min={mel.min().item():.4f}, max={mel.max().item():.4f}, mean={mel.mean().item():.4f}, std={mel.std().item():.4f}")
    
    # Check if all values are the same (would indicate model not working)
    mel_variance = mel.var().item()
    print(f"Mel variance: {mel_variance:.6f} (should be > 0.01 for real speech)")
    print(f"===========================\n")
    
    seconds = mel.size(2) * 256 / 24000
    print(f"Generated Audio Duration: {seconds:.2f}s for text length {text_tensor.size(1)}")
    
    # Denormalize (Reverse dataset.py normalization)
    mel = (mel * 2.0) - 4.0
    
    # Invert Log-Mel (Exp)
    mel = torch.exp(mel)
    
    # Vocoder (GriffinLim - Improved Settings)
    print("Vocoding (InverseMel -> GriffinLim with quality settings)...")
    
    # Gentle power compression to reduce artifacts
    mel_compressed = torch.pow(mel, 0.8)
    
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=1024 // 2 + 1,
        n_mels=80,
        sample_rate=24000,
        f_min=0,
        f_max=8000
    ).to(device)
    
    linear_spec = inverse_mel(mel_compressed)
    
    # More iterations = better quality (but slower)
    vocoder = torchaudio.transforms.GriffinLim(
        n_fft=1024, 
        n_iter=128,  # Increased from 32
        hop_length=256,
        win_length=1024,
        power=1.0,  # Use magnitude instead of power
        momentum=0.99  # Better convergence
    ).to(device)
    
    wav = vocoder(linear_spec)
    
    # Normalize to prevent clipping
    wav = wav / (torch.abs(wav).max() + 1e-8)
    wav = wav * 0.95  # Leave headroom
    
    output_path = "output.wav"
    sf.write(output_path, wav.squeeze().cpu().numpy(), 24000)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    synthesize()
