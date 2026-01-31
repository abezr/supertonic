"""
Download ONNX models from HuggingFace for Supertonic weight conversion
"""

import os
import urllib.request
from pathlib import Path

def download_file(url, output_path):
    """Download a file with progress indication"""
    print(f"Downloading {os.path.basename(output_path)}...")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if count % 50 == 0 or percent >= 100:
            print(f"  Progress: {percent}%", end='\r')
    
    urllib.request.urlretrieve(url, output_path, progress_hook)
    print(f"  ✓ Downloaded {os.path.basename(output_path)}")

def main():
    print("="*60)
    print("Downloading Supertonic ONNX Models")
    print("="*60)
    print()
    
    # Create onnx directory
    os.makedirs("onnx", exist_ok=True)
    
    base_url = "https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx"
    
    files = [
        ("text_encoder.onnx", "onnx/text_encoder.onnx"),
        ("duration_predictor.onnx", "onnx/duration_predictor.onnx"),
        ("vector_estimator.onnx", "onnx/vector_estimator.onnx"),
        ("vocoder.onnx", "onnx/vocoder.onnx"),
        ("unicode_indexer.json", "onnx/unicode_indexer.json"),
    ]
    
    for filename, output_path in files:
        if os.path.exists(output_path):
            print(f"✓ {filename} already exists, skipping")
            continue
        
        url = f"{base_url}/{filename}"
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")
            if "404" in str(e):
                print(f"    File may not exist on HuggingFace")
            continue
    
    print()
    print("="*60)
    print("Download complete!")
    print("="*60)
    print()
    print("Next step: Run weight conversion")
    print("  python training/onnx_to_pytorch.py --onnx-dir onnx --output pretrained_weights.pt")
    print()

if __name__ == "__main__":
    main()
