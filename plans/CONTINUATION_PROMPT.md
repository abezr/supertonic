# Continuation Prompt: Create Simple Local TTS Dataset Generator

## Context

StyleTTS2 is running locally in Docker at `http://127.0.0.1:7860`. Use this instead of HuggingFace Space to generate the dataset.

## Configuration

```yaml
provider: styletts2
styletts2:
  api_url: http://127.0.0.1:7860
  model_name: multi
  voice: "Марина Панас"
  voice_en: ""
  speed: 1.3
```

## Goal

Create a simple `generate_tts.py` script that:
1. Makes HTTP requests to local Docker Gradio API
2. Saves audio files to `dataset_marina/`
3. Updates `filelist.txt` with paths and text

## Implementation Steps

### Step 1: Create generate_tts.py

```python
import requests
import os
from pathlib import Path

# Configuration
API_URL = "http://127.0.0.1:7860/synthesize"
OUTPUT_DIR = Path("dataset_marina")
VOICE = "Марина Панас"
MODEL_NAME = "multi"
SPEED = 1.3

# Sentences to synthesize
sentences = [
    "Привіт, це тестовий запис голосу Марини Панас.",
    "Сьогодні чудовий день для того, щоб навчити штучний інтелект розмовляти.",
    # ... add all sentences
]

def generate_audio(text, output_path):
    """Call local TTS API and save audio"""
    response = requests.post(
        API_URL,
        json={
            "model_name": MODEL_NAME,
            "text": text,
            "speed": SPEED,
            "voice_name": VOICE
        }
    )
    response.raise_for_status()
    
    # Save audio
    with open(output_path, "wb") as f:
        f.write(response.content)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    filelist_path = OUTPUT_DIR / "filelist.txt"
    
    for i, text in enumerate(sentences):
        filename = f"sample_{i:04d}.wav"
        output_path = OUTPUT_DIR / filename
        
        if output_path.exists():
            print(f"[{i+1}/{len(sentences)}] Skipping {filename}, already exists")
            continue
            
        print(f"[{i+1}/{len(sentences)}] Generating: {text[:30]}...")
        generate_audio(text, output_path)
        
        # Update filelist
        with open(filelist_path, "a", encoding="utf-8") as f:
            f.write(f"{output_path.absolute()}|{text}\n")
        
        print(f"  ✓ Saved {filename}")

if __name__ == "__main__":
    main()
```

### Step 2: Test the Script

1. Ensure Docker container is running
2. Run: `python generate_tts.py`
3. Check `dataset_marina/` for audio files
4. Check `filelist.txt` for entries

## Important Notes

- The Gradio API returns audio file directly (binary)
- No gradio_client needed - just plain `requests`
- Local API means no quota limits
- No proxies needed

## Files to Create

- `generate_tts.py` - Main script

## Dependencies

Only `requests` library needed (already in Python stdlib as urllib, but requests is simpler)