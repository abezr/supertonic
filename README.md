# Supertonic — Lightning Fast, On-Device TTS

[![v2 Demo](https://img.shields.io/badge/🤗%20v2-Demo-yellow)](https://huggingface.co/spaces/Supertone/supertonic-2)
[![v2 Models](https://img.shields.io/badge/🤗%20v2-Models-blue)](https://huggingface.co/Supertone/supertonic-2)
[![v1 Demo](https://img.shields.io/badge/🤗%20v1%20(old)-Demo-lightgrey)](https://huggingface.co/spaces/Supertone/supertonic#interactive-demo)
[![v1 Models](https://img.shields.io/badge/🤗%20v1%20(old)-Models-lightgrey)](https://huggingface.co/Supertone/supertonic)

<p align="center">
  <img src="img/supertonic_preview_0.1.jpg" alt="Supertonic Banner">
</p>

**Supertonic** is a lightning-fast, on-device text-to-speech system designed for **extreme performance** with minimal computational overhead. Powered by ONNX Runtime, it runs entirely on your device—no cloud, no API calls, no privacy concerns.

### 📰 Update News

- **2026.01.22** - **[Voice Builder](https://supertonic.supertone.ai/voice_builder)** is now live! Turn your voice into a deployable, edge-native TTS with permanent ownership.

<p align="center">
    <img src="img/voicebuilder_img.png" alt="Voice Builder" width="600">
</p>

- **2026.01.06** - 🎉 **Supertonic 2** released with multilingual support! Now supports English (`en`), Korean (`ko`), Spanish (`es`), Portuguese (`pt`), and French (`fr`). [Demo](https://huggingface.co/spaces/Supertone/supertonic-2) | [Models](https://huggingface.co/Supertone/supertonic-2)
- **2025.12.10** - Added `supertonic` PyPI package! Install via `pip install supertonic`. For details, visit [supertonic-py documentation](https://supertone-inc.github.io/supertonic-py)
- **2025.12.10** - Added [6 new voice styles](https://huggingface.co/Supertone/supertonic/tree/b10dbaf18b316159be75b34d24f740008fddd381) (M3, M4, M5, F3, F4, F5). See [Voices](https://supertone-inc.github.io/supertonic-py/voices/) for details
- **2025.12.08** - Optimized ONNX models via [OnnxSlim](https://github.com/inisis/OnnxSlim) now available on [Hugging Face Models](https://huggingface.co/Supertone/supertonic)
- **2025.11.24** - Added Flutter SDK support with macOS compatibility

### Table of Contents

- [Demo](#demo)
- [Why Supertonic?](#why-supertonic)
- [Language Support](#language-support)
- [Getting Started](#getting-started)
- [Performance](#performance)
- [Built with Supertonic](#built-with-supertonic)
- [Citation](#citation)
- [License](#license)

## Demo

### Raspberry Pi

Watch Supertonic running on a **Raspberry Pi**, demonstrating on-device, real-time text-to-speech synthesis:

https://github.com/user-attachments/assets/ea66f6d6-7bc5-4308-8a88-1ce3e07400d2

### E-Reader

Experience Supertonic on an **Onyx Boox Go 6** e-reader in airplane mode, achieving an average RTF of 0.3× with zero network dependency:

https://github.com/user-attachments/assets/64980e58-ad91-423a-9623-78c2ffc13680

### Chrome Extension

Turns any webpage into audio in under one second, delivering lightning-fast, on-device text-to-speech with zero network dependency—free, private, and effortless:

https://github.com/user-attachments/assets/cc8a45fc-5c3e-4b2c-8439-a14c3d00d91c

---

> 🎧 **Try it now**: Experience Supertonic in your browser with our [**Interactive Demo**](https://huggingface.co/spaces/Supertone/supertonic-2), or get started with pre-trained models from [**Hugging Face Hub**](https://huggingface.co/Supertone/supertonic-2)

## Why Supertonic?

- **⚡ Blazingly Fast**: Generates speech up to **167× faster than real-time** on consumer hardware (M4 Pro)—unmatched by any other TTS system
- **🪶 Ultra Lightweight**: Only **66M parameters**, optimized for efficient on-device performance with minimal footprint
- **📱 On-Device Capable**: **Complete privacy** and **zero latency**—all processing happens locally on your device
- **🎨 Natural Text Handling**: Seamlessly processes numbers, dates, currency, abbreviations, and complex expressions without pre-processing
- **⚙️ Highly Configurable**: Adjust inference steps, batch processing, and other parameters to match your specific needs
- **🧩 Flexible Deployment**: Deploy seamlessly across servers, browsers, and edge devices with multiple runtime backends.

## Language Support

We provide ready-to-use TTS inference examples across multiple ecosystems:

| Language/Platform | Path | Description |
|-------------------|------|-------------|
| [**Python**](py/) | `py/` | ONNX Runtime inference |
| [**Node.js**](nodejs/) | `nodejs/` | Server-side JavaScript |
| [**Browser**](web/) | `web/` | WebGPU/WASM inference |
| [**Java**](java/) | `java/` | Cross-platform JVM |
| [**C++**](cpp/) | `cpp/` | High-performance C++ |
| [**C#**](csharp/) | `csharp/` | .NET ecosystem |
| [**Go**](go/) | `go/` | Go implementation |
| [**Swift**](swift/) | `swift/` | macOS applications |
| [**iOS**](ios/) | `ios/` | Native iOS apps |
| [**Rust**](rust/) | `rust/` | Memory-safe systems |
| [**Flutter**](flutter/) | `flutter/` | Cross-platform apps |

> For detailed usage instructions, please refer to the README.md in each language directory.

## Getting Started

First, clone the repository:

```bash
git clone https://github.com/supertone-inc/supertonic.git
cd supertonic
```

### Prerequisites

Before running the examples, download the ONNX models and preset voices, and place them in the `assets` directory:

> **Note:** The Hugging Face repository uses Git LFS. Please ensure Git LFS is installed and initialized before cloning or pulling large model files.
> - macOS: `brew install git-lfs && git lfs install`
> - Generic: see `https://git-lfs.com` for installers

```bash
git clone https://huggingface.co/Supertone/supertonic-2 assets
```

### Quick Start

**Python Example** ([Details](py/))
```bash
cd py
uv sync
uv run example_onnx.py
```

**Node.js Example** ([Details](nodejs/))
```bash
cd nodejs
npm install
npm start
```

**Browser Example** ([Details](web/))
```bash
cd web
npm install
npm run dev
```

**Java Example** ([Details](java/))
```bash
cd java
mvn clean install
mvn exec:java
```

**C++ Example** ([Details](cpp/))
```bash
cd cpp
mkdir build && cd build
cmake .. && cmake --build . --config Release
./example_onnx
```

**C# Example** ([Details](csharp/))
```bash
cd csharp
dotnet restore
dotnet run
```

**Go Example** ([Details](go/))
```bash
cd go
go mod download
go run example_onnx.go helper.go
```

**Swift Example** ([Details](swift/))
```bash
cd swift
swift build -c release
.build/release/example_onnx
```

**Rust Example** ([Details](rust/))
```bash
cd rust
cargo build --release
./target/release/example_onnx
```

**iOS Example** ([Details](ios/))
```bash
cd ios/ExampleiOSApp
xcodegen generate
open ExampleiOSApp.xcodeproj
```
- In Xcode: Targets → ExampleiOSApp → Signing: select your Team
- Choose your iPhone as run destination → Build & Run


### Technical Details

- **Runtime**: ONNX Runtime for cross-platform inference (CPU-optimized; GPU mode is not tested)
- **Browser Support**: onnxruntime-web for client-side inference
- **Batch Processing**: Supports batch inference for improved throughput
- **Audio Output**: Outputs 16-bit WAV files

## Performance

We evaluated Supertonic's performance (with 2 inference steps) using two key metrics across input texts of varying lengths: Short (59 chars), Mid (152 chars), and Long (266 chars).

**Metrics:**
- **Characters per Second**: Measures throughput by dividing the number of input characters by the time required to generate audio. Higher is better.
- **Real-time Factor (RTF)**: Measures the time taken to synthesize audio relative to its duration. Lower is better (e.g., RTF of 0.1 means it takes 0.1 seconds to generate one second of audio).

### Characters per Second
| System | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|--------|-----------------|----------------|-----------------|
| **Supertonic** (M4 pro - CPU) | 912 | 1048 | 1263 |
| **Supertonic** (M4 pro - WebGPU) | 996 | 1801 | 2509 |
| **Supertonic** (RTX4090) | 2615 | 6548 | 12164 |
| `API` [ElevenLabs Flash v2.5](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) | 144 | 209 | 287 |
| `API` [OpenAI TTS-1](https://platform.openai.com/docs/guides/text-to-speech) | 37 | 55 | 82 |
| `API` [Gemini 2.5 Flash TTS](https://ai.google.dev/gemini-api/docs/speech-generation) | 12 | 18 | 24 |
| `API` [Supertone Sona speech 1](https://docs.supertoneapi.com/en/api-reference/endpoints/text-to-speech) | 38 | 64 | 92 |
| `Open` [Kokoro](https://github.com/hexgrad/kokoro/) | 104 | 107 | 117 |
| `Open` [NeuTTS Air](https://github.com/neuphonic/neutts-air) | 37 | 42 | 47 |

> **Notes:**  
> `API` = Cloud-based API services (measured from Seoul)  
> `Open` = Open-source models  
> Supertonic (M4 pro - CPU) and (M4 pro - WebGPU): Tested with ONNX  
> Supertonic (RTX4090): Tested with PyTorch model  
> Kokoro: Tested on M4 Pro CPU with ONNX  
> NeuTTS Air: Tested on M4 Pro CPU with Q8-GGUF

### Real-time Factor

| System | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|--------|-----------------|----------------|-----------------|
| **Supertonic** (M4 pro - CPU) | 0.015 | 0.013 | 0.012 |
| **Supertonic** (M4 pro - WebGPU) | 0.014 | 0.007 | 0.006 |
| **Supertonic** (RTX4090) | 0.005 | 0.002 | 0.001 |
| `API` [ElevenLabs Flash v2.5](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) | 0.133 | 0.077 | 0.057 |
| `API` [OpenAI TTS-1](https://platform.openai.com/docs/guides/text-to-speech) | 0.471 | 0.302 | 0.201 |
| `API` [Gemini 2.5 Flash TTS](https://ai.google.dev/gemini-api/docs/speech-generation) | 1.060 | 0.673 | 0.541 |
| `API` [Supertone Sona speech 1](https://docs.supertoneapi.com/en/api-reference/endpoints/text-to-speech) | 0.372 | 0.206 | 0.163 |
| `Open` [Kokoro](https://github.com/hexgrad/kokoro/) | 0.144 | 0.124 | 0.126 |
| `Open` [NeuTTS Air](https://github.com/neuphonic/neutts-air) | 0.390 | 0.338 | 0.343 |

<details>
<summary><b>Additional Performance Data (5-step inference)</b></summary>

<br>

**Characters per Second (5-step)**

| System | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|--------|-----------------|----------------|-----------------|
| **Supertonic** (M4 pro - CPU) | 596 | 691 | 850 |
| **Supertonic** (M4 pro - WebGPU) | 570 | 1118 | 1546 |
| **Supertonic** (RTX4090) | 1286 | 3757 | 6242 |

**Real-time Factor (5-step)**

| System | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|--------|-----------------|----------------|-----------------|
| **Supertonic** (M4 pro - CPU) | 0.023 | 0.019 | 0.018 |
| **Supertonic** (M4 pro - WebGPU) | 0.024 | 0.012 | 0.010 |
| **Supertonic** (RTX4090) | 0.011 | 0.004 | 0.002 |

</details>

### Natural Text Handling

Supertonic is designed to handle complex, real-world text inputs that contain numbers, currency symbols, abbreviations, dates, and proper nouns.

> 🎧 **View audio samples more easily**: Check out our [**Interactive Demo**](https://huggingface.co/spaces/Supertone/supertonic#text-handling) for a better viewing experience of all audio examples

**Overview of Test Cases:**

| Category | Key Challenges | Supertonic | ElevenLabs | OpenAI | Gemini | Microsoft |
|:--------:|:--------------:|:----------:|:----------:|:------:|:------:|:---------:|
| Financial Expression | Decimal currency, abbreviated magnitudes (M, K), currency symbols, currency codes | ✅ | ❌ | ❌ | ❌ | ❌ |
| Time and Date | Time notation, abbreviated weekdays/months, date formats | ✅ | ❌ | ❌ | ❌ | ❌ |
| Phone Number | Area codes, hyphens, extensions (ext.) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Technical Unit | Decimal numbers with units, abbreviated technical notations | ✅ | ❌ | ❌ | ❌ | ❌ |

<details>
<summary><b>Example 1: Financial Expression</b></summary>

<br>

**Text:**
> "The startup secured **$5.2M** in venture capital, a huge leap from their initial **$450K** seed round."

**Challenges:**
- Decimal point in currency ($5.2M should be read as "five point two million")
- Abbreviated magnitude units (M for million, K for thousand)
- Currency symbol ($) that needs to be properly pronounced as "dollars"

**Audio Samples:**

| System | Result | Audio Sample |
|--------|--------|--------------|
| **Supertonic** | ✅ | [🎧 Play Audio](https://drive.google.com/file/d/1eancUOhiSXCVoTu9ddh4S-OcVQaWrPV-/view?usp=sharing) |
| ElevenLabs Flash v2.5 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1-r2scv7XQ1crIDu6QOh3eqVl445W6ap_/view?usp=sharing) |
| OpenAI TTS-1 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1MFDXMjfmsAVOqwPx7iveS0KUJtZvcwxB/view?usp=sharing) |
| Gemini 2.5 Flash TTS | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1dEHpNzfMUucFTJPQK0k4RcFZvPwQTt09/view?usp=sharing) |
| VibeVoice Realtime 0.5B | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1b69XWBQnSZZ0WZeR3avv7E8mSdoN6p6P/view?usp=sharing) |

</details>

<details>
<summary><b>Example 2: Time and Date</b></summary>

<br>

**Text:**
> "The train delay was announced at **4:45 PM** on **Wed, Apr 3, 2024** due to track maintenance."

**Challenges:**
- Time expression with PM notation (4:45 PM)
- Abbreviated weekday (Wed)
- Abbreviated month (Apr)
- Full date format (Apr 3, 2024)

**Audio Samples:**

| System | Result | Audio Sample |
|--------|--------|--------------|
| **Supertonic** | ✅ | [🎧 Play Audio](https://drive.google.com/file/d/1ehkZU8eiizBenG2DgR5tzBGQBvHS0Uaj/view?usp=sharing) |
| ElevenLabs Flash v2.5 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1ta3r6jFyebmA-sT44l8EaEQcMLVmuOEr/view?usp=sharing) |
| OpenAI TTS-1 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1sskmem9AzHAQ3Hv8DRSZoqX_pye-CXuU/view?usp=sharing) |
| Gemini 2.5 Flash TTS | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1zx9X8oMsLMXW0Zx_SURoqjju-By2yh_n/view?usp=sharing) |
| VibeVoice Realtime 0.5B | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1ZpGEstZr4hA0EdAWBMCUFFWuAkIpYsVh/view?usp=sharing) |

</details>

<details>
<summary><b>Example 3: Phone Number</b></summary>

<br>

**Text:**
> "You can reach the hotel front desk at **(212) 555-0142 ext. 402** anytime."

**Challenges:**
- Area code in parentheses that should be read as separate digits
- Phone number with hyphen separator (555-0142)
- Abbreviated extension notation (ext.)
- Extension number (402)

**Audio Samples:**

| System | Result | Audio Sample |
|--------|--------|--------------|
| **Supertonic** | ✅ | [🎧 Play Audio](https://drive.google.com/file/d/1z-e5iTsihryMR8ll1-N1YXkB2CIJYJ6F/view?usp=sharing) |
| ElevenLabs Flash v2.5 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1HAzVXFTZfZm0VEK2laSpsMTxzufcuaxA/view?usp=sharing) |
| OpenAI TTS-1 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/15tjfAmb3GbjP_kmvD7zSdIWkhtAaCPOg/view?usp=sharing) |
| Gemini 2.5 Flash TTS | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1BCL8n7yligUZyso970ud7Gf5NWb1OhKD/view?usp=sharing) |
| VibeVoice Realtime 0.5B | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1c0c0YM_Qm7XxSk2uSVYLbITgEDTqaVzL/view?usp=sharing) |

</details>

<details>
<summary><b>Example 4: Technical Unit</b></summary>

<br>

**Text:**
> "Our drone battery lasts **2.3h** when flying at **30kph** with full camera payload."

**Challenges:**
- Decimal time duration with abbreviation (2.3h = two point three hours)
- Speed unit with abbreviation (30kph = thirty kilometers per hour)
- Technical abbreviations (h for hours, kph for kilometers per hour)
- Technical/engineering context requiring proper pronunciation

**Audio Samples:**

| System | Result | Audio Sample |
|--------|--------|--------------|
| **Supertonic** | ✅ | [🎧 Play Audio](https://drive.google.com/file/d/1kvOBvswFkLfmr8hGplH0V2XiMxy1shYf/view?usp=sharing) |
| ElevenLabs Flash v2.5 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1_SzfjWJe5YEd0t3R7DztkYhHcI_av48p/view?usp=sharing) |
| OpenAI TTS-1 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1P5BSilj5xFPTV2Xz6yW5jitKZohO9o-6/view?usp=sharing) |
| Gemini 2.5 Flash TTS | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1GU82SnWC50OvC8CZNjhxvNZFKQb7I9_Y/view?usp=sharing) |
| VibeVoice Realtime 0.5B | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1lUTrxrAQy_viEK2Hlu3KLLtTCe8jvbdV/view?usp=sharing) |

</details>

> **Note:** These samples demonstrate how each system handles text normalization and pronunciation of complex expressions **without requiring pre-processing or phonetic annotations**.

## Built with Supertonic

| Project | Description | Links |
|---------|-------------|-------|
| **TLDRL** | Free, on-device TTS extension for reading any webpage | [Chrome](https://chromewebstore.google.com/detail/tldrl-lightning-tts-power/mdbiaajonlkomihpcaffhkagodbcgbme) |
| **Read Aloud** | Open-source TTS browser extension | [Chrome](https://chromewebstore.google.com/detail/read-aloud-a-text-to-spee/hdhinadidafjejdhmfkjgnolgimiaplp) · [Edge](https://microsoftedge.microsoft.com/addons/detail/read-aloud-a-text-to-spe/pnfonnnmfjnpfgagnklfaccicnnjcdkm) · [GitHub](https://github.com/ken107/read-aloud) |
| **PageEcho** | E-Book reader app for iOS | [App Store](https://apps.apple.com/us/app/pageecho/id6755965837) |
| **VoiceChat** | On-device voice-to-voice LLM chatbot in the browser | [Demo](https://huggingface.co/spaces/RickRossTN/ai-voice-chat) · [GitHub](https://github.com/irelate-ai/voice-chat) |
| **OmniAvatar** | Talking avatar video generator from photo + speech | [Demo](https://huggingface.co/spaces/alexnasa/OmniAvatar) |
| **CopiloTTS** | Kotlin Multiplatform TTS SDK via ONNX Runtime | [GitHub](https://github.com/sigmadeltasoftware/CopiloTTS) |
| **Voice Mixer** | PyQt5 tool for mixing and modifying voice styles | [GitHub](https://github.com/Topping1/Supertonic-Voice-Mixer) |
| **Supertonic MNN** | Lightweight library based on MNN (fp32/fp16/int8) | [GitHub](https://github.com/vra/supertonic-mnn) · [PyPI](https://pypi.org/project/supertonic-mnn/) |
| **Transformers.js** | Hugging Face's JS library with Supertonic support | [GitHub PR](https://github.com/huggingface/transformers.js/pull/1459) · [Demo](https://huggingface.co/spaces/webml-community/Supertonic-TTS-WebGPU) |
| **Pinokio** | 1-click localhost cloud for Mac, Windows, and Linux | [Pinokio](https://pinokio.co/) · [GitHub](https://github.com/SUP3RMASS1VE/SuperTonic-TTS) |

## Citation

The following papers describe the core technologies used in Supertonic. If you use this system in your research or find these techniques useful, please consider citing the relevant papers:

### SupertonicTTS: Main Architecture

This paper introduces the overall architecture of SupertonicTTS, including the speech autoencoder, flow-matching based text-to-latent module, and efficient design choices.

```bibtex
@article{kim2025supertonic,
  title={SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System},
  author={Kim, Hyeongju and Yang, Jinhyeok and Yu, Yechan and Ji, Seunghun and Morton, Jacob and Bous, Frederik and Byun, Joon and Lee, Juheon},
  journal={arXiv preprint arXiv:2503.23108},
  year={2025},
  url={https://arxiv.org/abs/2503.23108}
}
```

### Length-Aware RoPE: Text-Speech Alignment

This paper presents Length-Aware Rotary Position Embedding (LARoPE), which improves text-speech alignment in cross-attention mechanisms.

```bibtex
@article{kim2025larope,
  title={Length-Aware Rotary Position Embedding for Text-Speech Alignment},
  author={Kim, Hyeongju and Lee, Juheon and Yang, Jinhyeok and Morton, Jacob},
  journal={arXiv preprint arXiv:2509.11084},
  year={2025},
  url={https://arxiv.org/abs/2509.11084}
}
```

### Self-Purifying Flow Matching: Training with Noisy Labels

This paper describes the self-purification technique for training flow matching models robustly with noisy or unreliable labels.

```bibtex
@article{kim2025spfm,
  title={Training Flow Matching Models with Reliable Labels via Self-Purification},
  author={Kim, Hyeongju and Yu, Yechan and Yi, June Young and Lee, Juheon},
  journal={arXiv preprint arXiv:2509.19091},
  year={2025},
  url={https://arxiv.org/abs/2509.19091}
}
```

## ConvNeXt Fine-Tuning, Orchestration, and GPU Utilization

This section documents the training pipeline for fine-tuning Supertonic's ConvNeXt encoder with optimized GPU utilization for low VRAM environments (4GB).

### Quick Start

#### Converting ONNX to PyTorch (if needed)

If you have ONNX models and need to convert them to PyTorch format for fine-tuning:

```bash
./venv/bin/python training/onnx_to_pytorch.py --onnx-dir onnx --output pretrained_weights.pt
```

#### Single-Epoch Smoke Test

Run a quick test to verify your setup works:

```bash
COMPILE_MODEL=1 CONVNEXT_BLOCKS=6 PRETRAINED_WEIGHTS=pretrained_weights.pt \
BATCH_SIZE=4 GRAD_ACCUM_STEPS=6 MAX_GRAD_NORM=1.0 \
WARMUP_STEPS=1500 EPOCHS=1 \
./venv/bin/python tools/finetune_convnext.py
```

#### Full Orchestrator Command with Loss Gate

For production training with automatic phase advancement based on loss thresholds:

```bash
PHASES_JSON=phases_loss_gate.json CONVNEXT_BLOCKS=6 \
PRETRAINED_WEIGHTS=pretrained_weights.pt \
./venv/bin/python tools/phase_orchestrator_v2.py
```

### Environment Variables

The following environment variables control the fine-tuning behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `CONVNEXT_BLOCKS` | `6` | Number of ConvNeXt blocks in the encoder |
| `PRETRAINED_WEIGHTS` | `pretrained_weights.pt` | Path to pretrained weights file |
| `BATCH_SIZE` | `4` | Per-device batch size (reduce for lower VRAM) |
| `GRAD_ACCUM_STEPS` | `6` | Gradient accumulation steps (effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS) |
| `COMPILE_MODEL` | `0` | Enable torch.compile for faster training (Ampere+ GPUs) |
| `MAX_GRAD_NORM` | `1.0` | Maximum gradient norm for clipping (0.5–1.0 recommended) |
| `WARMUP_STEPS` | `1500` | Number of warmup steps for learning rate scheduling |
| `EPOCHS` | `100` | Total training epochs |
| `LOG_EVERY_N_BATCHES` | `10` | Logging frequency (higher = less overhead) |
| `LEARNING_RATE` | `3e-5` | Main learning rate |
| `LR_BASE` | `5e-6` | Learning rate for base encoder parameters |
| `LR_NEW` | `5e-5` | Learning rate for new parameters (embeddings, projections) |
| `FREEZE_LOWER_BLOCKS` | `0` | Number of lower ConvNeXt blocks to freeze |
| `UNFREEZE_AFTER_STEPS` | `0` | Steps after which to unfreeze frozen blocks |
| `CHECKPOINT_PATH` | `checkpoint_convnext.pt` | Path to save/load checkpoints |
| `DIAG_GPU` | `0` | Enable CUDA event timing for profiling |
| `DEV_STRESS` | `0` | Run GPU stress test before training |

### Low VRAM Configuration (4GB)

For GPUs with 4GB VRAM (e.g., RTX 3050), use these optimized configurations:

#### Recommended: Optimized Throughput (RTX 3050)
```bash
COMPILE_MODEL=1
BATCH_SIZE=4
GRAD_ACCUM_STEPS=6
MAX_GRAD_NORM=1.0
WARMUP_STEPS=1500
```
**Effective batch size: 24** | Higher throughput with torch.compile

#### Alternative: Conservative (Lower VRAM)
```bash
COMPILE_MODEL=0
BATCH_SIZE=2
GRAD_ACCUM_STEPS=10
MAX_GRAD_NORM=1.0
WARMUP_STEPS=1500
```
**Effective batch size: 20** | Use if OOM errors occur

#### Alternative: Higher Throughput (if VRAM permits)
```bash
COMPILE_MODEL=1
BATCH_SIZE=3
GRAD_ACCUM_STEPS=8
MAX_GRAD_NORM=0.5
WARMUP_STEPS=1500
```
**Effective batch size: 24** | More stable with gradient clipping

**Notes:**
- Effective batch size = BATCH_SIZE × GRAD_ACCUM_STEPS
- `COMPILE_MODEL=1` enables torch.compile (Ampere+ GPUs like RTX 3050)
- Higher BATCH_SIZE with lower GRAD_ACCUM_STEPS = better GPU utilization
- Monitor GPU memory usage and reduce BATCH_SIZE if OOM errors occur
- MAX_GRAD_NORM=0.5 provides more stable training but may converge slower

### WSL Dataset Optimization

For WSL users, migrate your dataset from Windows mounts to WSL native filesystem for 5-10x faster I/O:

```bash
# Migrate dataset to WSL native path
python tools/migrate_dataset_to_wsl.py --source /mnt/d/study/ai/supertonic/dataset_marina

# Or specify custom destination and filelist
python tools/migrate_dataset_to_wsl.py \
  --source /mnt/d/study/ai/supertonic/dataset_marina \
  --dest ~/datasets/dataset_marina \
  --filelist /mnt/d/study/ai/supertonic/dataset_marina/filelist.txt

# Dry run to preview what would be copied
python tools/migrate_dataset_to_wsl.py --source /mnt/d/study/ai/supertonic/dataset_marina --dry-run

# After migration, use the new dataset path in training
COMPILE_MODEL=1 BATCH_SIZE=4 GRAD_ACCUM_STEPS=6 \
  ./venv/bin/python tools/finetune_convnext.py
```

See `tools/migrate_dataset_to_wsl.py --help` for more options.

### Phase Orchestrator v2

The Phase Orchestrator v2 automates multi-phase training with intelligent advancement criteria.

#### Example phases_loss_gate.json

```json
[
    {
        "name": "warmup",
        "config": {
            "EPOCHS_PER_CALL": 5,
            "LEARNING_RATE": 3e-5,
            "LR_BASE": 5e-6,
            "LR_NEW": 5e-5,
            "FREEZE_LOWER_BLOCKS": 0
        },
        "criteria": {
            "min_steps": 1500,
            "loss_median_below": 12.0
        }
    },
    {
        "name": "main_training",
        "config": {
            "EPOCHS_PER_CALL": 10,
            "LEARNING_RATE": 3e-5,
            "LR_BASE": 5e-6,
            "LR_NEW": 5e-5,
            "FREEZE_LOWER_BLOCKS": 0
        },
        "criteria": {
            "min_steps": 15000,
            "loss_median_below": 8.0
        }
    }
]
```

#### Loss-Gated Advancement

Each phase has criteria that must be met before advancing:

- `min_steps`: Minimum cumulative optimizer steps required
- `loss_median_below`: Median loss must drop below this threshold

Both conditions must be satisfied. The orchestrator will continue running the current phase until criteria are met.

#### Auto-Retry Behavior

When resuming from a checkpoint where the saved epoch exceeds `EPOCHS_PER_CALL`:

1. Orchestrator detects no batches were processed
2. Automatically increases `EPOCHS_PER_CALL` (up to 2 retries)
3. Retries training chunk with adjusted epoch count

### GPU Utilization Optimizations

The training pipeline includes several optimizations for efficient GPU usage:

#### Device-Only Mask Building

- All masks (text, mel, attention) are built directly on the target device
- Eliminates CPU→GPU transfer overhead for mask tensors
- MAS (Monotonic Alignment Search) operations stay entirely on GPU

#### DataLoader Tuning

```python
DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=1,      # Single worker to reduce memory overhead
    pin_memory=True,    # Faster CPU→GPU transfers
    shuffle=True
)
```

#### Rolling Average Timing Instrumentation

Training logs include performance metrics:

```
[Perf] data=0.023s, fwd_bwd=0.156s, total=0.189s, gpu_util_hint=82.5%
```

- `data`: Data loading time
- `fwd_bwd`: Forward/backward pass time
- `total`: Total iteration time
- `gpu_util_hint`: Ratio of compute time to total time

#### Reduced Python Overhead

- `LOG_EVERY_N_BATCHES=10` (or higher) reduces logging frequency
- Non-blocking data transfers with `non_blocking=True`
- TF32 enabled on Ampere+ GPUs for faster matrix operations

### Example Commands

```bash
# Convert ONNX (if needed)
./venv/bin/python training/onnx_to_pytorch.py --onnx-dir onnx --output pretrained_weights.pt

# Single-epoch smoke test
CONVNEXT_BLOCKS=6 PRETRAINED_WEIGHTS=pretrained_weights.pt \
BATCH_SIZE=2 GRAD_ACCUM_STEPS=10 MAX_GRAD_NORM=1.0 \
WARMUP_STEPS=1500 EPOCHS=1 \
./venv/bin/python tools/finetune_convnext.py

# Full orchestrator v2 with loss gate
PHASES_JSON=phases_loss_gate.json CONVNEXT_BLOCKS=6 \
PRETRAINED_WEIGHTS=pretrained_weights.pt \
./venv/bin/python tools/phase_orchestrator_v2.py

# Direct fine-tuning with custom parameters
CONVNEXT_BLOCKS=6 PRETRAINED_WEIGHTS=pretrained_weights.pt \
BATCH_SIZE=2 GRAD_ACCUM_STEPS=12 MAX_GRAD_NORM=0.5 \
WARMUP_STEPS=1500 EPOCHS=10 \
LEARNING_RATE=3e-5 LR_BASE=5e-6 LR_NEW=2e-5 \
./venv/bin/python tools/finetune_convnext.py
```

## License

This project's sample code is released under the MIT License. - see the [LICENSE](https://github.com/supertone-inc/supertonic?tab=MIT-1-ov-file) for details.

The accompanying model is released under the OpenRAIL-M License. - see the [LICENSE](https://huggingface.co/Supertone/supertonic-2/blob/main/LICENSE) file for details.

This model was trained using PyTorch, which is licensed under the BSD 3-Clause License but is not redistributed with this project. - see the [LICENSE](https://docs.pytorch.org/FBGEMM/general/License.html) for details.

Copyright (c) 2026 Supertone Inc.






The training script is now running successfully in WSL with CUDA support! Here's what was done:

Summary of Changes
1. Fixed Model Architecture Mismatch
The pretrained weights were trained with embed_dim=256 and hidden_dim=256, but the training script was initializing the model with 192. Updated training/train_high_vram.py:

model = Supertonic(
    vocab_size=2000, 
    embed_dim=256,      # Changed from 192
    hidden_dim=256,     # Changed from 192
    style_dim=128
).to(device)
2. Fixed Missing device_type Variable
The autocast context manager was referencing an undefined device_type variable. Added the missing line in training/train_high_vram.py:

device_type = device.type  # Added this line
use_amp = (device_type == 'cuda')
Training is Now Running
The training is actively running on your RTX 3050 4GB GPU in WSL:

Device: NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB)
CUDA: 12.6 (PyTorch 2.10.0+cu126)
Current Epoch: ~36-37
Loss: ~1.02-1.05 (decreasing from initial ~1.5+)
How to Run Training
Use this command to run training in WSL with the venv:

wsl -d Ubuntu -e bash -c "cd /mnt/d/study/ai/supertonic && source venv_wsl/bin/activate && python -m training.train_high_vram"
Or enter WSL interactively:

wsl -d Ubuntu
cd /mnt/d/study/ai/supertonic
source venv_wsl/bin/activate
python -m training.train_high_vram
The training will save checkpoints to checkpoint_high_vram.pt after each epoch.