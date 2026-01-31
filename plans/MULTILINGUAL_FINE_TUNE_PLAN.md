# Multilingual Fine-Tune Plan (EN + RU + UK) with Single Voice

Status: Approved direction
Scope: Preserve pretrained Supertonic English voice while adding RU/UK via fine-tuning.
Prereqs: Access to Supertonic English ONNX, current training code, RU/UK datasets, English data for stability.

---

## Objective
- Use the existing Supertonic English model (ONNX, ConvNeXt-based) as the base.
- Extend it to support Russian and Ukrainian while preserving a single target voice.
- Use a phoneme front-end with language-ID tokens and add backward-compatible speaker conditioning.

---

## High-Level Strategy
1) Mirror ONNX ConvNeXt in PyTorch
   - Reimplement the text encoder to match the ONNX graph exactly (ConvNeXt blocks).
   - Ensure embed/channel dim = 256 consistently.
   - Keep naming and tensor shapes compatible for weight loading.

2) Load Pretrained English Weights
   - Convert ONNX weights to PyTorch state_dict or implement a custom key mapping.
   - Validate load: no unexpected/missing critical keys for core blocks.
   - Run a short English-only sanity fine-tune.

3) Phoneme Front-end + Language Conditioning
   - Use a multilingual phonemizer (e.g., eSpeak-ng/phonemizer) generating a single phoneme set (IPA-ish) across EN/RU/UK.
   - Prepend language tokens: <lang=en>, <lang=ru>, <lang=uk>.
   - Persist vocabulary (phoneme set + language tokens) for reproducibility.
   - Consider stress marks for RU/UK; optionally add a small lexicon for high-frequency words.

4) Backward-Compatible Speaker Conditioning
   - Add a learned speaker embedding table; inject via FiLM-like scale/bias or additive bias at selected blocks.
   - Initialize speaker embeddings to zeros so the pretrained English path is unchanged.
   - Assign ID 0 to the original English speaker.
   - Add IDs for RU/UK data speakers; learn during multilingual fine-tune.
   - At inference, select the English speaker ID for single-voice output across languages.

5) Staged Fine-Tuning Schedule
   - Stage 0 (Sanity):
     - English-only, freeze lower 50–75% of blocks.
     - LR: 1e-5 to 3e-5, warmup ~1k steps, EMA 0.999, WD 1e-4.
     - Few thousand steps; monitor intelligibility and loss.
   - Stage 1 (Introduce RU/UK gradually):
     - Sampling mix A: 80% EN / 10% RU / 10% UK.
     - Sampling mix B: 50% EN / 25% RU / 25% UK.
     - Train phoneme embeddings, language embeddings, speaker embeddings, and upper layers first; keep most ConvNeXt blocks frozen initially.
   - Stage 2 (Progressive unfreeze):
     - Unfreeze one stage at a time if RU/UK plateaus.
     - Keep LR low; re-warmup on each unfreeze.
   - Validation:
     - Fixed EN/RU/UK dev sets; listen to audio and track phoneme-level metrics.
     - Evaluate with the English speaker ID for all languages to monitor single-voice quality.

6) Data & Mixed Speakers
   - Different speakers in RU/UK are okay due to speaker embedding disentanglement.
   - If possible, add 15–30 minutes of RU/UK for the target English speaker near the end for stronger voice consistency.

7) Guardrails
   - Enforce embed_dim=256 everywhere (text embedder, ConvNeXt channels, projections).
   - Do not mix raw characters; use phonemes + language tokens only.
   - Keep some EN data always in the mix to avoid catastrophic forgetting.
   - Enable AMP, gradient clipping, TF32 (where applicable), and gradient accumulation for 4 GB VRAM training.

---

## Implementation Tasks (Repo)

1) Model Rewrite
- Replace Transformer-based text encoder with ConvNeXt-based encoder matching ONNX (embedding=256, ConvNeXt blocks with dwconv -> norm -> pwconv1 -> GELU -> pwconv2, residuals).
- Preserve output heads and projection shapes.
- Add optional speaker embedding injection (zero-init; ID 0 = pretrained speaker).

2) Weight Loading
- Implement ONNX->PyTorch mapping; verify parameter counts and shapes (~169 tensors expected in ONNX text encoder scope; may differ by module split).
- Fail fast if core ConvNeXt block weights don’t match; allow non-critical additions (e.g., language/speaker embeddings) to be randomly initialized.

3) Tokenizer/Dataset
- Integrate phonemizer with EN/RU/UK; build unified phoneme vocab + <lang=*> tokens.
- Update dataset to prepend language tokens and to provide speaker IDs.
- Persist vocab JSON and ensure inference path uses the same mapping.

4) Training Enhancements (Low VRAM)
- Mixed precision (autocast + GradScaler), TF32 where supported.
- Gradient accumulation to reach effective batch size > 1.
- Gradient clipping (e.g., 1.0 norm).
- EMA weights.
- Layer freezing/unfreezing schedule.
- Curriculum sampler to control EN/RU/UK ratios per phase.
- Robust checkpoint/save & periodic validation audio export.

5) Validation & Monitoring
- Add per-language validation script that synthesizes fixed prompts and saves audio.
- Track loss breakdowns and items/s; watch for regressions when unfreezing.

---

## Risks & Mitigations
- Architecture mismatch → Strict structural parity with ONNX; add tests to compare shapes and counts.
- Forgetting English voice → Keep EN in sampling mix; freeze lower layers initially; small LR.
- Pronunciation for RU/UK → Use phonemes; add stress handling; include language tokens.
- Voice leakage from RU/UK speakers → Use speaker embedding and inference with English speaker ID.
- VRAM constraints → AMP + grad accumulation + careful batch sizing; optionally checkpoint activations.

---

## Milestones
- M1: ONNX ConvNeXt parity in PyTorch; successful weight load; English sanity fine-tune audio OK.
- M2: Phoneme front-end + language tokens integrated; training stable with EN-only.
- M3: RU/UK introduced with speaker conditioning; intelligible RU/UK in English voice at inference.
- M4: Progressive unfreeze delivers improved RU/UK quality without degrading EN.

---

## Done Definition per Milestone
- M1: No unexpected keys for core ConvNeXt blocks; English audio comparable to baseline.
- M2: Loss stable; phoneme pipeline deterministic; vocab persisted and reused in inference.
- M3: Understandable RU/UK samples generated with English speaker ID; CER improving across epochs.
- M4: Balanced quality across EN/RU/UK; single voice preserved; inference latency acceptable.

---

## Notes
- Until the model is rewritten to ConvNeXt, do NOT rely on loading ONNX weights: it causes instability and exploding loss (as observed). Use strict checks and abort training if core weights don’t align.
