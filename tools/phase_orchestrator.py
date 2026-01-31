#!/usr/bin/env python3
import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path

STATE_PATH = Path("tools/orchestrator_state.json")

BATCH_LINE_RE = re.compile(r"Loss\s+([0-9]+\.?[0-9]*)")

# Default phases plan
DEFAULT_PHASES = [
    {
        "name": "PhaseA_stabilize",
        "env": {
            "ENCODER_TYPE": "convnext",
            "CONVNEXT_BLOCKS": os.environ.get("CONVNEXT_BLOCKS", "6"),
            "PRETRAINED_WEIGHTS": os.environ.get("PRETRAINED_WEIGHTS", "pretrained_weights.pt"),
            "LEARNING_RATE": "2e-5",
            "LR_BASE": "2e-6",
            "LR_NEW": "1e-5",
            "BATCH_SIZE": os.environ.get("BATCH_SIZE", "2"),
            "GRAD_ACCUM_STEPS": "16",
            "MAX_GRAD_NORM": "0.5",
            "WARMUP_STEPS": "2000",
            "FREEZE_LOWER_BLOCKS": "4",
            "UNFREEZE_AFTER_STEPS": "6000",
        },
        "criteria": {"min_steps": 6000}
    },
    {
        "name": "PhaseB_progressive_unfreeze",
        "env": {
            "ENCODER_TYPE": "convnext",
            "CONVNEXT_BLOCKS": os.environ.get("CONVNEXT_BLOCKS", "6"),
            "PRETRAINED_WEIGHTS": os.environ.get("PRETRAINED_WEIGHTS", "pretrained_weights.pt"),
            "LEARNING_RATE": "3e-5",
            "LR_BASE": "5e-6",
            "LR_NEW": "2e-5",
            "BATCH_SIZE": os.environ.get("BATCH_SIZE", "2"),
            "GRAD_ACCUM_STEPS": "12",
            "MAX_GRAD_NORM": "1.0",
            "WARMUP_STEPS": "1500",
            "FREEZE_LOWER_BLOCKS": "2",
            "UNFREEZE_AFTER_STEPS": "12000",
        },
        "criteria": {"min_steps": 12000}
    },
    {
        "name": "PhaseC_full_finetune",
        "env": {
            "ENCODER_TYPE": "convnext",
            "CONVNEXT_BLOCKS": os.environ.get("CONVNEXT_BLOCKS", "6"),
            "PRETRAINED_WEIGHTS": os.environ.get("PRETRAINED_WEIGHTS", "pretrained_weights.pt"),
            "LEARNING_RATE": "3e-5",
            "LR_BASE": "5e-6",
            "LR_NEW": "3e-5",
            "BATCH_SIZE": os.environ.get("BATCH_SIZE", "2"),
            "GRAD_ACCUM_STEPS": "12",
            "MAX_GRAD_NORM": "1.0",
            "WARMUP_STEPS": "1000",
            "FREEZE_LOWER_BLOCKS": "0",
            "UNFREEZE_AFTER_STEPS": "0",
        },
        "criteria": {"min_steps": 20000}
    }
]


def load_state():
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"phase_index": 0, "total_batches": 0, "total_steps": 0, "history": []}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def run_one_epoch(env: dict):
    """Run one training epoch by invoking: ./venv/bin/python -m tools.finetune_convnext with EPOCHS=1.
    Returns: batches_count, steps_increment (approx via GRAD_ACCUM_STEPS), loss_list
    """
    merged_env = os.environ.copy()
    merged_env.update(env)
    merged_env["EPOCHS"] = "1"

    # Ensure PYTHONPATH so training.* is importable
    merged_env["PYTHONPATH"] = merged_env.get("PYTHONPATH", ".")

    cmd = [sys.executable, "-m", "tools.finetune_convnext"]
    print("\n=== Launching epoch with env overrides ===")
    for k in [
        "LEARNING_RATE", "LR_BASE", "LR_NEW", "BATCH_SIZE", "GRAD_ACCUM_STEPS",
        "MAX_GRAD_NORM", "WARMUP_STEPS", "FREEZE_LOWER_BLOCKS", "UNFREEZE_AFTER_STEPS",
        "ENCODER_TYPE", "CONVNEXT_BLOCKS"
    ]:
        if k in merged_env:
            print(f"  {k}={merged_env[k]}")
    print("========================================\n")

    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=merged_env, text=True, bufsize=1
    )

    loss_list = []
    batch_count = 0
    try:
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            m = BATCH_LINE_RE.search(line)
            if m:
                try:
                    loss = float(m.group(1))
                    loss_list.append(loss)
                    batch_count += 1
                except Exception:
                    pass
    finally:
        p.wait()

    # Steps approx = floor(batches / accum)
    try:
        accum = int(merged_env.get("GRAD_ACCUM_STEPS", "12"))
    except Exception:
        accum = 12
    steps_inc = batch_count // max(1, accum)
    return batch_count, steps_inc, loss_list


def orchestrate(phases):
    state = load_state()
    total_batches = state.get("total_batches", 0)
    total_steps = state.get("total_steps", 0)
    phase_idx = state.get("phase_index", 0)

    while phase_idx < len(phases):
        phase = phases[phase_idx]
        name = phase.get("name", f"phase_{phase_idx}")
        env = phase.get("env", {})
        criteria = phase.get("criteria", {})
        min_steps = int(criteria.get("min_steps", 0))

        print(f"\n########## {name} ##########")
        print(f"Target min_steps: {min_steps} (cumulative)")
        print(f"Progress: total_steps={total_steps}, total_batches={total_batches}")

        # Run a single epoch and parse logs
        # batches, steps_inc, losses = run_one_epoch(env)
        # total_batches += batches
        # total_steps += steps_inc
        # Run chunk with auto-retry if no batches processed due to resume epoch >= EPOCHS
        max_retries = 2
        retry = 0
        while True:
            batches, steps_inc, losses, resumed_epoch, epochs_used = run_training_chunk(env)
            if batches > 0 or retry >= max_retries:
                break
            print("No batches processed; auto-adjusting EPOCHS_PER_CALL and retrying...")
            try:
          epc = int(env.get("EPOCHS_PER_CALL", "1"))
      except Exception:
          epc = 1
      bump = max(epc + 2, (resumed_epoch + 2) if resumed_epoch is not None else 4)
      env["EPOCHS_PER_CALL"] = str(bump)
      print(f"  resumed_epoch={resumed_epoch}, previous EPOCHS={epochs_used} -> new EPOCHS_PER_CALL={env['EPOCHS_PER_CALL']}")
      retry += 1
    total_batches += batches
    total_steps += steps_inc

    # Summarize
    loss_mean = (sum(losses) / len(losses)) if losses else None
    loss_median_val = median(losses) if losses else None
    loss_min = min(losses) if losses else None
    loss_max = max(losses) if losses else None

        # Keep short history
        state_entry = {
            "phase": name,
            "batches": batches,
            "steps_inc": steps_inc,
            "loss_mean": float(sum(losses)/len(losses)) if losses else None,
            "loss_median": float(sorted(losses)[len(losses)//2]) if losses else None,
            "loss_min": float(min(losses)) if losses else None,
            "loss_max": float(max(losses)) if losses else None,
        }
        state.setdefault("history", []).append(state_entry)
        state.update({
            "phase_index": phase_idx,
            "total_batches": total_batches,
            "total_steps": total_steps,
        })
        save_state(state)

        # Decide progression
        advance = False
        if total_steps >= min_steps:
            advance = True
        if advance:
            print(f"Advancing from {name} -> next phase (steps {total_steps} >= {min_steps})")
            phase_idx += 1
            state["phase_index"] = phase_idx
            save_state(state)
        else:
            print(f"Staying in {name}. Steps {total_steps} / {min_steps}")

    print("\nAll phases completed.")


if __name__ == "__main__":
    # Allow overriding phase plan via PHASES_JSON pointing to a JSON file
    phases = DEFAULT_PHASES
    phases_json = os.environ.get("PHASES_JSON")
    if phases_json and Path(phases_json).exists():
        try:
            with open(phases_json, "r", encoding="utf-8") as f:
                phases = json.load(f)
        except Exception as e:
            print(f"Failed to load PHASES_JSON: {e}. Falling back to defaults.")

    # Warn if COMPILE_MODEL is enabled (can cause hangs)
    if os.environ.get("COMPILE_MODEL", "0") == "1":
        print("\n" + "="*60)
        print("⚠️  WARNING: COMPILE_MODEL=1 detected!")
        print("   torch.compile may cause training to hang or freeze.")
        print("   If you experience hangs, disable with: export COMPILE_MODEL=0")
        print("="*60 + "\n")

    orchestrate(phases)
