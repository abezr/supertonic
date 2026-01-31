#!/usr/bin/env python3
import os
import re
import sys
import json
import subprocess
from pathlib import Path
from statistics import median

# Persisted state path
STATE_PATH = Path("tools/orchestrator_state.json")

# Parse "Loss <value>" from trainer output lines
LOSS_RE = re.compile(r"Loss\s+([0-9]+.?[0-9]*)")
RESUME_EPOCH_RE = re.compile(r"Resuming from epoch\s+(\d+)")
# Default staged plan (tuned for 4 GB VRAM)
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
            # Run more than one epoch per call to accelerate progress tracking
            "EPOCHS_PER_CALL": "5",
        },
        # Advance when cumulative optimizer steps >= 6000 (approx) OR median loss is below threshold
        "criteria": {"min_steps": 6000, "loss_median_below": None}
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
            "EPOCHS_PER_CALL": "3",
        },
        "criteria": {"min_steps": 12000, "loss_median_below": None}
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
            "EPOCHS_PER_CALL": "2",
        },
        "criteria": {"min_steps": 20000, "loss_median_below": None}
    },
]


def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"phase_index": 0, "total_batches": 0, "total_steps": 0, "history": []}


def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_training_chunk(env: dict): 
    """ Run tools.finetune_convnext for EPOCHS_PER_CALL epochs. Returns: (batch_count, steps_inc, losses, resumed_epoch, epochs_used) """ 
    merged_env = os.environ.copy() 
    merged_env.update(env)

    merged_env["PYTHONPATH"] = merged_env.get("PYTHONPATH", ".")

    epochs_per_call = int(str(env.get("EPOCHS_PER_CALL", "1")))
    merged_env["EPOCHS"] = str(epochs_per_call)

    cmd = [sys.executable, "-m", "tools.finetune_convnext"]

    print("\n=== Launching training chunk ===")
    for k in [
        "EPOCHS", "LEARNING_RATE", "LR_BASE", "LR_NEW", "BATCH_SIZE", "GRAD_ACCUM_STEPS",
        "MAX_GRAD_NORM", "WARMUP_STEPS", "FREEZE_LOWER_BLOCKS", "UNFREEZE_AFTER_STEPS",
        "ENCODER_TYPE", "CONVNEXT_BLOCKS"
    ]:
        if k in merged_env:
            print(f"  {k}={merged_env[k]}")
    print("===============================\n")

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=merged_env, text=True, bufsize=1)

    batch_count = 0
    losses = []
    resumed_epoch = None
    try:
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            m = LOSS_RE.search(line)
            if m:
                try:
                    losses.append(float(m.group(1)))
                    batch_count += 1
                except Exception:
                    pass
            r = RESUME_EPOCH_RE.search(line)
            if r:
                try:
                    resumed_epoch = int(r.group(1))
                except Exception:
                    resumed_epoch = None
    finally:
        p.wait()

    try:
        accum = int(merged_env.get("GRAD_ACCUM_STEPS", "12"))
    except Exception:
        accum = 12
    steps_inc = batch_count // max(1, accum)
    return batch_count, steps_inc, losses, resumed_epoch, int(merged_env.get("EPOCHS", "1"))

def orchestrate(phases: list):
    state = load_state()
    phase_idx = int(state.get("phase_index", 0))
    total_batches = int(state.get("total_batches", 0))
    total_steps = int(state.get("total_steps", 0))

    while phase_idx < len(phases):
        phase = phases[phase_idx]
        name = phase.get("name", f"phase_{phase_idx}")
        env = phase.get("env", {})
        crit = phase.get("criteria", {})
        min_steps = int(crit.get("min_steps", 0))
        loss_median_below = crit.get("loss_median_below")
        loss_median_below = float(loss_median_below) if loss_median_below is not None else None

        print(f"\n########## {name} ##########")
        print(f"Target min_steps: {min_steps} (cumulative)")
        if loss_median_below is not None:
            print(f"Target loss_median_below: {loss_median_below}")
        print(f"Progress: total_steps={total_steps}, total_batches={total_batches}")

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

        print("\nChunk summary:")
        print(f"  batches={batches}, steps_inc={steps_inc}")
        print(f"  loss_mean={loss_mean}, loss_median={loss_median_val}, loss_min={loss_min}, loss_max={loss_max}")

        # Update state
        entry = {
            "phase": name,
            "batches": batches,
            "steps_inc": steps_inc,
            "loss_mean": loss_mean,
            "loss_median": loss_median_val,
            "loss_min": loss_min,
            "loss_max": loss_max,
        }
        state.setdefault("history", []).append(entry)
        state.update({
            "phase_index": phase_idx,
            "total_batches": total_batches,
            "total_steps": total_steps,
        })
        save_state(state)

        # Decide whether to advance
        cond_steps = (total_steps >= min_steps)
        cond_loss = True
        if loss_median_below is not None:
            cond_loss = (loss_median_val is not None and loss_median_val <= loss_median_below)

        if cond_steps and cond_loss:
            print(f"Advancing from {name} -> next phase")
            phase_idx += 1
            state["phase_index"] = phase_idx
            save_state(state)
        else:
            detail = f" (loss_median={loss_median_val})" if loss_median_below is not None else ""
            print(f"Staying in {name}. Steps {total_steps}/{min_steps}{detail}")

    print("\nAll phases completed.")


if __name__ == "__main__":
    phases = DEFAULT_PHASES
    # Allow overriding with PHASES_JSON
    path = os.environ.get("PHASES_JSON")
    if path and Path(path).exists():
        try:
            phases = json.loads(Path(path).read_text(encoding="utf-8"))
            print(f"Loaded phases from {path}")
        except Exception as e:
            print(f"Failed to load PHASES_JSON {path}: {e}. Falling back to defaults.")
    orchestrate(phases)
