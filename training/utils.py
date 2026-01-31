import json
import re
import numpy as np
from unicodedata import normalize
import torch

class UnicodeProcessor:
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, "r") as f:
            self.indexer = json.load(f)

    def _preprocess_text(self, text: str, lang: str) -> str:
        # Simplified for training - assuming cleaner data or applying same regex
        text = normalize("NFKD", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Add lang tags logic if needed, or assume text is just text
        # The inference helper wraps in <lang>...</lang>
        # We should probably do the same if we want to learn language embeddings
        # But for now, let's keep it raw or add tags
        text = f"<{lang}>" + text + f"</{lang}>"
        return text

    def __call__(self, text: str, lang: str) -> np.ndarray:
        text = self._preprocess_text(text, lang)
        # Convert to IDs
        # max indexer len check handled by python list lookup
        ids = []
        for char in text:
            val = ord(char)
            if val < len(self.indexer) and self.indexer[val] != -1:
                ids.append(self.indexer[val])
            else:
                # Fallback for unknown?
                pass
        return np.array(ids, dtype=np.int64)

def get_best_device():
    if not torch.cuda.is_available():
        print("Creating CPU device (CUDA not available)")
        return torch.device('cpu')

    count = torch.cuda.device_count()
    print(f"\nScanning {count} CUDA devices:")

    best_index = 0
    max_score = -1

    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024**3
        name = props.name
        print(f"  Device [{i}]: {name} | VRAM: {vram_gb:.2f} GB")

        # Scoring system:
        # +1000 for NVIDIA
        # +VRAM count
        score = vram_gb
        if "NVIDIA" in name.upper():
            score += 1000

        if score > max_score:
            max_score = score
            best_index = i

    selected_device_name = torch.cuda.get_device_name(best_index)
    selected_vram = torch.cuda.get_device_properties(best_index).total_memory / 1024**3

    print(f"\n>>> SELECTED GPU: [{best_index}] {selected_device_name} ({selected_vram:.2f} GB)")
    print("If this is incorrect, set CUDA_VISIBLE_DEVICES environment variable.\n")

    return torch.device(f'cuda:{best_index}')


def maximum_path(value, mask, max_neg_val: float = -1e9):
    """
    Monotonic Alignment Search (MAS) - vectorized implementation.
    value: [b, t_text, t_mel]
    mask: [b, t_text, t_mel]

    Uses vectorized operations where possible for efficient path finding,
    with proper handling of tensor shapes.
    """
    b, t_t, t_s = value.shape
    device = value.device

    # Initialize DP table
    v = torch.full_like(value, max_neg_val)

    # Set initial position
    v[:, 0, 0] = value[:, 0, 0]

    # First row (i=0): can only come from left
    if t_s > 1:
        # row0_vals has shape [b, 1, t_s], squeeze to [b, t_s] for cumsum
        row0_vals = value[:, 0, :]
        v[:, 0, :] = torch.cumsum(row0_vals, dim=1)

    # Process remaining rows
    for i in range(1, t_t):
        # For position (i, j), we can come from (i-1, j-1) or (i, j-1)
        # Only valid for j >= i (monotonic constraint)

        # Diagonal contribution: from (i-1, j-1)
        # For j = i, we can only come from (i-1, j-1) = (i-1, i-1)
        # For j > i, we can come from (i-1, j-1)
        diag_vals = torch.full((b, t_s), max_neg_val, device=device)
        if i <= t_s - 1:
            # v[:, i-1, i-1:t_s-1] gives values from (i-1, i-1) to (i-1, t_s-2)
            # These correspond to positions j = i to j = t_s-1
            diag_vals[:, i:] = v[:, i-1, i-1:t_s-1] if t_s > i else v[:, i-1, :]

        # Horizontal contribution: from (i, j-1)
        horiz_vals = torch.full((b, t_s), max_neg_val, device=device)
        if i < t_s:
            # Position j=i can come from (i, i-1) if i > 0
            if i > 0:
                horiz_vals[:, i] = v[:, i, i-1]
            # Positions j > i can come from (i, j-1)
            if i + 1 < t_s:
                horiz_vals[:, i+1:] = v[:, i, i:t_s-1]

        # Take maximum and add current value
        max_prev = torch.maximum(diag_vals, horiz_vals)
        v[:, i, :] = max_prev + value[:, i, :]

    # Backtracking to find path
    path = torch.zeros_like(value)

    # Start from the end and work backwards
    curr_i = torch.full((b,), t_t - 1, dtype=torch.long, device=device)

    for curr_j in range(t_s - 1, -1, -1):
        # Mark path for current positions
        batch_idx = torch.arange(b, device=device)
        path[batch_idx, curr_i, curr_j] = 1.0

        # Determine next i position
        if curr_j > 0 and (curr_i > 0).any():
            # Get values for decision: stay (horizontal) vs move diagonal
            horiz_vals = v[batch_idx, curr_i, curr_j - 1]

            # For positions where curr_i > 0, check diagonal
            valid_diag = curr_i > 0
            diag_cond = torch.zeros(b, dtype=torch.bool, device=device)
            if valid_diag.any():
                diag_vals = v[batch_idx, curr_i - 1, curr_j - 1]
                diag_cond = valid_diag & (horiz_vals < diag_vals)

            # Move diagonal where condition is true
            curr_i = torch.where(diag_cond, curr_i - 1, curr_i)

    return path * mask
