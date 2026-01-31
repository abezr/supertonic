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

@torch.jit.script
def maximum_path(value, mask, max_neg_val: float = -1e9):
    """
    Monotonic Alignment Search (MAS)
    value: [b, t_text, t_mel]
    mask: [b, t_text, t_mel]
    """
    device = value.device
    dtype = value.dtype
    b, t_t, t_s = value.shape

    v = torch.zeros_like(value)
    
    for j in range(t_s):
        for i in range(t_t):
            if j < i:
                v[:, i, j] = max_neg_val
                continue
            
            if i == 0:
                if j == 0:
                    v[:, i, j] = value[:, i, j]
                else:
                    v[:, i, j] = v[:, i, j-1] + value[:, i, j]
            else:
                if j == i:
                    v[:, i, j] = v[:, i-1, j-1] + value[:, i, j]
                else:
                    v[:, i, j] = torch.max(v[:, i-1, j-1], v[:, i, j-1]) + value[:, i, j]

    path = torch.zeros_like(value)
    for b_idx in range(b):
        curr_i = t_t - 1
        for curr_j in range(t_s - 1, -1, -1):
            path[b_idx, curr_i, curr_j] = 1
            if curr_i > 0:
                # BUGFIX: Add boundary check to prevent negative indexing
                if curr_j == curr_i:
                    # Must move diagonally (can't stay on same i)
                    curr_i -= 1
                elif curr_j > 0 and v[b_idx, curr_i, curr_j-1] < v[b_idx, curr_i-1, curr_j-1]:
                    # Diagonal is better than horizontal
                    curr_i -= 1
                # else: stay on same i (horizontal move)
    return path * mask

