import torch
import torchaudio
from torch.utils.data import Dataset
from .utils import UnicodeProcessor

class TTSDataset(Dataset):
    def __init__(self, filelist_path, indexer_path, sample_rate=24000, lang='uk'):
        self.items = []
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    self.items.append((parts[0], parts[1])) # path, text
        
        self.processor = UnicodeProcessor(indexer_path)
        self.sample_rate = sample_rate
        self.lang = lang
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80, # Typical for HiFiGAN/TTS
            f_min=0,
            f_max=8000
        )

    def __len__(self):
        return len(self.items)

    def _fix_path(self, path):
        # Handle Windows paths when running in WSL (Linux)
        import os
        import re
        
        # Cleanup potential double backslashes or mixed slashes first for easier matching
        # But be careful not to break the drive letter check
        
        if os.name == 'posix':
            # Check for Windows drive letter pattern (e.g. D:\ or D:/)
            # Regex captures: (DriveLetter) + (Rest of path)
            match = re.match(r'^([a-zA-Z]):[\\/](.*)$', path)
            if match:
                drive_letter = match.group(1).lower()
                rest = match.group(2).replace('\\', '/')
                new_path = f"/mnt/{drive_letter}/{rest}"
                # print(f"Fixed path: {path} -> {new_path}") # Debug print if needed
                return new_path
        return path

    def __getitem__(self, idx):
        import os
        import sys
        audio_path, text = self.items[idx]
        
        if idx == 0:
            print(f"DEBUG dataset: Getting item 0, original path={audio_path}")
            sys.stdout.flush()
        
        audio_path = self._fix_path(audio_path)
        
        if idx == 0:
            print(f"DEBUG dataset: Fixed path={audio_path}")
            sys.stdout.flush()
        
        # Mel Cache Logic - Avoid redundant CPU extraction
        mel_cache_path = audio_path.replace(".wav", ".mel.pt")
        
        if os.path.exists(mel_cache_path):
            if idx == 0:
                print(f"DEBUG dataset: Loading cached mel from {mel_cache_path}")
                sys.stdout.flush()
            mel = torch.load(mel_cache_path, map_location='cpu', weights_only=True)
            if idx == 0:
                print(f"DEBUG dataset: Cached mel loaded, shape={mel.shape}")
                sys.stdout.flush()
        else:
            if idx == 0:
                print(f"DEBUG dataset: No cache found, loading audio from {audio_path}")
                sys.stdout.flush()
            # Load Audio
            # Windows compatibility fix: explicit backend or fallback
            try:
                if idx == 0:
                    print(f"DEBUG dataset: Calling torchaudio.load...")
                    sys.stdout.flush()
                waveform, sr = torchaudio.load(audio_path, backend="soundfile")
                if idx == 0:
                    print(f"DEBUG dataset: torchaudio.load succeeded, waveform shape={waveform.shape}, sr={sr}")
                    sys.stdout.flush()
            except Exception as e:
                # Fallback to soundfile directly
                import soundfile as sf
                try:
                    wav_data, sr = sf.read(audio_path)
                except Exception as e2:
                    print(f"Failed to load {audio_path}: {e2}")
                    raise e2
                    
                waveform = torch.from_numpy(wav_data).float()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.t() # [channels, time]
            
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            # Mel Spec
            mel = self.mel_transform(waveform).squeeze(0) # [n_mels, T]
            
            # Dynamic Range Compression (Log-Mel)
            mel = torch.log(torch.clamp(mel, min=1e-5))
            
            # Normalization (z-score approximation)
            mel = (mel + 4.0) / 2.0
            
            # Save to cache for next time
            torch.save(mel, mel_cache_path)
        
        # Text
        text_ids = self.processor(text, self.lang)
        text_ids = torch.from_numpy(text_ids).long()
        
        # Lang ID
        lang_id = 0 if self.lang == 'uk' else 1
        lang_id = torch.tensor(lang_id).long()
        
        return text_ids, mel, audio_path, lang_id

def collate_fn(batch):
    # Sort by text length for packing (optional but good practice)
    batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
    
    text_ids = [x[0] for x in batch]
    mels = [x[1] for x in batch]
    paths = [x[2] for x in batch]
    lang_ids = torch.stack([x[3] for x in batch])
    
    text_lens = torch.LongTensor([t.size(0) for t in text_ids])
    mel_lens = torch.LongTensor([m.size(1) for m in mels])
    
    text_padded = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=0)
    
    max_mel_len = mel_lens.max()
    mel_padded = torch.zeros(len(mels), mels[0].size(0), max_mel_len)
    for i, mel in enumerate(mels):
        mel_padded[i, :, :mel.size(1)] = mel
        
    return text_padded, text_lens, mel_padded, mel_lens, paths, lang_ids
