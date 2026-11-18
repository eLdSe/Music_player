import os
import tarfile
import urllib.request
from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

from model import GENRES, GENRE_TO_VIBE

GTZAN_URL = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 10.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC) # 160,000 samples
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512


def get_mel_transform():
    """MelSpectrogram + AmplitudeToDB transforms"""
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    amp_to_db = T.AmplitudeToDB()
    return mel_transform, amp_to_db


class SpecAugment(nn.Module):
    """Data augmentation for audio: Time Masking & Frequency Masking"""
    def __init__(self, freq_mask_param: int = 16, time_mask_param: int = 32):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        return self.time_mask(self.freq_mask(mel_spec))


def download_gtzan_if_needed(data_dir: Path) -> Path:
    """Download and extract GTZAN dataset if not already present"""
    genres_dir = data_dir / "genres"
    if genres_dir.exists() and len(list(genres_dir.glob("*/*.wav"))) >= 900:
        print(f"[Dataset] Found existing GTZAN at: {genres_dir}")
        return genres_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "genres.tar.gz"

    if not tar_path.exists():
        print(f"[Dataset] Downloading GTZAN (approx 1.2 GB)...")
        urllib.request.urlretrieve(GTZAN_URL, str(tar_path))
        print(f"[Dataset] Download complete!")

    print(f"[Dataset] Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print(f"[Dataset] Extracted to {genres_dir}")
    return genres_dir


class MusicDataset(Dataset):
    """
    GTZAN Music Dataset with multi-slice chunking.
    From a single 30-sec track we extract 3 overlapping 10-sec slices:
    - Slice 0: [ 0s .. 10s ]
    - Slice 1: [ 10s .. 20s ] (usually chorus/hook)
    - Slice 2: [ 20s .. 30s ]
    This expands 1,000 tracks to 3,000 diverse training samples!
    """
    def __init__(self, file_slices: list[tuple[Path, int, int, int]], is_train: bool = True):
        self.file_slices = file_slices
        self.is_train = is_train
        self.mel_transform, self.amp_to_db = get_mel_transform()
        self.augment = SpecAugment() if is_train else nn.Identity()

    def __len__(self):
        return len(self.file_slices)

    def __getitem__(self, idx: int):
        file_path, slice_idx, genre_idx, vibe_idx = self.file_slices[idx]
        
        try:
            waveform, sr = torchaudio.load(str(file_path))
        except Exception as e:
            # Fallback for occasional corrupt file header
            waveform = torch.zeros(1, CHUNK_SAMPLES)
            sr = SAMPLE_RATE

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16,000 Hz if different
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Extract specific 10-second slice
        start_sample = slice_idx * CHUNK_SAMPLES
        end_sample = start_sample + CHUNK_SAMPLES

        if waveform.shape[1] < end_sample:
            # Pad if shorter than chunk
            pad_len = end_sample - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        chunk = waveform[:, start_sample:end_sample]

        # Compute Log-Mel Spectrogram: (1, 128, 313)
        with torch.no_grad():
            mel = self.amp_to_db(self.mel_transform(chunk))
            # Normalize to zero mean, unit variance
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)
            mel = self.augment(mel)

        return mel, torch.tensor(genre_idx, dtype=torch.long), torch.tensor(vibe_idx, dtype=torch.long)


def prepare_dataset_splits(data_dir: Path, split_ratio: float = 0.8, seed: int = 42):
    """
    Split tracks 80% train / 20% val by file to prevent data leakage between slices.
    """
    random.seed(seed)
    train_slices = []
    val_slices = []

    for genre_idx, genre_name in enumerate(GENRES):
        genre_folder = data_dir / genre_name
        if not genre_folder.exists():
            continue
        
        files = sorted(list(genre_folder.glob("*.wav")))
        random.shuffle(files)
        
        split_point = int(len(files) * split_ratio)
        train_files = files[:split_point]
        val_files = files[split_point:]

        vibe_idx = GENRE_TO_VIBE[genre_name]

        # In train: 3 slices per track (0-10s, 10-20s, 20-30s) -> 3x data expansion!
        for f in train_files:
            for s_idx in (0, 1, 2):
                train_slices.append((f, s_idx, genre_idx, vibe_idx))

        # In val: take the middle 10-20s slice (most representative part)
        for f in val_files:
            val_slices.append((f, 1, genre_idx, vibe_idx))

    random.shuffle(train_slices)
    random.shuffle(val_slices)

    print(f"[Dataset] Prepared {len(train_slices)} training samples and {len(val_slices)} validation samples.")
    return train_slices, val_slices
