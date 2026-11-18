import torch
import torch.nn as nn
import torch.nn.functional as F

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

# Mapping GTZAN genres to real-world listening vibes
# 0: Drive / Workout (энергичный бит, гитары, бас)
# 1: Chill / Relax (спокойный вокал, акустика, расслабление)
# 2: Focus / Study (инструментал, джаз, классика для концентрации)
VIBES = [
    "Драйв / Энергия",      # 0
    "Чилл / Отдых",         # 1
    "Фокус / Концентрация", # 2
]

GENRE_TO_VIBE = {
    "disco": 0,
    "hiphop": 0,
    "metal": 0,
    "rock": 0,
    "blues": 1,
    "country": 1,
    "pop": 1,
    "reggae": 1,
    "classical": 2,
    "jazz": 2,
}


class ConvBlock(nn.Module):
    """Residual Convolutional Block with BatchNorm and LeakyReLU"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out


class MusicAudioNet(nn.Module):
    """
    Lightweight Audio CNN for Mel-Spectrogram classification.
    Takes Mel-Spectrogram (B, 1, 128, T) -> Outputs Genre Logits (10) and Vibe Logits (3).
    Total size in ONNX: ~4.5 MB (or ~1.2 MB in INT8).
    CPU Inference time: ~3-5 ms.
    """
    def __init__(self, num_genres: int = len(GENRES), num_vibes: int = len(VIBES)):
        super().__init__()
        
        # Stem: fast downsampling of high-res mel spectrogram
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=(2, 2), padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 4 Residual Stages
        self.stage1 = ConvBlock(32, 64, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.stage2 = ConvBlock(64, 96, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.stage3 = ConvBlock(96, 128, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.stage4 = ConvBlock(128, 192, stride=1)
        
        # Global Aggregation (Time + Frequency Pooling)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        
        # Dual Classification Heads
        # Head 1: 10 Genres
        self.genre_head = nn.Sequential(
            nn.Linear(192, 96),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(96, num_genres)
        )
        
        # Head 2: 3 Listening Vibes
        self.vibe_head = nn.Sequential(
            nn.Linear(192, 48),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(48, num_vibes)
        )

    def forward(self, mel: torch.Tensor):
        """
        Input: mel-spectrogram tensor of shape (batch, 1, n_mels=128, time_steps)
        Returns: (genre_logits [batch, 10], vibe_logits [batch, 3])
        """
        x = self.stem(mel)
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.stage4(x)
        
        feat = self.global_pool(x).flatten(1)
        feat = self.dropout(feat)
        
        genre_logits = self.genre_head(feat)
        vibe_logits = self.vibe_head(feat)
        
        return genre_logits, vibe_logits


if __name__ == "__main__":
    # Smoke test for shapes and parameters count
    model = MusicAudioNet()
    dummy_mel = torch.randn(2, 1, 128, 313) # 10s audio @ 16kHz
    g_out, v_out = model(dummy_mel)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MusicAudioNet Architecture Validated!")
    print(f"Total Parameters: {total_params:,} (~{total_params * 4 / (1024*1024):.2f} MB FP32)")
    print(f"Output shapes: Genre: {g_out.shape}, Vibe: {v_out.shape}")
