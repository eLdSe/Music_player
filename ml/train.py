import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from model import MusicAudioNet, GENRES, VIBES, GENRE_TO_VIBE
from dataset import (
    download_gtzan_if_needed,
    prepare_dataset_splits,
    MusicDataset,
    SAMPLE_RATE,
    CHUNK_DURATION_SEC,
    CHUNK_SAMPLES,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
)

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, criterion_genre, criterion_vibe, scaler):
    model.train()
    total_loss = 0.0
    correct_genre = 0
    correct_vibe = 0
    total_samples = 0

    use_cuda = DEVICE == "cuda"

    for mel, genre_tgt, vibe_tgt in loader:
        mel = mel.to(DEVICE)
        genre_tgt = genre_tgt.to(DEVICE)
        vibe_tgt = vibe_tgt.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=use_cuda):
            genre_logits, vibe_logits = model(mel)
            loss_g = criterion_genre(genre_logits, genre_tgt)
            loss_v = criterion_vibe(vibe_logits, vibe_tgt)
            loss = loss_g + 0.5 * loss_v

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * mel.size(0)
        correct_genre += (genre_logits.argmax(dim=-1) == genre_tgt).sum().item()
        correct_vibe += (vibe_logits.argmax(dim=-1) == vibe_tgt).sum().item()
        total_samples += mel.size(0)

    avg_loss = total_loss / total_samples
    acc_genre = (correct_genre / total_samples) * 100.0
    acc_vibe = (correct_vibe / total_samples) * 100.0
    return avg_loss, acc_genre, acc_vibe


@torch.no_grad()
def evaluate(model, loader, criterion_genre, criterion_vibe):
    model.eval()
    total_loss = 0.0
    correct_genre = 0
    correct_vibe = 0
    total_samples = 0

    for mel, genre_tgt, vibe_tgt in loader:
        mel = mel.to(DEVICE)
        genre_tgt = genre_tgt.to(DEVICE)
        vibe_tgt = vibe_tgt.to(DEVICE)

        genre_logits, vibe_logits = model(mel)
        loss_g = criterion_genre(genre_logits, genre_tgt)
        loss_v = criterion_vibe(vibe_logits, vibe_tgt)
        loss = loss_g + 0.5 * loss_v

        total_loss += loss.item() * mel.size(0)
        correct_genre += (genre_logits.argmax(dim=-1) == genre_tgt).sum().item()
        correct_vibe += (vibe_logits.argmax(dim=-1) == vibe_tgt).sum().item()
        total_samples += mel.size(0)

    avg_loss = total_loss / total_samples
    acc_genre = (correct_genre / total_samples) * 100.0
    acc_vibe = (correct_vibe / total_samples) * 100.0
    return avg_loss, acc_genre, acc_vibe


def export_to_onnx(model, export_dir: Path):
    export_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = export_dir / "music_classifier.onnx"
    int8_onnx = export_dir / "music_classifier.int8.onnx"
    labels_path = export_dir / "labels.json"

    model.eval().cpu()
    dummy_input = torch.randn(1, 1, N_MELS, 313, dtype=torch.float32)

    print(f"\n[Export] Exporting model to ONNX: {fp32_onnx}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_onnx),
        input_names=["mel_spectrogram"],
        output_names=["genre_logits", "vibe_logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size", 3: "time_steps"},
            "genre_logits": {0: "batch_size"},
            "vibe_logits": {0: "batch_size"},
        },
        opset_version=17,
    )

    # Validate ONNX model
    onnx_proto = onnx.load(str(fp32_onnx))
    onnx.checker.check_model(onnx_proto)
    print(f"[Export] FP32 ONNX model valid ({fp32_onnx.stat().st_size / (1024*1024):.2f} MB)")

    # Dynamic INT8 Quantization
    print(f"[Export] Quantizing to INT8: {int8_onnx}...")
    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        weight_type=QuantType.QInt8,
    )
    print(f"[Export] INT8 ONNX model ready ({int8_onnx.stat().st_size / (1024*1024):.2f} MB)")

    # Write metadata JSON for Rust application
    metadata = {
        "model_name": "MusicAudioNet",
        "sample_rate": SAMPLE_RATE,
        "chunk_duration_sec": CHUNK_DURATION_SEC,
        "chunk_samples": CHUNK_SAMPLES,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "genres": GENRES,
        "vibes": VIBES,
        "genre_to_vibe": GENRE_TO_VIBE,
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[Export] Metadata saved to {labels_path}")


def main():
    print("=========================================================")
    print("      Music Audio Classifier (Dual-Head Genre & Vibe)   ")
    print(f"      Device: {DEVICE.upper()} | PyTorch: {torch.__version__}")
    print("=========================================================")

    work_dir = Path("./data")
    export_dir = Path("./export")

    genres_dir = download_gtzan_if_needed(work_dir)
    train_slices, val_slices = prepare_dataset_splits(genres_dir, split_ratio=0.8)

    train_ds = MusicDataset(train_slices, is_train=True)
    val_ds = MusicDataset(val_slices, is_train=False)

    num_workers = 2 if DEVICE == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = MusicAudioNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    criterion_genre = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_vibe = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val_acc = 0.0
    best_weights_path = work_dir / "best_model.pth"

    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_g_acc, train_v_acc = train_one_epoch(
            model, train_loader, optimizer, criterion_genre, criterion_vibe, scaler
        )
        val_loss, val_g_acc, val_v_acc = evaluate(model, val_loader, criterion_genre, criterion_vibe)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} (Genre: {train_g_acc:.1f}%, Vibe: {train_v_acc:.1f}%) | "
            f"Val Loss: {val_loss:.4f} (Genre: {val_g_acc:.1f}%, Vibe: {val_v_acc:.1f}%)"
        )

        if val_g_acc > best_val_acc:
            best_val_acc = val_g_acc
            torch.save(model.state_dict(), best_weights_path)
            print(f"  >>> Best model saved! (Val Genre Acc: {val_g_acc:.2f}%)")

    # Load best weights and export
    print(f"\nLoading best checkpoint from {best_weights_path}...")
    model.load_state_dict(torch.load(best_weights_path, map_location="cpu"))
    export_to_onnx(model, export_dir)

    print("\nTraining and Export Complete!")


if __name__ == "__main__":
    main()
