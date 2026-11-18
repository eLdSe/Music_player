import sys
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import onnxruntime as ort

def load_audio_mel(audio_path: str, target_sr: int = 16000, duration_sec: float = 10.0, n_mels: int = 128):
    waveform, sr = torchaudio.load(audio_path)
    
    # Mono conversion
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample
    if sr != target_sr:
        waveform = T.Resample(sr, target_sr)(waveform)
        
    num_samples = int(target_sr * duration_sec)
    total_samples = waveform.shape[1]
    
    # Take 10 seconds slice from the middle of the track (most energetic part)
    if total_samples > num_samples:
        start = (total_samples - num_samples) // 2
        chunk = waveform[:, start:start + num_samples]
    else:
        pad_len = num_samples - total_samples
        chunk = torch.nn.functional.pad(waveform, (0, pad_len))
        
    # Mel-spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        n_mels=n_mels,
        power=2.0
    )
    amp_to_db = T.AmplitudeToDB()
    mel = amp_to_db(mel_transform(chunk))
    
    # Normalize
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.unsqueeze(0).numpy() # (1, 1, 128, 313)


def predict(onnx_model_path: str, labels_json_path: str, audio_file_path: str):
    with open(labels_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    genres = meta["genres"]
    vibes = meta["vibes"]
    
    print(f"\n[AI Classifier] Loading model: {onnx_model_path}")
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    
    print(f"[AI Classifier] Analyzing track: {audio_file_path}")
    mel_input = load_audio_mel(
        audio_file_path, 
        target_sr=meta["sample_rate"], 
        duration_sec=meta["chunk_duration_sec"],
        n_mels=meta["n_mels"]
    )
    
    outputs = session.run(None, {"mel_spectrogram": mel_input})
    genre_logits, vibe_logits = outputs[0][0], outputs[1][0]
    
    # Softmax probabilities
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    genre_probs = softmax(genre_logits)
    vibe_probs = softmax(vibe_logits)
    
    top_genres_idx = np.argsort(genre_probs)[::-1][:3]
    top_vibe_idx = np.argmax(vibe_probs)
    
    print("\n" + "="*50)
    print("             РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ            ")
    print("="*50)
    print(f"🔥 Основной жанр:  {genres[top_genres_idx[0]].upper()} ({genre_probs[top_genres_idx[0]]*100:.1f}%)")
    print(f"✨ Настроение/Вайб: {vibes[top_vibe_idx]} ({vibe_probs[top_vibe_idx]*100:.1f}%)")
    print("\nТоп-3 предсказанных жанра:")
    for idx in top_genres_idx:
        bar = "█" * int(genre_probs[idx] * 25)
        print(f"  • {genres[idx]:10} : {genre_probs[idx]*100:5.1f}% | {bar}")
    print("="*50 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python test_inference.py <путь_к_аудио_файлу.mp3/.wav>")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    model_path = "./export/music_classifier.int8.onnx"
    if not Path(model_path).exists():
        model_path = "./export/music_classifier.onnx"
        
    labels_path = "./export/labels.json"
    predict(model_path, labels_path, audio_path)
