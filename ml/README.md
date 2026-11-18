# 🎵 Music AI Classifier — Обучение модели для Rust-плеера

Модель **MusicAudioNet** классифицирует трек одновременно по двум критериям (**Multi-Task Learning**):
1. **Жанр (10 классов):** Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.
2. **Вайб / Энергия (3 класса):**
   - ⚡ **Драйв / Энергия** (Rock, Metal, Hiphop, Disco)
   - ☕ **Чилл / Отдых** (Pop, Reggae, Blues, Country)
   - 🧠 **Фокус / Концентрация** (Classical, Jazz)

Модель оптимизирована для встраивания в Rust: на входе принимает Mel-спектрограмму, а размер сжатого файла в INT8 составляет всего **~1.2 МБ**! Время отклика на CPU — **3–5 миллисекунд**.

---

## Вариант 1. Обучение в Kaggle (Рекомендуется — бесплатная GPU)

1. Зайди на [kaggle.com](https://www.kaggle.com/) и нажми **+ Create → New Notebook**.
2. В меню ноутбука: **File → Import Notebook** и выбери файл:
   `d:\Projects\my_tests\music_player\ml\train_kaggle.ipynb`
3. В панели настроек справа (Notebook Options):
   - **Accelerator:** выбери **GPU T4 x2** или **GPU P100**.
   - **Internet:** переключи в **On**.
4. Нажми **Run All** (или `Shift + Enter` по ячейкам).
   - Скрипт сам скачает датасет GTZAN, нарежет сэмплы и обучит модель за ~10–15 минут.
5. В конце в панели справа появится файл **`music_model_export.zip`**.
   - Скачай его и распакуй в папку: `music_player/ml/export/` (внутри будут `music_classifier.int8.onnx` и `labels.json`).

---

## Вариант 2. Обучение локально (если есть Python с PyTorch)

```bash
pip install torch torchaudio onnx onnxruntime
cd d:\Projects\my_tests\music_player\ml
python train.py
```

После завершения готовые модели появятся в папке `ml/export/`.

---

## Как проверить модель на любом своём треке:

```bash
python test_inference.py "C:\Music\MySong.mp3"
```
Скрипт выдаст вероятность жанров и определит вайб трека!
