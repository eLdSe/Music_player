# 📱 SoundPulse AI — Версия для iPhone 12 (iOS PWA + GitHub Pages)

Мобильная автономная версия плеера и AI-классификатора **SoundPulse AI** со встроенной моделью **`MusicAudioNetV2` (`music_classifier.int8.onnx`, 5.3 МБ)**:
- **5 вайбов:** `Драйв / Энергия`, `Вечеринка / Танцы`, `Чилл / Отдых`, `Лирика / Грусть`, `Фокус / Концентрация`
- **12 жанров:** `Blues`, `Classical`, `Country`, `Electronic`, `Folk`, `Hip-Hop`, `Jazz`, `Lo-Fi`, `Metal`, `Pop`, `Reggae`, `Rock`
- Работает прямо на процессоре **iPhone 12 (A14 Bionic)** через `onnxruntime-web` (WebAssembly) без внешнего сервера.
- Сохраняет добавленные и скачанные песни офлайн во внутреннюю базу iPhone (**IndexedDB**) и воспроизводит в фоне с управлением на заблокированном экране (`MediaSession API`).

---

## Как выложить на GitHub и установить на iPhone 12 (за 2 минуты)

### Шаг 1. Создай репозиторий на GitHub и запушь папку `pwa` (или весь проект)
В терминале из папки `D:\Projects\my_tests\music_player`:

```powershell
cd D:\Projects\my_tests\music_player
git init
git add pwa .github
git commit -m "Add SoundPulse iOS PWA with MusicAudioNetV2"
git branch -M main
git remote add origin https://github.com/<ТВОЙ_НИК>/<ИМЯ_РЕПО>.git
git push -u origin main
```

### Шаг 2. Включи GitHub Pages в настройках репозитория
1. Открой репозиторий на GitHub → **Settings → Pages**.
2. В разделе **Build and deployment → Source** выбери **GitHub Actions** (если пушишь весь проект с `.github/workflows/deploy-pwa.yml`)
   *(либо если запушил только содержимое папки `pwa` в корень — выбери **Deploy from a branch → `main` / `/ (root)`**).*
3. Через ~30 секунд появится ссылка вида:
   `https://<твой-ник>.github.io/<имя-репо>/`

### Шаг 3. Установи как приложение на iPhone 12
1. Открой эту ссылку в **Safari** на **iPhone 12**.
2. Нажми кнопку **«Поделиться»** (квадрат со стрелкой вверх внизу).
3. Прокрути чуть вниз и выбери **«На экран „Домой“» (Add to Home Screen)** → **Добавить**.
4. Запусти иконку **SoundPulse** с домашнего экрана — приложение откроется на весь экран без рамок браузера, закэширует модель `music_classifier.int8.onnx` и будет работать даже офлайн!

---

## Как проверить локально на ПК прямо сейчас
```powershell
cd D:\Projects\my_tests\music_player\pwa
python -m http.server 8080
```
Открой `http://localhost:8080` в браузере (или с iPhone в одной Wi-Fi сети по локальному IP компьютера `http://<IP_ПК>:8080`).
