/* SoundPulse AI — iPhone 12 PWA with On-Device MusicAudioNetV2 (ONNX Runtime Web) */

const SAMPLE_RATE = 16000;
const N_FFT = 1024;
const HOP_LENGTH = 512;
const N_MELS = 128;
const CHUNK_SEC = 10.0;
const CHUNK_SAMPLES = 160000;
const TARGET_FRAMES = 313;

const VIBE_COLORS = [
  "#e5566f", // 0: Драйв / Энергия
  "#f59e0b", // 1: Вечеринка / Танцы
  "#10b981", // 2: Чилл / Отдых
  "#c084fc", // 3: Лирика / Грусть
  "#38bdf8"  // 4: Фокус / Концентрация
];

const GENRE_NAMES = {
  blues: "Blues",
  classical: "Classical",
  country: "Country",
  disco: "Disco",
  electronic: "Electronic",
  folk: "Folk",
  hiphop: "Hip-Hop",
  jazz: "Jazz",
  lofi_ambient: "Lo-Fi",
  metal: "Metal",
  pop: "Pop",
  reggae: "Reggae",
  rock: "Rock"
};

let ortSession = null;
let modelLabels = {
  genres: ["blues", "classical", "country", "electronic", "folk", "hiphop", "jazz", "lofi_ambient", "metal", "pop", "reggae", "rock"],
  vibes: ["Драйв / Энергия", "Вечеринка / Танцы", "Чилл / Отдых", "Лирика / Грусть", "Фокус / Концентрация"]
};

let tracks = [];
let activeFilter = "all";
let currentTrackId = null;
let inspectedTrackId = null;
let currentObjectUrl = null;
let isUserSeeking = false;

// Precomputed DSP tables for MelSpectrogram
const hannWindow = new Float32Array(N_FFT);
const bitReverse = new Int32Array(N_FFT);
const cosTable = new Float32Array(N_FFT / 2);
const sinTable = new Float32Array(N_FFT / 2);
let melBands = [];

function initDspTables() {
  for (let i = 0; i < N_FFT; i++) {
    hannWindow[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / N_FFT));
  }
  const bits = Math.log2(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    let rev = 0, x = i;
    for (let b = 0; b < bits; b++) {
      rev = (rev << 1) | (x & 1);
      x >>= 1;
    }
    bitReverse[i] = rev;
  }
  for (let i = 0; i < N_FFT / 2; i++) {
    const angle = (-2.0 * Math.PI * i) / N_FFT;
    cosTable[i] = Math.cos(angle);
    sinTable[i] = Math.sin(angle);
  }

  // HTK Mel filterbank (matches torchaudio.transforms.MelSpectrogram)
  const nFreqs = N_FFT / 2 + 1;
  const fMax = SAMPLE_RATE / 2.0;
  const hzToMel = (hz) => 2595.0 * Math.log10(1.0 + hz / 700.0);
  const melToHz = (mel) => 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);

  const melMin = hzToMel(0);
  const melMax = hzToMel(fMax);
  const fPts = new Float64Array(N_MELS + 2);
  for (let i = 0; i < N_MELS + 2; i++) {
    fPts[i] = melToHz(melMin + ((melMax - melMin) * i) / (N_MELS + 1));
  }

  const allFreqs = new Float64Array(nFreqs);
  for (let b = 0; b < nFreqs; b++) {
    allFreqs[b] = (fMax * b) / (nFreqs - 1);
  }

  melBands = new Array(N_MELS);
  for (let m = 0; m < N_MELS; m++) {
    const fL = fPts[m], fC = fPts[m + 1], fR = fPts[m + 2];
    const lSpan = fC - fL, rSpan = fR - fC;
    let first = -1, last = -1;
    const dense = new Float32Array(nFreqs);
    for (let b = 0; b < nFreqs; b++) {
      const f = allFreqs[b];
      const down = lSpan > 0 ? (f - fL) / lSpan : 0;
      const up = rSpan > 0 ? (fR - f) / rSpan : 0;
      const w = Math.max(0, Math.min(down, up));
      if (w > 0) {
        if (first === -1) first = b;
        last = b;
        dense[b] = w;
      }
    }
    if (first === -1) {
      melBands[m] = { start: 0, weights: new Float32Array(0) };
    } else {
      melBands[m] = { start: first, weights: dense.slice(first, last + 1) };
    }
  }
}

function computeMelSpectrogram(pcm16k) {
  const pad = N_FFT / 2;
  const len = pcm16k.length;
  const paddedLen = len + 2 * pad;
  const padded = new Float32Array(paddedLen);
  padded.set(pcm16k, pad);

  if (len > 1) {
    for (let i = 0; i < pad; i++) {
      padded[i] = pcm16k[Math.min(pad - i, len - 1)];
      padded[pad + len + i] = pcm16k[Math.max(len - 2 - i, 0)];
    }
  }

  const numFrames = Math.min(TARGET_FRAMES, Math.floor((paddedLen - N_FFT) / HOP_LENGTH) + 1);
  const nBins = N_FFT / 2 + 1;
  const real = new Float32Array(N_FFT);
  const imag = new Float32Array(N_FFT);
  const power = new Float32Array(nBins);
  const out = new Float32Array(N_MELS * TARGET_FRAMES);
  out.fill(-100.0);

  for (let t = 0; t < numFrames; t++) {
    const offset = t * HOP_LENGTH;
    for (let i = 0; i < N_FFT; i++) {
      const rev = bitReverse[i];
      real[rev] = padded[offset + i] * hannWindow[i];
      imag[rev] = 0;
    }

    // Radix-2 FFT
    for (let size = 2; size <= N_FFT; size <<= 1) {
      const half = size >> 1;
      const step = N_FFT / size;
      for (let i = 0; i < N_FFT; i += size) {
        let k = 0;
        for (let j = i; j < i + half; j++) {
          const p = j + half;
          const c = cosTable[k], s = sinTable[k];
          const tr = real[p] * c - imag[p] * s;
          const ti = real[p] * s + imag[p] * c;
          real[p] = real[j] - tr;
          imag[p] = imag[j] - ti;
          real[j] += tr;
          imag[j] += ti;
          k += step;
        }
      }
    }

    for (let b = 0; b < nBins; b++) {
      power[b] = real[b] * real[b] + imag[b] * imag[b];
    }

    for (let m = 0; m < N_MELS; m++) {
      const band = melBands[m];
      const w = band.weights;
      const st = band.start;
      let energy = 0.0;
      for (let k = 0; k < w.length; k++) {
        energy += power[st + k] * w[k];
      }
      out[m * TARGET_FRAMES + t] = 10.0 * Math.log10(Math.max(energy, 1e-10));
    }
  }

  let sum = 0.0;
  for (let i = 0; i < out.length; i++) sum += out[i];
  const mean = sum / out.length;

  let sq = 0.0;
  for (let i = 0; i < out.length; i++) {
    const d = out[i] - mean;
    sq += d * d;
  }
  const std = Math.sqrt(sq / Math.max(1, out.length - 1)) + 1e-6;
  for (let i = 0; i < out.length; i++) {
    out[i] = (out[i] - mean) / std;
  }
  return out;
}

function estimateBpmAndEnergy(pcm16k) {
  const numFrames = Math.max(1, Math.floor((pcm16k.length - N_FFT) / HOP_LENGTH) + 1);
  const rms = new Float32Array(numFrames);
  const onset = new Float32Array(numFrames);
  let totalSq = 0;

  for (let t = 0; t < numFrames; t++) {
    const st = t * HOP_LENGTH;
    let sq = 0;
    for (let i = st; i < st + N_FFT && i < pcm16k.length; i++) {
      sq += pcm16k[i] * pcm16k[i];
    }
    rms[t] = Math.sqrt(sq / N_FFT);
    totalSq += sq;
    if (t > 0) onset[t] = Math.max(0, rms[t] - rms[t - 1]);
  }

  const overallRms = Math.sqrt(totalSq / Math.max(1, pcm16k.length));
  const rmsDb = 20 * Math.log10(Math.max(overallRms, 1e-5));
  const loudScore = Math.max(0, Math.min(1, (rmsDb + 32) / 24));

  // Autocorrelation for BPM
  let meanOnset = 0;
  for (let i = 0; i < numFrames; i++) meanOnset += onset[i];
  meanOnset /= numFrames;

  const centered = new Float32Array(numFrames);
  let e0 = 0;
  for (let i = 0; i < numFrames; i++) {
    centered[i] = onset[i] - meanOnset;
    e0 += centered[i] * centered[i];
  }

  const fps = SAMPLE_RATE / HOP_LENGTH;
  let bestLag = 16, bestScore = -1;
  if (e0 > 1e-8) {
    for (let lag = 10; lag <= 32 && lag < numFrames - 4; lag++) {
      let dot = 0;
      for (let i = 0; i < numFrames - lag; i++) dot += centered[i] * centered[i + lag];
      const acf = dot / e0;
      const candBpm = (60 * fps) / lag;
      const prior = Math.exp(-0.5 * Math.pow(Math.log2(candBpm / 112), 2) / (0.65 * 0.65));
      if (acf * prior > bestScore) {
        bestScore = acf * prior;
        bestLag = lag;
      }
    }
  }

  const bpm = Math.max(60, Math.min(185, Math.round((60 * fps) / bestLag)));
  const tempoFactor = Math.max(0, Math.min(1, (bpm - 66) / 68));
  const energy = Math.max(0.05, Math.min(0.99, loudScore * 0.65 + tempoFactor * 0.35));
  return { bpm, energy };
}

function softmax(arr) {
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
  let sum = 0;
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    out[i] = Math.exp(arr[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum || 1;
  return out;
}

// ==================== IndexedDB Offline Storage ====================
const DB_NAME = "SoundPulsePWA_DB";
const DB_VERSION = 1;

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains("tracks")) {
        db.createObjectStore("tracks", { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbGetAllTracks() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("tracks", "readonly");
    const req = tx.objectStore("tracks").getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}

async function dbGetTrack(id) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("tracks", "readonly");
    const req = tx.objectStore("tracks").get(id);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

async function dbSaveTrack(track) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("tracks", "readwrite");
    tx.objectStore("tracks").put(track);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbDeleteTrack(id) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("tracks", "readwrite");
    tx.objectStore("tracks").delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbClearAll() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("tracks", "readwrite");
    tx.objectStore("tracks").clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// ==================== Audio Decoding & ONNX Inference ====================
async function decodeAudioBlobTo16kChunks(blob) {
  const arrayBuf = await blob.arrayBuffer();
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const tempCtx = new AudioCtx();
  let audioBuffer;
  try {
    audioBuffer = await tempCtx.decodeAudioData(arrayBuf);
  } finally {
    if (tempCtx.close) await tempCtx.close();
  }

  const durationSec = audioBuffer.duration;
  const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;

  let starts = [0];
  if (durationSec >= CHUNK_SEC * 2.2) {
    const maxSt = Math.max(0, durationSec - CHUNK_SEC);
    starts = [
      Math.max(0, Math.min(durationSec * 0.25 - CHUNK_SEC * 0.5, maxSt)),
      Math.max(0, Math.min(durationSec * 0.50 - CHUNK_SEC * 0.5, maxSt)),
      Math.max(0, Math.min(durationSec * 0.75 - CHUNK_SEC * 0.5, maxSt))
    ];
  } else if (durationSec > CHUNK_SEC) {
    starts = [(durationSec - CHUNK_SEC) * 0.5];
  }

  const chunks = [];
  for (const st of starts) {
    const offline = new OfflineCtx(1, CHUNK_SAMPLES, SAMPLE_RATE);
    const src = offline.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(offline.destination);
    src.start(0, st, Math.min(CHUNK_SEC, durationSec - st));
    const rendered = await offline.startRendering();
    const pcm = new Float32Array(CHUNK_SAMPLES);
    const chData = rendered.getChannelData(0);
    pcm.set(chData.subarray(0, Math.min(CHUNK_SAMPLES, chData.length)));
    chunks.push(pcm);
  }

  return { chunks, durationSec };
}

async function analyzeTrackBlob(blob) {
  const { chunks, durationSec } = await decodeAudioBlobTo16kChunks(blob);
  const dsp = estimateBpmAndEnergy(chunks[Math.floor(chunks.length / 2)]);

  const numGenres = modelLabels.genres.length;
  const numVibes = modelLabels.vibes.length;
  const sumGenre = new Float64Array(numGenres);
  const sumVibe = new Float64Array(numVibes);

  for (const pcm of chunks) {
    const mel = computeMelSpectrogram(pcm);
    const tensor = new ort.Tensor("float32", mel, [1, 1, N_MELS, TARGET_FRAMES]);
    const out = await ortSession.run({ mel_spectrogram: tensor });
    const gData = out.genre_logits.data;
    const vData = out.vibe_logits.data;
    for (let i = 0; i < numGenres; i++) sumGenre[i] += gData[i] || 0;
    for (let i = 0; i < numVibes; i++) sumVibe[i] += vData[i] || 0;
  }

  const gProbs = softmax(Array.from(sumGenre, (v) => v / chunks.length));
  const vProbs = softmax(Array.from(sumVibe, (v) => v / chunks.length));

  const genreScores = modelLabels.genres.map((key, idx) => ({
    key,
    name: GENRE_NAMES[key] || key,
    prob: gProbs[idx] || 0
  })).sort((a, b) => b.prob - a.prob);

  const vibeScores = modelLabels.vibes.map((name, idx) => ({
    index: idx,
    name,
    prob: vProbs[idx] || 0,
    color: VIBE_COLORS[idx % VIBE_COLORS.length]
  }));

  let bestVibe = vibeScores[0];
  for (const v of vibeScores) {
    if (v.prob > bestVibe.prob) bestVibe = v;
  }

  return {
    durationSec,
    genre: genreScores[0].key,
    genreDisplay: genreScores[0].name,
    genreConfidence: genreScores[0].prob,
    vibe: bestVibe.name,
    vibeShort: bestVibe.name.split("/")[0].trim(),
    vibeIndex: bestVibe.index,
    vibeConfidence: bestVibe.prob,
    vibeColor: bestVibe.color,
    topGenres: genreScores.slice(0, 5),
    vibeScores,
    bpm: dsp.bpm,
    energy: dsp.energy
  };
}

// ==================== UI & Player State ====================
// IMPORTANT: audioEl plays natively without createMediaElementSource routing.
// Routing HTML5 audio through Web Audio AudioContext causes iOS WebKit to immediately
// mute the audio whenever the screen locks or app goes to background.
// Leaving audioEl unhijacked allows continuous, non-muted background & lock screen playback!
const audioEl = document.getElementById("audioEl");
let vizRafId = null;

// Background / Lock Screen auto-advance state
let isTransitioning = false;
let transitionTimer = null;
let userPaused = false;
let lastObservedTime = 0;
let lastManualSeekAt = 0;
let lastAdvanceAt = 0;
let lastPosSyncAt = 0;
let errorRecoveredTrackId = null;
let playGeneration = 0;
let playbackStartedWallMs = 0;
let playbackStartAudioPos = 0;

function isLoopWrapDetected(cur, dur) {
  if (!Number.isFinite(dur) || dur <= 2.0) return false;
  if ((Date.now() - lastManualSeekAt) < 1500) return false;
  if (cur >= Math.min(5.0, dur * 0.4)) return false;
  const observedPassedHalf = lastObservedTime > Math.min(dur * 0.5, dur - 1.5) && (lastObservedTime - cur) > 1.5;
  const elapsedSec = playbackStartedWallMs > 0
    ? ((Date.now() - playbackStartedWallMs) / 1000) * (audioEl.playbackRate || 1)
    : 0;
  const wallClockPassedEof = (playbackStartAudioPos + elapsedSec) >= (dur - 1.5);
  return observedPassedHalf || wallClockPassedEof;
}

function setAudioSessionPlayback() {
  try {
    if ("audioSession" in navigator && navigator.audioSession) {
      navigator.audioSession.type = "playback";
    }
  } catch (_) {}
}

function ensureTypedAudioBlob(blob, title = "") {
  if (!blob) return blob;
  if (blob.type && blob.type.startsWith("audio/")) return blob;
  const lower = (title || "").toLowerCase();
  let mime = "audio/mpeg";
  if (lower.endsWith(".m4a") || lower.endsWith(".mp4")) mime = "audio/mp4";
  else if (lower.endsWith(".wav")) mime = "audio/wav";
  else if (lower.endsWith(".ogg") || lower.endsWith(".opus")) mime = "audio/ogg";
  else if (lower.endsWith(".flac")) mime = "audio/flac";
  else if (lower.endsWith(".aac")) mime = "audio/aac";
  return blob.slice(0, blob.size, mime);
}

function updateActiveTrackRowHighlight() {
  document.querySelectorAll("#trackList .track-row").forEach((row) => {
    row.classList.toggle("playing", row.dataset.id === currentTrackId);
  });
}

function formatTime(sec) {
  if (!sec || !isFinite(sec) || sec < 0) return "00:00";
  const s = Math.floor(sec);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

function parseFileName(fileName) {
  const base = fileName.replace(/\.[^/.]+$/, "");
  const parts = base.split(/\s+[-–—]\s+/);
  if (parts.length >= 2) {
    return { artist: parts[0].trim(), title: parts.slice(1).join(" - ").trim() };
  }
  return { artist: "Неизвестный исполнитель", title: base.trim() || "Трек" };
}

function renderFilterChips() {
  const row = document.getElementById("filterChipsRow");
  const chips = [
    { key: "all", label: "Все", count: tracks.length, color: null }
  ];

  modelLabels.vibes.forEach((vName, idx) => {
    const shortName = vName.split("/")[0].trim();
    const count = tracks.filter((t) => t.vibeIndex === idx).length;
    chips.push({
      key: `vibe:${idx}`,
      label: shortName,
      count,
      color: VIBE_COLORS[idx % VIBE_COLORS.length]
    });
  });

  modelLabels.genres.forEach((gKey) => {
    const count = tracks.filter((t) => t.genre === gKey).length;
    if (count > 0 || activeFilter === `genre:${gKey}`) {
      chips.push({
        key: `genre:${gKey}`,
        label: GENRE_NAMES[gKey] || gKey,
        count,
        color: null
      });
    }
  });

  row.innerHTML = chips.map((c) => `
    <button class="filter-chip ${activeFilter === c.key ? "active" : ""}" data-filter="${c.key}" type="button">
      ${c.color ? `<span class="chip-dot" style="background:${c.color}"></span>` : ""}
      <span>${c.label}</span>
      <span class="chip-count">${c.count}</span>
    </button>
  `).join("");

  row.querySelectorAll(".filter-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeFilter = btn.dataset.filter;
      renderFilterChips();
      renderTrackList();
    });
  });
}

function getFilteredTracks() {
  const q = document.getElementById("searchInput").value.trim().toLowerCase();
  return tracks.filter((t) => {
    if (activeFilter.startsWith("vibe:")) {
      const vIdx = parseInt(activeFilter.split(":")[1], 10);
      if (t.vibeIndex !== vIdx) return false;
    } else if (activeFilter.startsWith("genre:")) {
      const gKey = activeFilter.split(":")[1];
      if (t.genre !== gKey) return false;
    }
    if (!q) return true;
    return (
      t.title.toLowerCase().includes(q) ||
      t.artist.toLowerCase().includes(q) ||
      (t.genreDisplay || "").toLowerCase().includes(q) ||
      (t.vibe || "").toLowerCase().includes(q)
    );
  });
}

function renderTrackList() {
  const listEl = document.getElementById("trackList");
  const emptyBox = document.getElementById("emptyLibraryBox");
  const filtered = getFilteredTracks();

  if (filtered.length === 0) {
    listEl.innerHTML = "";
    emptyBox.style.display = "block";
    return;
  }

  emptyBox.style.display = "none";
  listEl.innerHTML = filtered.map((t) => {
    const isPlaying = t.id === currentTrackId && !audioEl.paused;
    const color = t.vibeColor || "#10b981";
    const pct = t.genreConfidence ? `${Math.round(t.genreConfidence * 100)}%` : "";
    return `
      <div class="track-row ${t.id === currentTrackId ? "playing" : ""}" data-id="${t.id}">
        <div class="track-main">
          <div class="track-line1">
            <span class="track-title">${escapeHtml(t.title)}</span>
            <span class="track-dur">${formatTime(t.durationSec)}</span>
          </div>
          <div class="track-line2">
            <span class="vibe-pill" style="background:${color}22; color:${color};">
              <span class="chip-dot" style="background:${color}; width:5px; height:5px;"></span>
              ${escapeHtml(t.vibeShort || "Анализ…")}
            </span>
            <span class="genre-tag">${escapeHtml(t.genreDisplay || "—")} ${pct}</span>
            <span>· ${escapeHtml(t.artist)}</span>
          </div>
        </div>
        <button class="btn-row-info" data-inspect="${t.id}" type="button" aria-label="AI анализ">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="8" cy="8" r="6" />
            <path d="M8 7v4M8 5h.01" stroke-linecap="round" />
          </svg>
        </button>
      </div>
    `;
  }).join("");

  listEl.querySelectorAll(".track-row").forEach((row) => {
    row.addEventListener("click", (e) => {
      const inspectBtn = e.target.closest("[data-inspect]");
      if (inspectBtn) {
        e.stopPropagation();
        openInspectorSheet(inspectBtn.dataset.inspect);
        return;
      }
      playTrackById(row.dataset.id);
    });
  });
}

function updateStatsView() {
  document.getElementById("statTotalTracks").textContent = String(tracks.length);
  if (tracks.length === 0) {
    document.getElementById("statAvgBpm").textContent = "—";
    document.getElementById("statAvgEnergy").textContent = "—";
    document.getElementById("statStorageMb").textContent = "0 МБ";
    document.getElementById("statVibeBars").innerHTML = "";
    return;
  }

  let sumBpm = 0, sumEnergy = 0, totalBytes = 0;
  tracks.forEach((t) => {
    sumBpm += t.bpm || 100;
    sumEnergy += t.energy || 0.5;
    if (t.blob && t.blob.size) totalBytes += t.blob.size;
  });

  document.getElementById("statAvgBpm").textContent = `${Math.round(sumBpm / tracks.length)} BPM`;
  document.getElementById("statAvgEnergy").textContent = `${Math.round((sumEnergy / tracks.length) * 100)}%`;
  document.getElementById("statStorageMb").textContent = `${(totalBytes / (1024 * 1024)).toFixed(1)} МБ`;

  const barsHtml = modelLabels.vibes.map((vName, idx) => {
    const cnt = tracks.filter((t) => t.vibeIndex === idx).length;
    const pct = Math.round((cnt / tracks.length) * 100);
    const color = VIBE_COLORS[idx % VIBE_COLORS.length];
    return `
      <div class="bar-list-row">
        <div class="bar-list-head">
          <span>${escapeHtml(vName)}</span>
          <span class="num" style="color:var(--muted)">${cnt} (${pct}%)</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct}%; background:${color};"></div>
        </div>
      </div>
    `;
  }).join("");
  document.getElementById("statVibeBars").innerHTML = barsHtml;
}

async function addAudioBlobs(fileEntries) {
  if (!fileEntries.length) return;
  const banner = document.getElementById("queueBanner");
  const queueText = document.getElementById("queueText");
  const queueFill = document.getElementById("queueBarFill");

  banner.classList.add("visible");
  let done = 0;

  for (const entry of fileEntries) {
    queueText.textContent = `AI-анализ (${done + 1}/${fileEntries.length}): ${entry.title}`;
    queueFill.style.width = `${Math.round((done / fileEntries.length) * 100)}%`;

    try {
      const res = await analyzeTrackBlob(entry.blob);
      const trackRecord = {
        id: `tr_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
        title: entry.title,
        artist: entry.artist,
        blob: entry.blob,
        ...res
      };
      await dbSaveTrack(trackRecord);
      tracks.unshift(trackRecord);
      renderFilterChips();
      renderTrackList();
      updateStatsView();
    } catch (err) {
      console.error("Analysis failed:", err);
    }

    done++;
    queueFill.style.width = `${Math.round((done / fileEntries.length) * 100)}%`;
  }

  setTimeout(() => banner.classList.remove("visible"), 800);
}

// ==================== Playback & Lock Screen MediaSession ====================
function getNextTrackInPool(dir = 1) {
  const list = getFilteredTracks();
  const pool = list.length ? list : tracks;
  if (!pool.length) return null;
  const idx = pool.findIndex((t) => t.id === currentTrackId);
  if (idx === -1) return dir >= 0 ? pool[0] : pool[pool.length - 1];
  const nextIdx = (idx + dir + pool.length) % pool.length;
  return pool[nextIdx] || null;
}

function updateMediaSessionPosition() {
  if (!("mediaSession" in navigator) || typeof navigator.mediaSession.setPositionState !== "function") return;
  const dur = audioEl.duration;
  if (!Number.isFinite(dur) || dur <= 0) return;
  const cur = Math.max(0, Math.min(audioEl.currentTime || 0, dur));
  try {
    navigator.mediaSession.setPositionState({
      duration: dur,
      playbackRate: audioEl.playbackRate || 1,
      position: cur
    });
    lastPosSyncAt = Date.now();
  } catch (_) {}
}

function attemptPlay(expectedTrackId, expectedGen, attempt = 0) {
  if (expectedGen !== playGeneration || currentTrackId !== expectedTrackId || userPaused) return;
  setAudioSessionPlayback();
  const p = audioEl.play();
  if (p && typeof p.then === "function") {
    p.then(() => {
      if (expectedGen !== playGeneration || currentTrackId !== expectedTrackId || userPaused) return;
      isTransitioning = false;
      clearTimeout(transitionTimer);
      if ("mediaSession" in navigator) navigator.mediaSession.playbackState = "playing";
      updateMediaSessionPosition();
    }).catch((err) => {
      if (expectedGen !== playGeneration || currentTrackId !== expectedTrackId || userPaused) return;
      console.warn(`Play attempt ${attempt + 1} failed:`, err);
      if (attempt < 3) {
        let retried = false;
        const doRetry = () => {
          if (retried || expectedGen !== playGeneration || currentTrackId !== expectedTrackId || userPaused) return;
          retried = true;
          attemptPlay(expectedTrackId, expectedGen, attempt + 1);
        };
        if (audioEl.readyState < 2) {
          audioEl.addEventListener("canplay", doRetry, { once: true });
        }
        setTimeout(doRetry, (attempt + 1) * 350);
      } else {
        isTransitioning = false;
        clearTimeout(transitionTimer);
      }
    });
  } else {
    isTransitioning = false;
    clearTimeout(transitionTimer);
  }
}

function playTrackById(id) {
  const track = tracks.find((t) => t.id === id);
  if (!track || !track.blob) return;

  setAudioSessionPlayback();
  const gen = ++playGeneration;
  isTransitioning = true;
  userPaused = false;
  lastObservedTime = 0;
  lastAdvanceAt = Date.now();
  playbackStartedWallMs = lastAdvanceAt;
  playbackStartAudioPos = 0;
  errorRecoveredTrackId = null;

  clearTimeout(transitionTimer);
  transitionTimer = setTimeout(() => {
    isTransitioning = false;
  }, 5000);

  const prevUrl = currentObjectUrl;
  currentObjectUrl = URL.createObjectURL(ensureTypedAudioBlob(track.blob, track.title));
  currentTrackId = track.id;
  inspectedTrackId = track.id;

  audioEl.loop = true;
  audioEl.src = currentObjectUrl;
  attemptPlay(track.id, gen, 0);

  if (prevUrl) {
    setTimeout(() => {
      try { URL.revokeObjectURL(prevUrl); } catch (_) {}
    }, 3000);
  }

  setupMediaSession(track);
  updateMiniPlayer(track);
  updateActiveTrackRowHighlight();
  if (!document.hidden) {
    populateSheet(track);
  }
}

function advanceToNextTrack(reason) {
  if (isTransitioning || userPaused) return;
  const now = Date.now();
  if (now - lastAdvanceAt < 1200) return;

  const nextTrack = getNextTrackInPool(1);
  if (!nextTrack) return;

  if (nextTrack.id === currentTrackId) {
    // Only 1 track in pool: let native loop repeat seamlessly without clipping or reloading
    playbackStartedWallMs = now;
    playbackStartAudioPos = 0;
    if (reason === "ended") {
      lastManualSeekAt = now;
      lastObservedTime = 0;
      audioEl.currentTime = 0;
      attemptPlay(currentTrackId, ++playGeneration, 0);
    } else {
      lastAdvanceAt = now;
      lastObservedTime = 0;
      updateMediaSessionPosition();
    }
    return;
  }

  playTrackById(nextTrack.id);
}

function updateMiniPlayer(track) {
  const mini = document.getElementById("miniPlayer");
  if (!track) {
    mini.classList.remove("visible");
    return;
  }
  mini.classList.add("visible");
  document.getElementById("miniTitle").textContent = track.title;
  document.getElementById("miniSub").textContent = `${track.artist} · ${track.vibeShort || ""}`;
  document.getElementById("miniProgressFill").style.background = track.vibeColor || "#10b981";
}

function setupMediaSession(track) {
  if (!("mediaSession" in navigator)) return;
  const art180 = new URL("./apple-touch-icon.png", window.location.href).href;
  const art512 = new URL("./icon-512.png", window.location.href).href;

  navigator.mediaSession.metadata = new MediaMetadata({
    title: track.title,
    artist: `${track.artist} · ${track.vibeShort || ""}`,
    album: `SoundPulse AI (${track.genreDisplay || "Музыка"})`,
    artwork: [
      { src: art180, sizes: "180x180", type: "image/png" },
      { src: art512, sizes: "512x512", type: "image/png" }
    ]
  });
  navigator.mediaSession.playbackState = "playing";

  const safeHandler = (action, fn) => {
    try {
      navigator.mediaSession.setActionHandler(action, fn);
    } catch (_) {}
  };

  safeHandler("play", () => {
    userPaused = false;
    setAudioSessionPlayback();
    if (currentTrackId) attemptPlay(currentTrackId, ++playGeneration, 0);
    else audioEl.play().catch(() => {});
  });
  safeHandler("pause", () => {
    playGeneration++;
    userPaused = true;
    audioEl.pause();
  });
  safeHandler("previoustrack", () => {
    if (audioEl.currentTime > 3) {
      const now = Date.now();
      lastManualSeekAt = now;
      lastObservedTime = 0;
      playbackStartedWallMs = now;
      playbackStartAudioPos = 0;
      audioEl.currentTime = 0;
      updateMediaSessionPosition();
    } else {
      playAdjacent(-1);
    }
  });
  safeHandler("nexttrack", () => playAdjacent(1));
  safeHandler("seekto", (details) => {
    if (details.seekTime != null && Number.isFinite(details.seekTime)) {
      const now = Date.now();
      lastManualSeekAt = now;
      lastObservedTime = details.seekTime;
      playbackStartedWallMs = now;
      playbackStartAudioPos = details.seekTime;
      audioEl.currentTime = details.seekTime;
      updateMediaSessionPosition();
    }
  });
}

function playAdjacent(dir) {
  const nextTrack = getNextTrackInPool(dir);
  if (!nextTrack) return;
  playTrackById(nextTrack.id);
}

function openInspectorSheet(id) {
  const track = tracks.find((t) => t.id === id) || tracks.find((t) => t.id === currentTrackId);
  if (!track) return;
  inspectedTrackId = track.id;
  populateSheet(track);
  document.getElementById("playerSheet").classList.add("open");
  startVisualizerLoop();
}

function populateSheet(track) {
  if (!track) return;
  const color = track.vibeColor || "#10b981";
  document.getElementById("sheetVibeTitle").textContent = track.vibe || "—";
  document.getElementById("sheetVibeTitle").style.color = color;
  document.getElementById("sheetVibePct").textContent = track.vibeConfidence
    ? `${Math.round(track.vibeConfidence * 100)}%`
    : "—";
  document.getElementById("sheetVibePct").style.color = color;

  document.getElementById("sheetTrackTitle").textContent = track.title;
  document.getElementById("sheetTrackArtist").textContent = track.artist;
  document.getElementById("sheetTrackDsp").textContent =
    `${track.genreDisplay || "—"} · ${track.bpm || "—"} BPM · Энергия ${Math.round((track.energy || 0.5) * 100)}%`;

  // Render 5 Vibes bars
  const vibesHtml = (track.vibeScores || []).map((v) => {
    const pct = (v.prob * 100).toFixed(1);
    return `
      <div class="bar-list-row">
        <div class="bar-list-head">
          <span>${escapeHtml(v.name)}</span>
          <span class="num" style="color:var(--muted)">${pct}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct}%; background:${v.color};"></div>
        </div>
      </div>
    `;
  }).join("");
  document.getElementById("sheetVibesList").innerHTML = vibesHtml;

  // Render Top-5 Genres bars
  const genresHtml = (track.topGenres || []).map((g) => {
    const pct = (g.prob * 100).toFixed(1);
    return `
      <div class="bar-list-row">
        <div class="bar-list-head">
          <span>${escapeHtml(g.name)}</span>
          <span class="num" style="color:var(--muted)">${pct}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct}%; background:${color};"></div>
        </div>
      </div>
    `;
  }).join("");
  document.getElementById("sheetGenresList").innerHTML = genresHtml;
}

// ==================== 12-Bar FFT Visualizer ====================
function initVizBars() {
  const container = document.getElementById("sheetVizBars");
  container.innerHTML = Array.from({ length: 12 }, () => `<div class="viz-bar"></div>`).join("");
}

function startVisualizerLoop() {
  if (vizRafId) {
    cancelAnimationFrame(vizRafId);
    vizRafId = null;
  }
  const bars = document.querySelectorAll("#sheetVizBars .viz-bar");

  function tick() {
    const sheetOpen = document.getElementById("playerSheet").classList.contains("open");
    const track = tracks.find((t) => t.id === currentTrackId);
    const color = track?.vibeColor || "#10b981";

    if (!sheetOpen || document.hidden || audioEl.paused) {
      bars.forEach((bar) => {
        bar.style.height = "5px";
        bar.style.background = color;
      });
      vizRafId = null;
      return;
    }

    const time = audioEl.currentTime || 0;
    const bpm = track?.bpm || 110;
    const energy = track?.energy || 0.75;
    const beatSec = 60 / Math.max(40, bpm);
    const phase = (time % beatSec) / beatSec;
    const beatPulse = Math.sin(phase * Math.PI); // Pulse with the track's real beat

    bars.forEach((bar, i) => {
      const harmonic = Math.sin(time * 5.2 + i * 0.9) * 0.5 + 0.5;
      const sub = Math.cos(time * 3.1 + i * 1.3) * 0.3 + 0.3;
      const weight = (beatPulse * 0.55 + harmonic * 0.30 + sub * 0.15) * energy;
      const h = Math.max(5, Math.min(42, Math.round(weight * 44)));
      bar.style.height = `${h}px`;
      bar.style.background = color;
    });

    vizRafId = requestAnimationFrame(tick);
  }
  vizRafId = requestAnimationFrame(tick);
}

// ==================== Deterministic Pointer-Captured Seek Bar ====================
function initSeekEvents() {
  const area = document.getElementById("seekTouchArea");
  const fill = document.getElementById("seekFill");
  const knob = document.getElementById("seekKnob");
  const curText = document.getElementById("seekCurText");

  function getRatioFromEvent(e) {
    const rect = area.getBoundingClientRect();
    const x = (e.clientX ?? (e.touches?.[0]?.clientX || 0)) - rect.left;
    return Math.max(0, Math.min(1, x / Math.max(1, rect.width)));
  }

  function previewRatio(ratio) {
    const pct = `${(ratio * 100).toFixed(2)}%`;
    fill.style.width = pct;
    knob.style.left = pct;
    if (audioEl.duration && isFinite(audioEl.duration)) {
      curText.textContent = formatTime(ratio * audioEl.duration);
    }
  }

  area.addEventListener("pointerdown", (e) => {
    if (!audioEl.duration || !isFinite(audioEl.duration)) return;
    isUserSeeking = true;
    area.setPointerCapture(e.pointerId);
    previewRatio(getRatioFromEvent(e));
  });

  area.addEventListener("pointermove", (e) => {
    if (!isUserSeeking) return;
    previewRatio(getRatioFromEvent(e));
  });

  function commitPointerSeek(e) {
    if (!isUserSeeking) return;
    const ratio = getRatioFromEvent(e);
    isUserSeeking = false;
    if (audioEl.duration && isFinite(audioEl.duration)) {
      const target = ratio * audioEl.duration;
      if (target >= audioEl.duration - 0.35) {
        playAdjacent(1);
      } else {
        lastManualSeekAt = Date.now();
        lastObservedTime = target;
        playbackStartedWallMs = lastManualSeekAt;
        playbackStartAudioPos = target;
        audioEl.currentTime = target;
        updateMediaSessionPosition();
      }
    }
  }

  area.addEventListener("pointerup", commitPointerSeek);
  area.addEventListener("pointercancel", () => { isUserSeeking = false; });
}

// ==================== Audius Search & Direct Download ====================
async function searchAudiusApi(q) {
  try {
    const url = `https://api.audius.co/v1/tracks/search?query=${encodeURIComponent(q)}&app_name=SoundPulsePWA`;
    const resp = await fetch(url);
    if (!resp.ok) return [];
    const json = await resp.json();
    return (json.data || []).filter((item) => item.is_streamable !== false);
  } catch (e) {
    console.warn("Audius query error:", e);
    return [];
  }
}

async function executeAudiusSearch(query, resultsEl, statusText) {
  statusText.innerHTML = `Поиск «<b>${escapeHtml(query)}</b>» в каталоге…`;

  // 1. Try full cleaned query
  let items = await searchAudiusApi(query);

  // 2. If 0 results, try query without hyphens/dashes
  if (!items.length && /[-–—]/.test(query)) {
    const simplified = query.replace(/[-–—]+/g, " ").replace(/\s+/g, " ").trim();
    items = await searchAudiusApi(simplified);
  }

  // 3. If still 0 results, try track title (after dash) or artist (before dash)
  if (!items.length && /[-–—]/.test(query)) {
    const parts = query.split(/[-–—]/).map((s) => s.trim()).filter(Boolean);
    if (parts.length >= 2) {
      items = await searchAudiusApi(parts[1]); // e.g. "Bailando"
      if (!items.length) {
        items = await searchAudiusApi(parts[0]); // e.g. "Enrique Iglesias"
      }
    }
  }

  // 4. Fallback: try first 2 words
  if (!items.length) {
    const words = query.split(/\s+/).slice(0, 2).join(" ");
    if (words && words !== query) {
      items = await searchAudiusApi(words);
    }
  }

  if (!items.length) {
    statusText.innerHTML = `По запросу «<b>${escapeHtml(query)}</b>» ничего не найдено.<br><span style="font-size:12px;opacity:0.85">Попробуйте ввести только название песни или имя исполнителя (например, <i>Bailando</i>).</span>`;
    resultsEl.innerHTML = "";
    return;
  }

  statusText.innerHTML = `Найдено треков: <b>${items.length}</b>. Нажмите «Скачать» для сохранения в память iPhone:`;
  resultsEl.innerHTML = items.slice(0, 15).map((item) => {
    const title = item.title || "Без названия";
    const artist = item.user?.name || "Audius Artist";
    const thumb = item.artwork?.["150x150"] || "";
    const dur = formatTime(item.duration || 0);
    return `
      <div class="dl-result-row">
        ${thumb ? `<img class="dl-thumb" src="${escapeHtml(thumb)}" alt="" loading="lazy" />` : `<div class="dl-thumb"></div>`}
        <div class="dl-meta">
          <div class="track-title">${escapeHtml(title)}</div>
          <div class="track-line2">${escapeHtml(artist)} · <span class="num">${dur}</span></div>
        </div>
        <button class="btn-dl-item" data-id="${escapeHtml(item.id)}" data-title="${escapeHtml(title)}" data-artist="${escapeHtml(artist)}" type="button">Скачать</button>
      </div>
    `;
  }).join("");

  resultsEl.querySelectorAll(".btn-dl-item").forEach((btn) => {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "Загрузка…";
      try {
        const streamUrl = `https://api.audius.co/v1/tracks/${encodeURIComponent(btn.dataset.id)}/stream?app_name=SoundPulsePWA`;
        const r = await fetch(streamUrl);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const blob = await r.blob();
        btn.textContent = "AI-анализ…";
        await addAudioBlobs([{
          title: btn.dataset.title,
          artist: btn.dataset.artist,
          blob
        }]);
        btn.textContent = "В памяти ✓";
        btn.style.background = "rgba(16, 185, 129, 0.2)";
        btn.style.borderColor = "var(--accent-emerald)";
        btn.style.color = "var(--accent-emerald)";
      } catch (e) {
        btn.textContent = "Ошибка";
        btn.disabled = false;
        console.error("Download track error:", e);
      }
    });
  });
}

async function handleDownloadSearch() {
  const input = document.getElementById("dlQueryInput").value.trim();
  if (!input) return;

  const banner = document.getElementById("dlStatusBanner");
  const statusText = document.getElementById("dlStatusText");
  const resultsEl = document.getElementById("dlResultsList");

  banner.classList.add("visible");
  resultsEl.innerHTML = "";

  // 1. YouTube link detection -> resolve song title via oEmbed and search Audius
  const isYouTube = /(?:youtu\.be\/|youtube\.com\/(?:watch\?v=|embed\/|shorts\/))([\w-]{11})/i.test(input);
  if (isYouTube) {
    statusText.textContent = "Распознаём название трека по ссылке YouTube…";
    try {
      const oembedUrl = `https://noembed.com/embed?url=${encodeURIComponent(input)}`;
      const oembedResp = await fetch(oembedUrl);
      if (oembedResp.ok) {
        const oembedData = await oembedResp.json();
        let rawTitle = oembedData.title || "";
        // 1. Strip bracketed / parenthesized metadata: (Lyrics), [Official Video], etc.
        let cleanTitle = rawTitle
          .replace(/\s*[\(\[][^\)\]]*(?:Official|Music|Video|Lyrics|Audio|HD|4K|Visualizer|Clip|Remix|Edit|ft\.|feat\.)[^\)\]]*[\)\]]/gi, "")
          .replace(/\s*\|\s*.*$/, "")
          .replace(/\s*\/\/\s*.*$/, "")
          .trim();

        // 2. Strip trailing "feat. ...", "ft. ...", "featuring ..."
        cleanTitle = cleanTitle.replace(/\s+\b(?:feat\.?|ft\.?|featuring)\b.*$/i, "").trim();

        if (cleanTitle) {
          statusText.innerHTML = `🎵 Распознано: <b>${escapeHtml(cleanTitle)}</b><br><span style="font-size:12px;opacity:0.85">Ищем для скачивания в память iPhone…</span>`;
          await executeAudiusSearch(cleanTitle, resultsEl, statusText);
          return;
        }
      }
    } catch (e) {
      console.warn("YouTube oEmbed lookup failed:", e);
    }

    statusText.innerHTML = `⚠️ YouTube блокирует прямое скачивание через браузер на iPhone (CORS).<br>💡 Введите <b>название песни или исполнителя текстом</b> в строку поиска, либо используйте SoundPulse.exe на ПК (где встроен yt-dlp).`;
    return;
  }

  // 2. Direct Audio URL download (.mp3, .m4a, .wav, .flac, .ogg)
  if (/^https?:\/\//i.test(input)) {
    statusText.textContent = "Скачивание аудио по прямой ссылке…";
    try {
      const resp = await fetch(input);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      const name = decodeURIComponent(input.split("/").pop().split("?")[0]) || "Track.mp3";
      const meta = parseFileName(name);
      statusText.textContent = "Запущен AI-анализ…";
      await addAudioBlobs([{ title: meta.title, artist: meta.artist, blob }]);
      statusText.textContent = "✅ Трек успешно добавлен в память iPhone!";
      document.getElementById("dlQueryInput").value = "";
    } catch (err) {
      statusText.textContent = `Ошибка скачивания: ${err.message}. Убедитесь, что ссылка ведёт напрямую на .mp3/.m4a файл с поддержкой CORS.`;
    }
    return;
  }

  // 3. Normal text query search
  await executeAudiusSearch(input, resultsEl, statusText);
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ==================== Bootstrap ====================
async function initApp() {
  initDspTables();
  initVizBars();
  initSeekEvents();
  setAudioSessionPlayback();
  audioEl.loop = true;

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").then((reg) => {
      reg.update();
    }).catch(() => {});

    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (!window.__swReloaded) {
        window.__swReloaded = true;
        window.location.reload();
      }
    });
  }

  // Load saved tracks from IndexedDB
  try {
    tracks = await dbGetAllTracks();
  } catch (_) {
    tracks = [];
  }
  renderFilterChips();
  renderTrackList();
  updateStatsView();

  // Load labels.json + MusicAudioNetV2 ONNX model
  const statusDot = document.getElementById("aiStatusDot");
  const statusText = document.getElementById("aiStatusText");
  let lastAiError = null;

  document.getElementById("aiStatusPill").addEventListener("click", () => {
    if (ortSession) {
      alert("🟢 SoundPulse AI (MusicAudioNetV2)\nМодель загружена в память iPhone.\n12 жанров + 5 вайбов работают локально офлайн.");
    } else {
      alert("⚠️ Статус AI: " + (lastAiError || "Инициализация нейросети..."));
    }
  });

  try {
    const lblResp = await fetch("./models/labels.json");
    if (lblResp.ok) {
      modelLabels = await lblResp.json();
      renderFilterChips();
      updateStatsView();
    }

    statusText.textContent = "Загрузка AI…";

    // Configure single-threaded WebAssembly execution provider for iOS Safari / Mobile WebKit
    if (window.ort && ort.env && ort.env.wasm) {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
    }

    const modelResp = await fetch("./models/music_classifier.onnx");
    if (!modelResp.ok) throw new Error(`HTTP ${modelResp.status} загрузки модели`);
    const modelBuf = await modelResp.arrayBuffer();

    try {
      ortSession = await ort.InferenceSession.create(new Uint8Array(modelBuf), {
        executionProviders: ["wasm"]
      });
    } catch (simdErr) {
      console.warn("WASM SIMD failed on this device, trying fallback...", simdErr);
      if (window.ort && ort.env && ort.env.wasm) {
        ort.env.wasm.simd = false;
      }
      ortSession = await ort.InferenceSession.create(new Uint8Array(modelBuf), {
        executionProviders: ["wasm"]
      });
    }

    statusDot.className = "status-dot ready";
    statusText.textContent = "AI готов";
  } catch (err) {
    statusDot.className = "status-dot error";
    statusText.textContent = "Офлайн / Ошибка AI";
    lastAiError = err.message || String(err);
    console.error("ONNX init error:", err);
  }

  // Wire UI events
  const fileInput = document.getElementById("fileInput");
  document.getElementById("btnAddFiles").addEventListener("click", () => fileInput.click());
  document.getElementById("btnEmptyAdd").addEventListener("click", () => fileInput.click());
  document.getElementById("btnEmptyGoDownload").addEventListener("click", () => {
    document.querySelector('[data-target="viewDownload"]').click();
  });

  fileInput.addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    const entries = files.map((f) => {
      const meta = parseFileName(f.name);
      return { title: meta.title, artist: meta.artist, blob: f };
    });
    fileInput.value = "";
    await addAudioBlobs(entries);
  });

  document.getElementById("searchInput").addEventListener("input", renderTrackList);

  document.querySelectorAll(".nav-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".view-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(tab.dataset.target).classList.add("active");
    });
  });

  document.getElementById("btnDlSearch").addEventListener("click", handleDownloadSearch);
  document.getElementById("dlQueryInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleDownloadSearch();
  });

  // Player controls
  const togglePlay = () => {
    if (!currentTrackId && tracks.length) {
      playTrackById(tracks[0].id);
      return;
    }
    if (!currentTrackId) return;
    if (audioEl.paused) {
      userPaused = false;
      setAudioSessionPlayback();
      attemptPlay(currentTrackId, ++playGeneration, 0);
    } else {
      playGeneration++;
      userPaused = true;
      audioEl.pause();
    }
  };

  document.getElementById("btnMiniPlay").addEventListener("click", togglePlay);
  document.getElementById("btnSheetPlay").addEventListener("click", togglePlay);
  document.getElementById("btnMiniNext").addEventListener("click", () => playAdjacent(1));
  document.getElementById("btnSheetNext").addEventListener("click", () => playAdjacent(1));
  document.getElementById("btnSheetPrev").addEventListener("click", () => {
    if (audioEl.currentTime > 3) {
      const now = Date.now();
      lastManualSeekAt = now;
      lastObservedTime = 0;
      playbackStartedWallMs = now;
      playbackStartAudioPos = 0;
      audioEl.currentTime = 0;
      updateMediaSessionPosition();
    } else {
      playAdjacent(-1);
    }
  });

  document.getElementById("miniOpenSheet").addEventListener("click", () => openInspectorSheet(currentTrackId));
  document.getElementById("btnCloseSheet").addEventListener("click", () => {
    document.getElementById("playerSheet").classList.remove("open");
  });

  document.getElementById("btnDeleteTrack").addEventListener("click", async () => {
    const id = inspectedTrackId || currentTrackId;
    if (!id) return;
    await dbDeleteTrack(id);
    tracks = tracks.filter((t) => t.id !== id);
    if (currentTrackId === id) {
      playGeneration++;
      userPaused = true;
      isTransitioning = false;
      clearTimeout(transitionTimer);
      audioEl.pause();
      audioEl.removeAttribute("src");
      currentTrackId = null;
      updateMiniPlayer(null);
    }
    document.getElementById("playerSheet").classList.remove("open");
    renderFilterChips();
    renderTrackList();
    updateStatsView();
  });

  document.getElementById("btnClearAll").addEventListener("click", async () => {
    if (!tracks.length) return;
    if (!confirm("Очистить все сохранённые треки из памяти устройства?")) return;
    await dbClearAll();
    tracks = [];
    playGeneration++;
    userPaused = true;
    isTransitioning = false;
    clearTimeout(transitionTimer);
    audioEl.pause();
    currentTrackId = null;
    updateMiniPlayer(null);
    renderFilterChips();
    renderTrackList();
    updateStatsView();
  });

  audioEl.addEventListener("playing", () => {
    isTransitioning = false;
    clearTimeout(transitionTimer);
    errorRecoveredTrackId = null;
    playbackStartedWallMs = Date.now();
    playbackStartAudioPos = audioEl.currentTime || 0;
    syncPlayButtons();
    if ("mediaSession" in navigator) navigator.mediaSession.playbackState = "playing";
    updateMediaSessionPosition();
    if (!document.hidden && document.getElementById("playerSheet").classList.contains("open")) {
      startVisualizerLoop();
    }
  });

  audioEl.addEventListener("play", () => {
    userPaused = false;
    setAudioSessionPlayback();
    syncPlayButtons();
    if ("mediaSession" in navigator) navigator.mediaSession.playbackState = "playing";
  });

  audioEl.addEventListener("pause", () => {
    const dur = audioEl.duration;
    const cur = audioEl.currentTime || 0;
    const nearEnd = Number.isFinite(dur) && dur > 1.5 && (audioEl.ended || (dur - cur) <= 0.7);
    if (isTransitioning || (!userPaused && nearEnd)) {
      if (!isTransitioning && nearEnd) advanceToNextTrack("pause-near-end");
      return;
    }
    syncPlayButtons();
    if ("mediaSession" in navigator) navigator.mediaSession.playbackState = "paused";
  });

  audioEl.addEventListener("loadedmetadata", () => {
    if (!document.hidden && Number.isFinite(audioEl.duration) && audioEl.duration > 0) {
      document.getElementById("seekDurText").textContent = formatTime(audioEl.duration);
    }
    updateMediaSessionPosition();
  });

  audioEl.addEventListener("ended", () => {
    advanceToNextTrack("ended");
  });

  audioEl.addEventListener("seeked", () => {
    const dur = audioEl.duration;
    const cur = audioEl.currentTime || 0;
    if (!isTransitioning && !isUserSeeking && !audioEl.paused && !userPaused && isLoopWrapDetected(cur, dur)) {
      advanceToNextTrack("seeked-loop-wrap");
      return;
    }
    lastObservedTime = cur;
    updateMediaSessionPosition();
  });

  audioEl.addEventListener("timeupdate", () => {
    const dur = audioEl.duration || 0;
    const cur = audioEl.currentTime || 0;

    if (!document.hidden && dur > 0) {
      const pct = `${Math.min(100, (cur / dur) * 100).toFixed(2)}%`;
      document.getElementById("miniProgressFill").style.width = pct;
      if (!isUserSeeking) {
        document.getElementById("seekFill").style.width = pct;
        document.getElementById("seekKnob").style.left = pct;
        document.getElementById("seekCurText").textContent = formatTime(cur);
      }
      document.getElementById("seekDurText").textContent = formatTime(dur);
    }

    if (Number.isFinite(dur) && dur > 0 && !isTransitioning && !isUserSeeking && !audioEl.paused && !userPaused) {
      // Layer 1: proactive handoff right before EOF while iOS audio session is still active
      if (dur > 1.5 && (dur - cur) <= 0.45) {
        const next = getNextTrackInPool(1);
        if (next && next.id !== currentTrackId) {
          advanceToNextTrack("timeupdate-near-end");
          return;
        }
      }

      // Layer 2: loop-wrap rescue if iOS throttled timeupdate past EOF and native loop wrapped to 0
      if (isLoopWrapDetected(cur, dur)) {
        advanceToNextTrack("timeupdate-loop-wrap");
        return;
      }

      lastObservedTime = cur;
      const now = Date.now();
      if (now - lastPosSyncAt >= 2000) {
        updateMediaSessionPosition();
      }
    }
  });

  audioEl.addEventListener("error", async () => {
    if (!currentTrackId || userPaused || errorRecoveredTrackId === currentTrackId) return;
    errorRecoveredTrackId = currentTrackId;
    const failedTrackId = currentTrackId;
    const gen = ++playGeneration;
    try {
      const fresh = await dbGetTrack(failedTrackId);
      if (!fresh || !fresh.blob || currentTrackId !== failedTrackId || userPaused || gen !== playGeneration) return;
      const idx = tracks.findIndex((t) => t.id === failedTrackId);
      if (idx >= 0) tracks[idx].blob = fresh.blob;
      const prevUrl = currentObjectUrl;
      currentObjectUrl = URL.createObjectURL(ensureTypedAudioBlob(fresh.blob, fresh.title));
      audioEl.loop = true;
      audioEl.src = currentObjectUrl;
      attemptPlay(failedTrackId, gen, 0);
      if (prevUrl) {
        setTimeout(() => {
          try { URL.revokeObjectURL(prevUrl); } catch (_) {}
        }, 3000);
      }
    } catch (recoverErr) {
      console.warn("Audio error recovery failed:", recoverErr);
    }
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    const track = tracks.find((t) => t.id === currentTrackId);
    if (track) {
      updateMiniPlayer(track);
      populateSheet(track);
    }
    updateActiveTrackRowHighlight();
    syncPlayButtons();

    const dur = audioEl.duration || 0;
    const cur = audioEl.currentTime || 0;
    if (dur > 0) {
      const pct = `${Math.min(100, (cur / dur) * 100).toFixed(2)}%`;
      document.getElementById("miniProgressFill").style.width = pct;
      if (!isUserSeeking) {
        document.getElementById("seekFill").style.width = pct;
        document.getElementById("seekKnob").style.left = pct;
        document.getElementById("seekCurText").textContent = formatTime(cur);
      }
      document.getElementById("seekDurText").textContent = formatTime(dur);
    }

    if (document.getElementById("playerSheet").classList.contains("open") && !audioEl.paused) {
      startVisualizerLoop();
    }
  });
}

function syncPlayButtons() {
  const playing = !audioEl.paused;
  document.getElementById("miniPlaySvg").style.display = playing ? "none" : "block";
  document.getElementById("miniPauseSvg").style.display = playing ? "block" : "none";
  document.getElementById("sheetPlaySvg").style.display = playing ? "none" : "block";
  document.getElementById("sheetPauseSvg").style.display = playing ? "block" : "none";
  updateActiveTrackRowHighlight();
}

document.addEventListener("DOMContentLoaded", initApp);
