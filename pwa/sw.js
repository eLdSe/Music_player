const CACHE_NAME = "soundpulse-pwa-v3";
const CORE_ASSETS = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./manifest.json",
  "./apple-touch-icon.png",
  "./icon-512.png",
  "./models/labels.json",
  "./models/music_classifier.int8.onnx"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(CORE_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  let url;
  try {
    url = new URL(req.url);
  } catch {
    return;
  }

  // Bypass service worker cache for external APIs (Audius, YouTube oembed, etc.)
  if (
    url.hostname.includes("audius.co") ||
    url.hostname.includes("noembed.com") ||
    url.hostname.includes("youtube.com") ||
    url.hostname.includes("youtu.be") ||
    url.hostname.includes("googlevideo.com") ||
    (url.origin !== self.location.origin && !url.hostname.includes("jsdelivr.net"))
  ) {
    return;
  }

  event.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req).then((resp) => {
        if (resp && resp.status === 200 && (url.origin === self.location.origin || url.hostname.includes("jsdelivr.net"))) {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(req, clone));
        }
        return resp;
      });
    }).catch((err) => {
      console.warn("SW fetch offline fallback:", err);
      return new Response("Offline or Network Error", { status: 503, statusText: "Offline" });
    })
  );
});
