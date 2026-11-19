const CACHE_NAME = "soundpulse-pwa-v10";
const CORE_ASSETS = [
  "./",
  "./index.html",
  "./index.html?v=10",
  "./styles.css?v=10",
  "./app.js?v=10",
  "./manifest.json",
  "./apple-touch-icon.png",
  "./icon-512.png",
  "./models/labels.json",
  "./models/music_classifier.onnx"
];

self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(CORE_ASSETS))
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((k) => {
        if (k !== CACHE_NAME) return caches.delete(k);
      }))
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

  // Never intercept blob:, data:, chrome-extension:, or non-HTTP(S) URLs
  if (url.protocol !== "http:" && url.protocol !== "https:") {
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

  // Network-first for HTML, JS and CSS to ensure instant updates on mobile!
  if (req.mode === "navigate" || url.pathname.endsWith(".html") || url.pathname.endsWith(".js") || url.pathname.endsWith(".css")) {
    event.respondWith(
      fetch(req).then((resp) => {
        if (resp && resp.status === 200) {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(req, clone));
        }
        return resp;
      }).catch(() => caches.match(req))
    );
    return;
  }

  // Cache-first for large static assets (ONNX model, icons, sounds)
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
    }).catch(() => new Response("Offline", { status: 503, statusText: "Offline" }))
  );
});
