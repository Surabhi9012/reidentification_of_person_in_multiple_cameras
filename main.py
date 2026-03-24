"""
Multi-Camera Person Re-Identification System
Tracks the same person across multiple RTSP cameras with consistent IDs.
"""

import cv2
import numpy as np
import threading
import time
import json
import base64
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import argparse

# ── Try importing deep learning libs ──────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed. Run: pip install ultralytics")

try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] torch not installed. Run: pip install torch torchvision")

try:
    from scipy.spatial.distance import cosine
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARN] scipy not installed. Run: pip install scipy")

try:
    from flask import Flask, Response, render_template_string, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("[WARN] flask not installed. Run: pip install flask")


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Person:
    global_id: int
    features: np.ndarray          # appearance embedding
    last_seen_cam: int
    last_seen_time: float
    bbox_history: List[Tuple]     # (cam_id, bbox, timestamp)
    thumbnail: Optional[np.ndarray] = None  # small crop for dashboard
    color: Tuple = (0, 255, 0)    # display color


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    local_id: int                     # tracker-assigned id (per-camera)
    global_id: Optional[int] = None  # re-id assigned id
    features: Optional[np.ndarray] = None


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR  (ResNet50 body appearance)
# ══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    def __init__(self):
        self.available = TORCH_AVAILABLE
        if self.available:
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model.fc = torch.nn.Identity()   # remove classifier head
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
            ])
            print(f"[INFO] Feature extractor ready on {self.device}")
        else:
            print("[WARN] Using histogram fallback for features (install torch for better accuracy)")

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract a 2048-d embedding from a person crop (BGR image)."""
        if crop is None or crop.size == 0:
            return np.zeros(2048 if self.available else 512)

        if self.available:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(tensor).cpu().numpy().flatten()
            norm = np.linalg.norm(feat)
            return feat / (norm + 1e-6)
        else:
            return self._histogram_feature(crop)

    def _histogram_feature(self, crop: np.ndarray) -> np.ndarray:
        """Fallback: color histogram feature (no torch needed)."""
        crop_resized = cv2.resize(crop, (64, 128))
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
        feat = []
        for ch, bins in enumerate([180, 256, 256]):
            hist = cv2.calcHist([hsv], [ch], None, [bins], [0, bins])
            feat.extend(hist.flatten())
        feat = np.array(feat, dtype=np.float32)
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL RE-ID GALLERY
# ══════════════════════════════════════════════════════════════════════════════

class ReIDGallery:
    def __init__(self, similarity_threshold: float = 0.75, max_age: float = 60.0):
        self.persons: Dict[int, Person] = {}
        self.next_id = 1
        self.threshold = similarity_threshold   # cosine similarity threshold
        self.max_age = max_age                  # seconds before a person is forgotten
        self.lock = threading.Lock()
        self.colors = self._generate_colors(100)
        print(f"[INFO] ReID gallery ready. Threshold={threshold}")

    def _generate_colors(self, n: int) -> List[Tuple]:
        np.random.seed(42)
        colors = []
        for i in range(n):
            hue = int(i * 180 / n)
            hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
        return colors

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if SCIPY_AVAILABLE:
            return 1.0 - cosine(a, b)
        # manual cosine
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        return float(np.dot(a, b) / denom)

    def match_or_register(
        self,
        features: np.ndarray,
        cam_id: int,
        bbox: Tuple,
        thumbnail: Optional[np.ndarray] = None
    ) -> int:
        """
        Given appearance features from a camera, find the best matching
        global Person ID or assign a new one.
        Returns: global_id (int)
        """
        now = time.time()
        with self.lock:
            # Prune stale persons
            stale = [pid for pid, p in self.persons.items()
                     if now - p.last_seen_time > self.max_age]
            for pid in stale:
                del self.persons[pid]

            best_id = None
            best_sim = -1.0

            for pid, person in self.persons.items():
                sim = self._similarity(features, person.features)
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid

            if best_sim >= self.threshold and best_id is not None:
                # Update existing person
                p = self.persons[best_id]
                # Running average of features for robustness
                p.features = 0.7 * p.features + 0.3 * features
                p.last_seen_cam = cam_id
                p.last_seen_time = now
                p.bbox_history.append((cam_id, bbox, now))
                if thumbnail is not None:
                    p.thumbnail = thumbnail
                return best_id
            else:
                # New person
                gid = self.next_id
                self.next_id += 1
                color = self.colors[gid % len(self.colors)]
                self.persons[gid] = Person(
                    global_id=gid,
                    features=features.copy(),
                    last_seen_cam=cam_id,
                    last_seen_time=now,
                    bbox_history=[(cam_id, bbox, now)],
                    thumbnail=thumbnail,
                    color=color,
                )
                return gid

    def get_snapshot(self) -> dict:
        """Return serialisable gallery state for the dashboard."""
        now = time.time()
        with self.lock:
            out = {}
            for gid, p in self.persons.items():
                out[gid] = {
                    "global_id": gid,
                    "last_seen_cam": p.last_seen_cam,
                    "last_seen_ago": round(now - p.last_seen_time, 1),
                    "color": p.color,
                    "seen_on_cams": list({h[0] for h in p.bbox_history[-50:]}),
                    "has_thumbnail": p.thumbnail is not None,
                }
            return out

    def get_thumbnail_b64(self, gid: int) -> Optional[str]:
        with self.lock:
            p = self.persons.get(gid)
            if p is None or p.thumbnail is None:
                return None
            _, buf = cv2.imencode(".jpg", p.thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return base64.b64encode(buf).decode()


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE CAMERA WORKER
# ══════════════════════════════════════════════════════════════════════════════

class CameraWorker(threading.Thread):
    def __init__(
        self,
        cam_id: int,
        rtsp_url: str,
        gallery: ReIDGallery,
        extractor: FeatureExtractor,
        detector,                      # YOLO model or None
        process_every: int = 5,        # run detection every N frames
        display_scale: float = 0.5,
    ):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.gallery = gallery
        self.extractor = extractor
        self.detector = detector
        self.process_every = process_every
        self.display_scale = display_scale

        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.active_tracks: Dict[int, Detection] = {}
        self.running = True
        self.frame_count = 0
        self.fps = 0.0
        self._fps_time = time.time()
        self._fps_frames = 0

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print(f"[ERROR] Camera {self.cam_id}: Cannot open {self.rtsp_url}")
            return

        # Use MJPEG for speed on RTSP
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[INFO] Camera {self.cam_id} connected: {self.rtsp_url}")

        tracker = None
        if YOLO_AVAILABLE and self.detector:
            pass   # YOLO handles its own ByteTrack

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] Camera {self.cam_id}: Frame read failed, retrying...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(self.rtsp_url)
                continue

            self.frame_count += 1
            self._fps_frames += 1
            now = time.time()
            if now - self._fps_time >= 2.0:
                self.fps = self._fps_frames / (now - self._fps_time)
                self._fps_frames = 0
                self._fps_time = now

            # ── Detect & Re-ID every N frames ─────────────────────────────
            if self.frame_count % self.process_every == 0:
                detections = self._detect(frame)
                self.active_tracks = {}
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size == 0:
                        continue
                    feat = self.extractor.extract(crop)
                    det.features = feat
                    thumb = cv2.resize(crop, (64, 128)) if crop.size > 0 else None
                    gid = self.gallery.match_or_register(feat, self.cam_id, det.bbox, thumb)
                    det.global_id = gid
                    self.active_tracks[gid] = det

            # ── Draw overlays ──────────────────────────────────────────────
            annotated = self._draw(frame.copy())

            with self.frame_lock:
                self.latest_frame = annotated

        cap.release()

    def _detect(self, frame: np.ndarray) -> List[Detection]:
        detections = []
        if YOLO_AVAILABLE and self.detector:
            results = self.detector.track(
                frame,
                persist=True,
                classes=[0],         # class 0 = person
                conf=0.35,
                iou=0.45,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    tid = int(box.id[0]) if box.id is not None else i
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        local_id=tid,
                    ))
        else:
            # Fallback: OpenCV HOG person detector
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (min(w, 640), min(h, 480)))
            scale = w / small.shape[1]
            rects, weights = hog.detectMultiScale(
                small, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            for i, (x, y, bw, bh) in enumerate(rects):
                x1, y1 = int(x * scale), int(y * scale)
                x2, y2 = int((x + bw) * scale), int((y + bh) * scale)
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(weights[i]) if len(weights) > i else 0.5,
                    local_id=i,
                ))
        return detections

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        for gid, det in self.active_tracks.items():
            x1, y1, x2, y2 = det.bbox
            color = self.gallery.persons[gid].color if gid in self.gallery.persons else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{gid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # HUD
        cv2.putText(frame, f"CAM {self.cam_id}  {self.fps:.1f}fps",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
        return frame

    def get_jpeg(self) -> Optional[bytes]:
        with self.frame_lock:
            f = self.latest_frame
        if f is None:
            return None
        _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return bytes(buf)

    def stop(self):
        self.running = False


# ══════════════════════════════════════════════════════════════════════════════
# FLASK WEB DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-Cam ReID · Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@400;700;900&display=swap');
  :root {
    --bg: #0a0c10; --panel: #111520; --border: #1e2d45;
    --accent: #00d4ff; --green: #00ff88; --red: #ff4060;
    --text: #c8d8e8; --dim: #4a6070;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Exo 2', sans-serif; min-height: 100vh; }

  header {
    display: flex; align-items: center; gap: 16px;
    padding: 16px 28px; border-bottom: 1px solid var(--border);
    background: linear-gradient(90deg, #0d1420 0%, #0a0c10 100%);
  }
  .logo { font-size: 22px; font-weight: 900; letter-spacing: 2px; color: var(--accent);
    text-shadow: 0 0 20px rgba(0,212,255,.4); }
  .sub { font-size: 11px; color: var(--dim); letter-spacing: 4px; text-transform: uppercase; margin-top: 2px; }
  .pill { margin-left: auto; padding: 4px 14px; border-radius: 20px;
    background: rgba(0,255,136,.1); border: 1px solid var(--green);
    color: var(--green); font-size: 12px; font-family: 'Share Tech Mono', monospace; }

  .main { display: grid; grid-template-columns: 1fr 300px; height: calc(100vh - 65px); }

  /* Camera grid */
  .cam-grid { padding: 16px; display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 12px; overflow-y: auto; }
  .cam-card { background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden; position: relative; }
  .cam-card img { width: 100%; display: block; aspect-ratio: 16/9; object-fit: cover; background: #000; }
  .cam-label { position: absolute; top: 8px; left: 8px;
    background: rgba(0,0,0,.65); padding: 3px 10px; border-radius: 4px;
    font-family: 'Share Tech Mono', monospace; font-size: 12px; color: var(--accent); }

  /* Sidebar */
  .sidebar { border-left: 1px solid var(--border); background: var(--panel);
    display: flex; flex-direction: column; overflow: hidden; }
  .sidebar-hdr { padding: 14px 16px; border-bottom: 1px solid var(--border);
    font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: var(--dim); }
  .gallery { flex: 1; overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 8px; }
  .person-card {
    background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
    padding: 10px; display: grid; grid-template-columns: 64px 1fr; gap: 10px;
    align-items: center; transition: border-color .2s;
  }
  .person-card:hover { border-color: var(--accent); }
  .person-card img { width: 64px; height: 96px; object-fit: cover; border-radius: 4px; background: #1a2030; }
  .person-card .pid { font-size: 20px; font-weight: 900; color: var(--accent); font-family: 'Share Tech Mono', monospace; }
  .person-card .meta { font-size: 11px; color: var(--dim); margin-top: 4px; }
  .cam-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 4px; background: var(--green); }
  .cam-badges { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
  .cam-badge { padding: 2px 7px; border-radius: 12px; font-size: 10px;
    background: rgba(0,212,255,.1); border: 1px solid var(--border);
    font-family: 'Share Tech Mono', monospace; }
  .stats-bar { padding: 10px 16px; border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; font-size: 12px; color: var(--dim); }
  .stats-bar span { color: var(--accent); }
  .no-person { text-align: center; color: var(--dim); font-size: 13px; margin-top: 40px; }
</style>
</head>
<body>
<header>
  <div>
    <div class="logo">⬡ MULTICAM Re-ID</div>
    <div class="sub">Cross-Camera Person Identification</div>
  </div>
  <div class="pill" id="status">● LIVE</div>
</header>
<div class="main">
  <div class="cam-grid" id="camGrid"></div>
  <div class="sidebar">
    <div class="sidebar-hdr">Tracked Persons</div>
    <div class="gallery" id="gallery"><div class="no-person">Waiting for detections…</div></div>
    <div class="stats-bar">
      Persons: <span id="pCount">0</span> &nbsp;|&nbsp; Cams: <span id="cCount">{{ num_cams }}</span>
    </div>
  </div>
</div>

<script>
const NUM_CAMS = {{ num_cams }};
let initialized = false;

// Build camera cards
function initCams() {
  const grid = document.getElementById('camGrid');
  for (let i = 0; i < NUM_CAMS; i++) {
    const card = document.createElement('div');
    card.className = 'cam-card';
    card.innerHTML = `
      <div class="cam-label">CAM ${i}</div>
      <img id="cam${i}" src="/stream/${i}" alt="Camera ${i}" onerror="this.src='/static/offline.png'">
    `;
    grid.appendChild(card);
  }
}

// Poll gallery state
async function refreshGallery() {
  try {
    const r = await fetch('/api/gallery');
    const data = await r.json();
    const gallery = document.getElementById('gallery');
    const ids = Object.keys(data);
    document.getElementById('pCount').textContent = ids.length;

    if (ids.length === 0) {
      gallery.innerHTML = '<div class="no-person">No persons detected yet…</div>';
      return;
    }

    // Update or create cards
    for (const gid of ids) {
      const p = data[gid];
      let card = document.getElementById(`person_${gid}`);
      if (!card) {
        card = document.createElement('div');
        card.id = `person_${gid}`;
        card.className = 'person-card';
        gallery.prepend(card);
      }
      const color = `rgb(${p.color[2]},${p.color[1]},${p.color[0]})`;
      const badges = p.seen_on_cams.map(c => `<span class="cam-badge">CAM ${c}</span>`).join('');
      card.style.borderLeftColor = color;
      card.style.borderLeftWidth = '3px';
      card.innerHTML = `
        <img src="/api/thumbnail/${gid}?t=${Date.now()}" alt="Person ${gid}">
        <div>
          <div class="pid" style="color:${color}">ID ${gid}</div>
          <div class="meta"><span class="cam-dot"></span>Cam ${p.last_seen_cam} · ${p.last_seen_ago}s ago</div>
          <div class="cam-badges">${badges}</div>
        </div>
      `;
    }

    // Remove stale cards
    const current = new Set(ids.map(id => `person_${id}`));
    gallery.querySelectorAll('.person-card').forEach(el => {
      if (!current.has(el.id)) el.remove();
    });
  } catch(e) { console.error(e); }
}

initCams();
setInterval(refreshGallery, 2000);
refreshGallery();
</script>
</body>
</html>
"""


def create_app(workers: List[CameraWorker], gallery: ReIDGallery) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML, num_cams=len(workers))

    def _gen_stream(worker: CameraWorker):
        while True:
            jpg = worker.get_jpeg()
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.04)   # ~25 fps cap

    @app.route("/stream/<int:cam_id>")
    def stream(cam_id: int):
        if cam_id >= len(workers):
            return "Not found", 404
        return Response(
            _gen_stream(workers[cam_id]),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/gallery")
    def api_gallery():
        return jsonify(gallery.get_snapshot())

    @app.route("/api/thumbnail/<int:gid>")
    def api_thumbnail(gid: int):
        b64 = gallery.get_thumbnail_b64(gid)
        if b64 is None:
            return "", 204
        data = base64.b64decode(b64)
        return Response(data, mimetype="image/jpeg")

    return app


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Person Re-ID System")
    parser.add_argument("--cameras", nargs="+", required=True,
                        help="RTSP URLs (space-separated). Use 0,1,2 for webcams.")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Similarity threshold for Re-ID (0-1). Default=0.75")
    parser.add_argument("--max-age", type=float, default=60.0,
                        help="Seconds before a person is forgotten. Default=60")
    parser.add_argument("--process-every", type=int, default=5,
                        help="Run detection every N frames. Default=5")
    parser.add_argument("--port", type=int, default=5000,
                        help="Dashboard port. Default=5000")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable web dashboard (show windows instead)")
    args = parser.parse_args()

    # Fix threshold reference in gallery
    global threshold
    threshold = args.threshold

    print("=" * 60)
    print("  Multi-Camera Person Re-ID System")
    print(f"  Cameras  : {len(args.cameras)}")
    print(f"  Threshold: {args.threshold}")
    print(f"  YOLO     : {'✓' if YOLO_AVAILABLE else '✗ (fallback to HOG)'}")
    print(f"  Torch    : {'✓' if TORCH_AVAILABLE else '✗ (fallback to histogram)'}")
    print("=" * 60)

    # Shared components
    gallery = ReIDGallery(similarity_threshold=args.threshold, max_age=args.max_age)
    extractor = FeatureExtractor()

    # Optionally load YOLO
    detector = None
    if YOLO_AVAILABLE:
        try:
            detector = YOLO("yolov8n.pt")   # nano model – fast
            print("[INFO] YOLOv8n loaded")
        except Exception as e:
            print(f"[WARN] YOLO load failed: {e}. Falling back to HOG.")

    # Parse camera sources
    cam_urls = []
    for src in args.cameras:
        try:
            cam_urls.append(int(src))    # webcam index
        except ValueError:
            cam_urls.append(src)         # RTSP URL

    # Start camera workers
    workers: List[CameraWorker] = []
    for cam_id, url in enumerate(cam_urls):
        # Each camera gets its own YOLO instance to avoid thread conflicts
        cam_detector = YOLO("yolov8n.pt") if (YOLO_AVAILABLE and detector) else None
        w = CameraWorker(
            cam_id=cam_id,
            rtsp_url=str(url),
            gallery=gallery,
            extractor=extractor,
            detector=cam_detector,
            process_every=args.process_every,
        )
        w.start()
        workers.append(w)
        time.sleep(0.5)   # stagger starts

    if args.no_web:
        # OpenCV window mode
        print("[INFO] Press Q in any window to quit.")
        try:
            while True:
                for w in workers:
                    jpg = w.get_jpeg()
                    if jpg:
                        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        cv2.imshow(f"Camera {w.cam_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.03)
        finally:
            for w in workers:
                w.stop()
            cv2.destroyAllWindows()
    else:
        if not FLASK_AVAILABLE:
            print("[ERROR] Flask not installed. Run: pip install flask")
            return
        app = create_app(workers, gallery)
        print(f"\n[INFO] Dashboard → http://localhost:{args.port}\n")
        app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()