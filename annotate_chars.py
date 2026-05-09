"""
License Plate Annotation Tool — with Deskew + Character Segmentation
Deskews each plate, detects character bounding boxes, lets you annotate
each character interactively, and saves everything for CNN training.

Output structure:
    output_dir/
        deskewed/          ← deskewed plate images
        annotations.csv    ← filename, plate_text, char, x, y, w, h

Usage:
    pip install flask opencv-python numpy
    python annotate_chars.py --input ./plates --output ./output

Keyboard shortcuts (browser):
    Enter      → confirm current character label & move to next box
    Backspace  → go back one box
    Delete     → remove current bounding box
    n          → skip entire plate (no annotation)
    ← →        → previous / next plate
"""

import csv
import json
import argparse
import re
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, render_template_string, request, jsonify, send_file

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = Flask(__name__)

state = {
    "images":      [],   # list of original image Paths
    "index":       0,    # current plate index
    "annotations": {},   # fname -> {"plate": str, "chars": [{char,x,y,w,h}]}
    "output_dir":  "",
}

# ──────────────────────────────────────────────────────────────────────────────
# Deskew  (same two-method approach as before)
# ──────────────────────────────────────────────────────────────────────────────

def detect_angle(gray: np.ndarray, max_angle: float = 15.0) -> float:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines   = cv2.HoughLinesP(edges, 1, np.pi / 180,
                               threshold=60,
                               minLineLength=gray.shape[1] // 4,
                               maxLineGap=20)
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(a) <= max_angle:
                    angles.append(a)
        if angles:
            return float(np.median(angles))

    _, thresh   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        rect    = cv2.minAreaRect(largest)
        angle   = rect[-1]
        if angle < -45:
            angle += 90
        if abs(angle) <= max_angle:
            return float(angle)
    return 0.0


def deskew(img: np.ndarray, max_angle: float = 15.0) -> tuple[np.ndarray, float]:
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = detect_angle(gray, max_angle)
    if abs(angle) < 0.5:
        return img, 0.0
    h, w   = img.shape[:2]
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w  = int(h * sin + w * cos)
    new_h  = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


# ──────────────────────────────────────────────────────────────────────────────
# Character segmentation
# ──────────────────────────────────────────────────────────────────────────────

def _is_dark_on_light(gray: np.ndarray) -> bool:
    """Return True if characters are dark on a light background (e.g. white plate)."""
    return int(gray.mean()) > 127


def _extract_boxes(thresh: np.ndarray, img_h: int, img_w: int) -> list[dict]:
    """Extract character-like bounding boxes from a binary threshold image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        rel_h  = bh / img_h
        if (0.1 < aspect < 1.2
                and 0.3 < rel_h < 0.95
                and bw > 5 and bh > 8):
            boxes.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "char": ""})
    return boxes


def segment_chars(img: np.ndarray) -> list[dict]:
    """
    Returns list of {x, y, w, h} bounding boxes for each character candidate,
    sorted left-to-right. Handles both dark-on-light and light-on-dark plates
    by trying multiple thresholding strategies and picking the best result.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale if too small
    img_h, img_w = gray.shape
    if img_w < 300:
        scale = 300 / img_w
        gray  = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        img_h, img_w = gray.shape

    # Denoise slightly
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    dark_on_light = _is_dark_on_light(gray)

    candidates = []

    # Strategy 1: Adaptive threshold, correct polarity for this plate type
    polarity = cv2.THRESH_BINARY if dark_on_light else cv2.THRESH_BINARY_INV
    for block in (11, 15, 21):
        for C in (5, 8, 12):
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, polarity, block, C)
            candidates.append(_extract_boxes(thresh, img_h, img_w))

    # Strategy 2: Otsu global threshold, both polarities
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(_extract_boxes(otsu, img_h, img_w))
    candidates.append(_extract_boxes(255 - otsu, img_h, img_w))

    # Strategy 3: CLAHE + Otsu (helps uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    _, clahe_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if dark_on_light:
        candidates.append(_extract_boxes(clahe_thresh, img_h, img_w))
    else:
        candidates.append(_extract_boxes(255 - clahe_thresh, img_h, img_w))

    # Pick best: most boxes with consistent heights (implausibly many = noise)
    def score(boxes):
        n = len(boxes)
        if n == 0 or n > 14:
            return -1
        heights = [b["h"] for b in boxes]
        std = np.std(heights) if len(heights) > 1 else 0
        mean_h = np.mean(heights)
        consistency = 1.0 - min(std / mean_h, 1.0) if mean_h > 0 else 0
        return n * consistency

    best = max(candidates, key=score)
    best.sort(key=lambda b: b["x"])
    best = merge_close_boxes(best, gap_threshold=img_w // 30)
    return best


def merge_close_boxes(boxes: list[dict], gap_threshold: int = 8) -> list[dict]:
    """Merge horizontally overlapping or nearly-touching boxes."""
    if not boxes:
        return boxes
    merged = [boxes[0].copy()]
    for b in boxes[1:]:
        prev = merged[-1]
        prev_right = prev["x"] + prev["w"]
        if b["x"] - prev_right < gap_threshold:
            # Merge
            new_x = min(prev["x"], b["x"])
            new_y = min(prev["y"], b["y"])
            new_r = max(prev_right, b["x"] + b["w"])
            new_b = max(prev["y"] + prev["h"], b["y"] + b["h"])
            prev.update({"x": new_x, "y": new_y, "w": new_r - new_x, "h": new_b - new_y})
        else:
            merged.append(b.copy())
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline: load image → deskew → segment → save deskewed
# ──────────────────────────────────────────────────────────────────────────────

def process_image(img_path: Path) -> tuple[np.ndarray, float, list[dict]]:
    img             = cv2.imread(str(img_path))
    deskewed, angle = deskew(img)
    boxes           = segment_chars(deskewed)
    return deskewed, angle, boxes


def save_deskewed(img: np.ndarray, img_path: Path) -> str:
    out_dir = Path(state["output_dir"]) / "deskewed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), img)
    return str(out_path)


# ──────────────────────────────────────────────────────────────────────────────
# CSV persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_csv():
    out_path = Path(state["output_dir"]) / "annotations.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "plate", "char", "x", "y", "w", "h"])
        writer.writeheader()
        for fname, ann in state["annotations"].items():
            plate = ann.get("plate", "")
            chars = ann.get("chars", [])
            if not chars:
                writer.writerow({"filename": fname, "plate": plate,
                                 "char": "", "x": "", "y": "", "w": "", "h": ""})
            for ch in chars:
                writer.writerow({"filename": fname, "plate": plate,
                                 "char": ch.get("char",""), "x": ch["x"],
                                 "y": ch["y"], "w": ch["w"], "h": ch["h"]})


def load_csv_if_exists():
    out_path = Path(state["output_dir"]) / "annotations.csv"
    if not out_path.exists():
        return {}
    existing = {}
    with open(out_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            if fname not in existing:
                existing[fname] = {"plate": row["plate"], "chars": []}
            if row["char"] or row["x"]:
                try:
                    existing[fname]["chars"].append({
                        "char": row["char"],
                        "x": int(row["x"]), "y": int(row["y"]),
                        "w": int(row["w"]), "h": int(row["h"]),
                    })
                except (ValueError, KeyError):
                    pass
    return existing


# ──────────────────────────────────────────────────────────────────────────────
# Flask routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML.replace("{{ start_index }}", str(state.get("start_index", 0))))


@app.route("/plate/<int:idx>")
def plate_data(idx: int):
    images = state["images"]
    if idx >= len(images):
        return jsonify({"done": True})

    img_path = images[idx]
    fname    = img_path.name

    # Use cached annotation if available, otherwise process fresh
    if fname in state["annotations"]:
        ann = state["annotations"][fname]
        # Load the already-saved deskewed image
        deskewed_path = str(Path(state["output_dir"]) / "deskewed" / fname)
        deskewed_img  = cv2.imread(deskewed_path)
        if deskewed_img is None:
            deskewed_img, _, _ = process_image(img_path)
        ih, iw = deskewed_img.shape[:2]
    else:
        deskewed_img, angle, boxes = process_image(img_path)
        save_deskewed(deskewed_img, img_path)
        ih, iw = deskewed_img.shape[:2]
        ann = {"plate": "", "chars": boxes, "angle": round(angle, 2)}
        state["annotations"][fname] = ann

    return jsonify({
        "done":      False,
        "index":     idx,
        "total":     len(images),
        "annotated": len(state["annotations"]),
        "filename":  fname,
        "plate":     ann.get("plate", ""),
        "boxes":     ann.get("chars", []),
        "img_w":     iw,
        "img_h":     ih,
    })


@app.route("/image/<int:idx>")
def serve_image(idx: int):
    img_path     = state["images"][idx]
    deskewed_dir = Path(state["output_dir"]) / "deskewed"
    deskewed_path = deskewed_dir / img_path.name
    if deskewed_path.exists():
        return send_file(str(deskewed_path), mimetype="image/jpeg")
    return send_file(str(img_path), mimetype="image/jpeg")


@app.route("/save", methods=["POST"])
def save_annotation():
    data  = request.json
    idx   = data["index"]
    fname = state["images"][idx].name
    state["annotations"][fname] = {
        "plate": data["plate"].strip().upper(),
        "chars": data["boxes"],
    }
    save_csv()
    return jsonify({"ok": True})


@app.route("/navigate", methods=["POST"])
def navigate():
    data          = request.json
    state["index"] = max(0, min(data["index"], len(state["images"]) - 1))
    return jsonify({"ok": True, "index": state["index"]})


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Plate Char Annotator</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0c0c0c;--surface:#141414;--border:#252525;
  --accent:#e8f400;--accent2:#ff4d00;--green:#2ecc71;
  --text:#f0f0f0;--muted:#555;
  --mono:'JetBrains Mono',monospace;--display:'Syne',sans-serif;
}
body{background:var(--bg);color:var(--text);font-family:var(--mono);
     min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:20px}
.shell{width:100%;max-width:900px;display:flex;flex-direction:column;gap:16px}

/* Header */
.header{display:flex;justify-content:space-between;align-items:center}
.title{font-family:var(--display);font-size:1rem;font-weight:800;
       letter-spacing:.15em;text-transform:uppercase;color:var(--accent)}
.counter{font-size:.7rem;color:var(--muted);letter-spacing:.1em}
.progress-track{height:2px;background:var(--border);border-radius:2px;overflow:hidden}
.progress-fill{height:100%;background:var(--accent);transition:width .3s}

/* Canvas card */
.card{background:var(--surface);border:1px solid var(--border);border-radius:4px;overflow:hidden}
.canvas-wrap{
  position:relative;display:flex;align-items:center;justify-content:center;
  padding:24px;min-height:160px;
  background:repeating-linear-gradient(45deg,#0e0e0e,#0e0e0e 10px,#0c0c0c 10px,#0c0c0c 20px);
  cursor:crosshair;
}
canvas{display:block;border:1px solid var(--border);image-rendering:pixelated}
.fname{padding:7px 14px;font-size:.6rem;color:var(--muted);border-top:1px solid var(--border);
       white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* Controls row */
.controls{display:flex;gap:10px;align-items:stretch}
.plate-input-wrap{flex:1;display:flex;flex-direction:column;gap:6px}
label{font-size:.6rem;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}
input[type=text]{
  background:var(--surface);border:2px solid var(--border);color:var(--text);
  font-family:var(--mono);font-size:1.4rem;font-weight:700;letter-spacing:.2em;
  text-transform:uppercase;padding:10px 16px;border-radius:4px;outline:none;
  transition:border-color .15s;width:100%
}
input[type=text]:focus{border-color:var(--accent)}

/* Box list */
.box-list{display:flex;flex-wrap:wrap;gap:8px}
.box-item{
  display:flex;align-items:center;gap:6px;
  background:var(--surface);border:1px solid var(--border);
  border-radius:4px;padding:4px 8px;cursor:pointer;transition:all .15s
}
.box-item.active{border-color:var(--accent);background:#1a1a00}
.box-item.labeled{border-color:var(--green)}
.box-thumb{width:28px;height:28px;object-fit:contain;background:#000;border-radius:2px}
.box-char{
  width:28px;height:28px;background:var(--bg);border:1px solid var(--border);
  border-radius:3px;font-size:1rem;font-weight:700;text-align:center;
  color:var(--text);font-family:var(--mono);outline:none;text-transform:uppercase;
  display:flex;align-items:center;justify-content:center
}
.box-char input{
  width:100%;height:100%;background:transparent;border:none;outline:none;
  font:inherit;color:inherit;text-align:center;text-transform:uppercase;
  letter-spacing:0;font-size:1rem;padding:0
}
.box-del{
  background:transparent;border:none;color:var(--muted);cursor:pointer;
  font-size:.8rem;padding:0 2px;line-height:1;transition:color .15s
}
.box-del:hover{color:var(--accent2)}

/* Buttons */
.btn{
  background:transparent;border:2px solid var(--border);color:var(--muted);
  font-family:var(--mono);font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;
  padding:10px 18px;border-radius:4px;cursor:pointer;transition:all .15s;white-space:nowrap
}
.btn:hover{border-color:var(--text);color:var(--text)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#000;font-weight:700}
.btn.primary:hover{background:#fff;border-color:#fff}
.btn.danger{border-color:var(--accent2);color:var(--accent2)}
.btn.danger:hover{background:var(--accent2);color:#fff}

.hints{display:flex;gap:16px;font-size:.58rem;color:var(--muted);flex-wrap:wrap}
.hints kbd{background:var(--border);color:var(--text);padding:1px 5px;border-radius:3px}

.flash{
  position:fixed;top:18px;right:18px;background:var(--green);color:#000;
  font-family:var(--mono);font-size:.7rem;font-weight:700;letter-spacing:.1em;
  padding:7px 14px;border-radius:4px;opacity:0;transform:translateY(-8px);
  transition:all .2s;pointer-events:none
}
.flash.show{opacity:1;transform:translateY(0)}

/* Drawing hint */
.draw-hint{font-size:.6rem;color:var(--muted);letter-spacing:.08em;
           padding:4px 0;text-align:center}
</style>
</head>
<body>
<div class="shell">
  <div class="header">
    <span class="title">Plate Char Annotator</span>
    <span class="counter" id="counter">— / —</span>
  </div>
  <div class="progress-track"><div class="progress-fill" id="progress" style="width:0%"></div></div>

  <div class="card">
    <div class="canvas-wrap" id="canvas-wrap">
      <canvas id="plate-canvas"></canvas>
    </div>
    <div class="fname" id="fname"></div>
  </div>

  <div class="draw-hint">Click &amp; drag on the image to add a missing bounding box</div>

  <div class="controls">
    <div class="plate-input-wrap">
      <label>Full plate text</label>
      <input type="text" id="plate-input" placeholder="FULL PLATE…" autocomplete="off" spellcheck="false">
    </div>
    <button class="btn" onclick="navigate(-1)">← Prev</button>
    <button class="btn danger" onclick="skipPlate()">Skip</button>
    <button class="btn primary" onclick="savePlate()">Save ↵</button>
    <button class="btn" onclick="navigate(1)">Next →</button>
  </div>

  <div>
    <label style="font-size:.6rem;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;display:block;margin-bottom:8px">
      Character boxes — click a box to highlight, type its label
    </label>
    <div class="box-list" id="box-list"></div>
  </div>

  <div class="hints">
    <span><kbd>Enter</kbd> save &amp; next</span>
    <span><kbd>←</kbd><kbd>→</kbd> prev / next plate</span>
    <span><kbd>Del</kbd> remove selected box</span>
    <span><kbd>n</kbd> skip plate</span>
    <span>Drag on image to add box</span>
  </div>
</div>
<div class="flash" id="flash"></div>

<script>
let current = {};
let boxes   = [];
let activeBox = -1;

// ── Drawing state ──────────────────────────────────────────────────
let drawing = false, dragStart = {x:0,y:0};
let displayScale = 1;  // canvas CSS pixels per image pixel

const canvas = document.getElementById('plate-canvas');
const ctx    = canvas.getContext('2d');
let plateImg = new Image();

// ── Load plate ─────────────────────────────────────────────────────
async function loadPlate(idx) {
  const r    = await fetch('/plate/' + idx);
  current    = await r.json();
  if (current.done) { alert('All plates done!'); return; }

  boxes      = current.boxes.map(b => ({...b}));
  activeBox  = -1;

  document.getElementById('counter').textContent =
    (current.index+1) + ' / ' + current.total + '  (' + current.annotated + ' annotated)';
  document.getElementById('progress').style.width =
    (current.annotated / current.total * 100) + '%';
  document.getElementById('fname').textContent = current.filename;
  document.getElementById('plate-input').value = current.plate || '';

  plateImg = new Image();
  plateImg.onload = () => { renderCanvas(); renderBoxList(); };
  plateImg.src = '/image/' + current.index + '?t=' + Date.now();
}

// ── Canvas rendering ───────────────────────────────────────────────
function renderCanvas() {
  const wrap   = document.getElementById('canvas-wrap');
  const maxW   = wrap.clientWidth  - 48;
  const maxH   = 240;
  const scaleW = maxW / current.img_w;
  const scaleH = maxH / current.img_h;
  displayScale = Math.min(scaleW, scaleH, 4);  // cap at 4× for tiny plates

  canvas.width  = Math.round(current.img_w * displayScale);
  canvas.height = Math.round(current.img_h * displayScale);

  ctx.drawImage(plateImg, 0, 0, canvas.width, canvas.height);
  drawBoxes();
}

function drawBoxes() {
  ctx.drawImage(plateImg, 0, 0, canvas.width, canvas.height);
  boxes.forEach((b, i) => {
    const x = b.x * displayScale, y = b.y * displayScale;
    const w = b.w * displayScale, h = b.h * displayScale;
    ctx.strokeStyle = i === activeBox ? '#e8f400'
                    : b.char          ? '#2ecc71'
                    :                   '#ff4d00';
    ctx.lineWidth   = i === activeBox ? 2.5 : 1.5;
    ctx.strokeRect(x, y, w, h);

    if (b.char) {
      ctx.fillStyle = i === activeBox ? '#e8f400' : '#2ecc71';
      ctx.font      = `bold ${Math.max(10, h * 0.4)}px JetBrains Mono`;
      ctx.fillText(b.char, x + 2, y + Math.max(10, h * 0.4));
    }
  });
}

// ── Box list UI ────────────────────────────────────────────────────
function renderBoxList() {
  const list = document.getElementById('box-list');
  list.innerHTML = '';
  boxes.forEach((b, i) => {
    const item = document.createElement('div');
    item.className = 'box-item' + (i === activeBox ? ' active' : '') + (b.char ? ' labeled' : '');
    item.onclick = () => setActive(i);

    // Tiny crop thumbnail
    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width  = 28;
    thumbCanvas.height = 28;
    const tc = thumbCanvas.getContext('2d');
    tc.drawImage(plateImg,
      b.x, b.y, b.w, b.h,
      0, 0, 28, 28);
    thumbCanvas.className = 'box-thumb';

    // Char input
    const charWrap  = document.createElement('div');
    charWrap.className = 'box-char';
    const charInput = document.createElement('input');
    charInput.maxLength = 1;
    charInput.value     = b.char || '';
    charInput.dataset.idx = i;
    charInput.addEventListener('input', e => {
      boxes[i].char = e.target.value.toUpperCase();
      e.target.value = boxes[i].char;
      drawBoxes();
      renderBoxList();
    });
    charInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); savePlate(); }
    });
    charWrap.appendChild(charInput);

    // Delete button
    const del = document.createElement('button');
    del.className = 'box-del';
    del.textContent = '✕';
    del.onclick = (e) => { e.stopPropagation(); deleteBox(i); };

    item.appendChild(thumbCanvas);
    item.appendChild(charWrap);
    item.appendChild(del);
    list.appendChild(item);
  });
}

function setActive(i) {
  activeBox = i;
  drawBoxes();
  renderBoxList();
  // Focus the char input
  const inputs = document.querySelectorAll('.box-char input');
  if (inputs[i]) inputs[i].focus();
}

function deleteBox(i) {
  boxes.splice(i, 1);
  if (activeBox >= boxes.length) activeBox = boxes.length - 1;
  drawBoxes();
  renderBoxList();
}

// ── Draw new box by dragging ───────────────────────────────────────
canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  drawing  = true;
  const r  = canvas.getBoundingClientRect();
  dragStart = { x: e.clientX - r.left, y: e.clientY - r.top };
});

canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const r  = canvas.getBoundingClientRect();
  const cx = e.clientX - r.left;
  const cy = e.clientY - r.top;
  drawBoxes();
  ctx.strokeStyle = '#e8f400';
  ctx.lineWidth   = 2;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(dragStart.x, dragStart.y, cx - dragStart.x, cy - dragStart.y);
  ctx.setLineDash([]);
});

canvas.addEventListener('mouseup', e => {
  if (!drawing) return;
  drawing = false;
  const r  = canvas.getBoundingClientRect();
  const cx = e.clientX - r.left;
  const cy = e.clientY - r.top;

  let x = Math.min(dragStart.x, cx) / displayScale;
  let y = Math.min(dragStart.y, cy) / displayScale;
  let w = Math.abs(cx - dragStart.x) / displayScale;
  let h = Math.abs(cy - dragStart.y) / displayScale;

  if (w > 4 && h > 4) {
    boxes.push({x: Math.round(x), y: Math.round(y),
                w: Math.round(w), h: Math.round(h), char: ''});
    boxes.sort((a, b) => a.x - b.x);
    activeBox = boxes.length - 1;
    drawBoxes();
    renderBoxList();
    setActive(activeBox);
  } else {
    drawBoxes();
  }
});

// ── Save / navigate ────────────────────────────────────────────────
async function savePlate() {
  const plate = document.getElementById('plate-input').value.trim().toUpperCase();
  await fetch('/save', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({index: current.index, plate, boxes})
  });
  flash('Saved: ' + plate);
  loadPlate(current.index + 1);
}

async function skipPlate() {
  await fetch('/save', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({index: current.index, plate:'SKIP', boxes:[]})
  });
  flash('Skipped', '#ff4d00');
  loadPlate(current.index + 1);
}

async function navigate(dir) {
  const next = Math.max(0, Math.min(current.index + dir, current.total - 1));
  await fetch('/navigate', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({index: next})
  });
  loadPlate(next);
}

function flash(msg, color='#2ecc71') {
  const el = document.getElementById('flash');
  el.textContent = msg; el.style.background = color;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 900);
}

// ── Keyboard shortcuts ─────────────────────────────────────────────
document.addEventListener('keydown', e => {
  const tag = document.activeElement.tagName;
  if (e.key === 'Enter'      && tag !== 'INPUT') { e.preventDefault(); savePlate(); }
  if (e.key === 'ArrowRight' && tag !== 'INPUT') { e.preventDefault(); navigate(1); }
  if (e.key === 'ArrowLeft'  && tag !== 'INPUT') { e.preventDefault(); navigate(-1); }
  if (e.key === 'n'          && tag !== 'INPUT') { e.preventDefault(); skipPlate(); }
  if (e.key === 'Delete' && activeBox >= 0)      { e.preventDefault(); deleteBox(activeBox); }
});

// ── Init ───────────────────────────────────────────────────────────
loadPlate({{ start_index }});
</script>
</body>
</html>
"""

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plate deskew + char segmentation + annotation")
    parser.add_argument("--input",  default="./plates", help="Folder with plate images")
    parser.add_argument("--output", default="./output", help="Output folder")
    parser.add_argument("--port",   type=int, default=5000)
    args = parser.parse_args()

    input_path = Path(args.input)
    images = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        print(f"No images found in {args.input}")
        exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    state["images"]     = images
    state["output_dir"] = args.output
    state["annotations"] = load_csv_if_exists()

    # Resume from first unannotated
    annotated = set(state["annotations"].keys())
    start_idx = next((i for i, img in enumerate(images) if img.name not in annotated), len(images))
    state["index"] = start_idx

    state["start_index"] = start_idx

    print(f"\n  {len(images)} images | {len(annotated)} already annotated")
    print(f"  Output → {args.output}/")
    print(f"  Open   → http://localhost:{args.port}\n")
    app.run(port=args.port, debug=False)