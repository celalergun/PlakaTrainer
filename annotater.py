"""
License Plate Manual Annotation Tool
A fast local web app for annotating plate images one by one.

Usage:
    pip install flask
    python annotate.py --input ./plates --output ground_truth.csv

Keybindings (in browser):
    Type plate text  →  fills the input
    Enter            →  save & next
    Backspace        →  go back to previous
    Escape           →  skip (marks as SKIP)
"""

import csv
import argparse
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_file

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = Flask(__name__)

# ─── Global state ─────────────────────────────────────────────────────────────
state = {
    "images": [],       # list of Path objects
    "index": 0,         # current position
    "annotations": {},  # filename -> plate text
    "output_csv": "",
}

# ─── HTML Template ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Plate Annotator</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0a0a0a;
    --surface: #111;
    --border: #222;
    --accent: #e8f400;
    --accent2: #ff4d00;
    --text: #f0f0f0;
    --muted: #555;
    --mono: 'JetBrains Mono', monospace;
    --display: 'Syne', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }

  .shell {
    width: 100%;
    max-width: 720px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  /* Header */
  .header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .title {
    font-family: var(--display);
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
  }
  .counter {
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.1em;
  }

  /* Progress bar */
  .progress-track {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.3s ease;
  }

  /* Image card */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }
  .img-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 32px;
    min-height: 180px;
    background: repeating-linear-gradient(
      45deg, #0d0d0d, #0d0d0d 10px, #0a0a0a 10px, #0a0a0a 20px
    );
  }
  .img-wrap img {
    max-width: 100%;
    max-height: 200px;
    image-rendering: pixelated;
    border: 2px solid var(--border);
    display: block;
  }
  .filename {
    padding: 8px 16px;
    font-size: 0.65rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Input area */
  .input-row {
    display: flex;
    gap: 10px;
  }
  input[type="text"] {
    flex: 1;
    background: var(--surface);
    border: 2px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    padding: 14px 20px;
    border-radius: 4px;
    outline: none;
    transition: border-color 0.15s;
    text-align: center;
  }
  input[type="text"]:focus {
    border-color: var(--accent);
  }
  input[type="text"].saved {
    border-color: #2ecc71;
    transition: border-color 0s;
  }

  /* Buttons */
  .btn {
    background: transparent;
    border: 2px solid var(--border);
    color: var(--muted);
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0 20px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }
  .btn:hover { border-color: var(--text); color: var(--text); }
  .btn.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #000;
    font-weight: 700;
    padding: 0 28px;
  }
  .btn.primary:hover { background: #fff; border-color: #fff; }
  .btn.danger { border-color: var(--accent2); color: var(--accent2); }
  .btn.danger:hover { background: var(--accent2); color: #fff; }

  /* Footer hints */
  .hints {
    display: flex;
    gap: 20px;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 0.08em;
  }
  .hint kbd {
    background: var(--border);
    color: var(--text);
    padding: 2px 6px;
    border-radius: 3px;
    font-family: var(--mono);
  }

  /* Flash feedback */
  .flash {
    position: fixed;
    top: 20px;
    right: 20px;
    background: #2ecc71;
    color: #000;
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 8px 16px;
    border-radius: 4px;
    opacity: 0;
    transform: translateY(-8px);
    transition: all 0.2s;
    pointer-events: none;
  }
  .flash.show { opacity: 1; transform: translateY(0); }

  /* Done screen */
  .done {
    text-align: center;
    padding: 60px 0;
  }
  .done h2 { font-family: var(--display); font-size: 2rem; color: var(--accent); margin-bottom: 12px; }
  .done p { color: var(--muted); font-size: 0.8rem; }
</style>
</head>
<body>
<div class="shell">
  <div class="header">
    <span class="title">Plate Annotator</span>
    <span class="counter" id="counter">— / —</span>
  </div>

  <div class="progress-track">
    <div class="progress-fill" id="progress" style="width:0%"></div>
  </div>

  <div id="main-area">
    <div class="card">
      <div class="img-wrap">
        <img id="plate-img" src="" alt="plate">
      </div>
      <div class="filename" id="fname"></div>
    </div>

    <div class="input-row">
      <input type="text" id="plate-input" placeholder="TYPE PLATE…" autocomplete="off" spellcheck="false">
      <button class="btn" id="btn-back" onclick="goBack()">← Back</button>
      <button class="btn danger" onclick="skip()">Skip</button>
      <button class="btn primary" onclick="save()">Save ↵</button>
    </div>

    <div class="hints">
      <span><kbd>Enter</kbd> save &amp; next</span>
      <span><kbd>Backspace</kbd> (empty box) go back</span>
      <span><kbd>Esc</kbd> skip</span>
    </div>
  </div>
</div>

<div class="flash" id="flash"></div>

<script>
  let current = {};

  async function load() {
    const r = await fetch('/current');
    current = await r.json();
    if (current.done) {
      document.getElementById('main-area').innerHTML =
        '<div class="done"><h2>ALL DONE ✓</h2><p>Ground truth saved to ' + current.output + '</p></div>';
      document.getElementById('counter').textContent = 'Complete';
      document.getElementById('progress').style.width = '100%';
      return;
    }
    document.getElementById('plate-img').src = '/image/' + current.index;
    document.getElementById('fname').textContent = current.filename;
    document.getElementById('counter').textContent =
      (current.index + 1) + ' / ' + current.total + '  (' + current.annotated + ' annotated)';
    document.getElementById('progress').style.width =
      (current.annotated / current.total * 100) + '%';

    const inp = document.getElementById('plate-input');
    inp.value = current.existing || '';
    inp.focus();
    inp.select();
  }

  async function save() {
    const val = document.getElementById('plate-input').value.trim().toUpperCase();
    if (!val) return;
    await fetch('/annotate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({index: current.index, plate: val})
    });
    flash('Saved: ' + val);
    load();
  }

  async function skip() {
    await fetch('/annotate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({index: current.index, plate: 'SKIP'})
    });
    flash('Skipped', '#ff4d00');
    load();
  }

  async function goBack() {
    await fetch('/back', {method: 'POST'});
    load();
  }

  function flash(msg, color = '#2ecc71') {
    const el = document.getElementById('flash');
    el.textContent = msg;
    el.style.background = color;
    el.classList.add('show');
    setTimeout(() => el.classList.remove('show'), 900);
  }

  document.addEventListener('keydown', e => {
    const inp = document.getElementById('plate-input');
    if (e.key === 'Enter') { e.preventDefault(); save(); }
    if (e.key === 'Escape') { e.preventDefault(); skip(); }
    if (e.key === 'Backspace' && inp.value === '') { e.preventDefault(); goBack(); }
  });

  load();
</script>
</body>
</html>
"""

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/current")
def current():
    images = state["images"]
    idx = state["index"]
    if idx >= len(images):
        return jsonify({"done": True, "output": state["output_csv"]})
    img = images[idx]
    return jsonify({
        "done": False,
        "index": idx,
        "total": len(images),
        "annotated": len(state["annotations"]),
        "filename": img.name,
        "existing": state["annotations"].get(img.name, ""),
        "output": state["output_csv"],
    })

@app.route("/image/<int:idx>")
def serve_image(idx):
    img_path = state["images"][idx]
    return send_file(str(img_path), mimetype="image/jpeg")

@app.route("/annotate", methods=["POST"])
def annotate():
    data = request.json
    idx = data["index"]
    plate = data["plate"].strip().upper()
    filename = state["images"][idx].name
    state["annotations"][filename] = plate
    state["index"] = idx + 1
    save_csv()
    return jsonify({"ok": True})

@app.route("/back", methods=["POST"])
def go_back():
    if state["index"] > 0:
        state["index"] -= 1
    return jsonify({"ok": True})

def save_csv():
    """Write annotations to CSV after every entry (crash-safe)."""
    with open(state["output_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "plate"])
        writer.writeheader()
        for filename, plate in state["annotations"].items():
            writer.writerow({"filename": filename, "plate": plate})

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual license plate annotation tool")
    parser.add_argument("--input",  default="./plates",         help="Folder with plate images")
    parser.add_argument("--output", default="ground_truth.csv", help="Output CSV path")
    parser.add_argument("--port",   type=int, default=5000,     help="Port (default: 5000)")
    args = parser.parse_args()

    input_path = Path(args.input)
    images = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        print(f"No images found in {args.input}")
        exit(1)

    # Load existing annotations (resume support)
    existing = {}
    if Path(args.output).exists():
        with open(args.output, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["filename"]] = row["plate"]
        print(f"Resuming: {len(existing)} annotations already saved.")

    state["images"] = images
    state["annotations"] = existing
    state["output_csv"] = args.output
    # Start from first unannotated image
    annotated_names = set(existing.keys())
    state["index"] = next(
        (i for i, img in enumerate(images) if img.name not in annotated_names),
        len(images)
    )

    print(f"\n  {len(images)} images found.")
    print(f"  Open http://localhost:{args.port} in your browser\n")
    app.run(port=args.port, debug=False)
    