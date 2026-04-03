#!/usr/bin/env python3
"""
03_review_digits.py — Visual review tool for classified character images.

Shows each character alongside its source license plate with the character
position highlighted by a green rectangle. Character index is displayed
on the plate image above each detected character box.

Keyboard controls:
    Enter / Space  — Accept current classification, next image
    A-Z, 0-9      — Reclassify: move image to typed character's folder
    d              — Delete current image
    s              — Skip (mark suspicious, move to 'review' folder)
    q              — Quit
    Left arrow     — Previous image
    Right arrow    — Next image
    n              — Next folder
    p              — Previous folder

Usage:
    python 03_review_digits.py
    python 03_review_digits.py --letters-dir letters/ --input-dir /path/to/images
"""

import argparse
import json
import os
import re
import sys
import shutil
from collections import defaultdict

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Plate analysis functions (must match 02_extract_digits.py logic exactly)
# ---------------------------------------------------------------------------

def enhance_plate(plate_img):
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def find_character_contours(enhanced, plate_h, plate_w):
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 19, 9
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_h = plate_h * 0.3
    max_h = plate_h * 0.95
    min_w = plate_w * 0.01
    min_area = plate_h * plate_w * 0.005

    char_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < min_h or h > max_h or w < min_w or w * h < min_area or w > plate_w * 0.5:
            continue
        char_rects.append((x, y, w, h))

    char_rects = remove_overlapping(char_rects)
    char_rects.sort(key=lambda r: r[0])
    return char_rects


def remove_overlapping(rects):
    if not rects:
        return rects
    keep = [True] * len(rects)
    for i in range(len(rects)):
        if not keep[i]:
            continue
        xi, yi, wi, hi = rects[i]
        for j in range(i + 1, len(rects)):
            if not keep[j]:
                continue
            xj, yj, wj, hj = rects[j]
            ox1, oy1 = max(xi, xj), max(yi, yj)
            ox2, oy2 = min(xi + wi, xj + wj), min(yi + hi, yj + hj)
            if ox2 > ox1 and oy2 > oy1:
                overlap = (ox2 - ox1) * (oy2 - oy1)
                if overlap > 0.5 * min(wi * hi, wj * hj):
                    if wi * hi >= wj * hj:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
    return [r for r, k in zip(rects, keep) if k]


def split_merged(rects):
    final = []
    for x, y, w, h in rects:
        if h > 0 and w / h > 0.85:
            half = w // 2
            final.append((x, y, half, h))
            final.append((x + half, y, w - half, h))
        else:
            final.append((x, y, w, h))
    final.sort(key=lambda r: r[0])
    return final


# ---------------------------------------------------------------------------
# Annotations and filename helpers
# ---------------------------------------------------------------------------

def read_annotations(path):
    dets = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 6)
            fname = parts[0]
            x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            conf, cls = float(parts[5]), int(parts[6])
            dets[fname].append((x1, y1, x2, y2, conf, cls))
    return dets


def parse_char_filename(filepath):
    """Parse {base}_plate{N}_char{M}.ext -> (base, plate_idx, char_idx)."""
    name = os.path.splitext(os.path.basename(filepath))[0]
    m = re.match(r'^(.+)_plate(\d+)_char(\d+)$', name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None


class PlateCache:
    """Caches plate crop and character rectangles for the last accessed plate."""

    def __init__(self, annotations, input_dir, base_to_filename):
        self._annotations = annotations
        self._input_dir = input_dir
        self._base_to_filename = base_to_filename
        self._key = None
        self._plate_crop = None
        self._char_rects = None

    def get(self, base_name, plate_idx):
        """Returns (plate_color_crop, char_rects) or (None, None)."""
        key = (base_name, plate_idx)
        if key == self._key:
            return self._plate_crop, self._char_rects

        orig_fname = self._base_to_filename.get(base_name)
        if not orig_fname:
            return None, None

        boxes = self._annotations.get(orig_fname, [])
        if plate_idx >= len(boxes):
            return None, None

        img_path = os.path.join(self._input_dir, orig_fname)
        img = cv2.imread(img_path)
        if img is None:
            return None, None

        box = boxes[plate_idx]
        h, w = img.shape[:2]
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
        if x2 <= x1 or y2 <= y1:
            return None, None

        plate_crop = img[y1:y2, x1:x2]
        ph, pw = plate_crop.shape[:2]
        if ph < 10 or pw < 20:
            return None, None

        enhanced = enhance_plate(plate_crop)
        char_rects = find_character_contours(enhanced, ph, pw)
        char_rects = split_merged(char_rects)

        self._key = key
        self._plate_crop = plate_crop
        self._char_rects = char_rects
        return plate_crop, char_rects


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

DISPLAY_W = 600


def build_display(char_img, plate_crop, char_rects, char_idx,
                  folder_name, img_idx, total, filepath):
    """Build composite image: info + plate with highlight + enlarged char + help."""

    # --- Info bar ---
    info_bar = 195 * np.ones((45, DISPLAY_W, 3), dtype=np.uint8)
    fname = os.path.basename(filepath)
    if len(fname) > 60:
        fname = fname[:57] + "..."
    cv2.putText(info_bar, f"[{folder_name}] {img_idx + 1}/{total}",
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    char_text = f"Char #{char_idx}" if char_idx is not None else ""
    cv2.putText(info_bar, f"{char_text}  {fname}",
                (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)

    # --- Plate image with highlighted character ---
    plate_row = None
    if plate_crop is not None:
        vis = plate_crop.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if char_rects:
            for i, (cx, cy, cw, ch) in enumerate(char_rects):
                is_cur = (i == char_idx)
                color = (0, 255, 0) if is_cur else (128, 128, 128)
                pad = 3 if is_cur else 0
                rx1 = max(0, cx - pad)
                ry1 = max(0, cy - pad)
                rx2 = min(vis.shape[1], cx + cw + pad)
                ry2 = min(vis.shape[0], cy + ch + pad)
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), color, 1)
                cv2.putText(vis, str(i), (cx, max(cy - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        ph, pw = vis.shape[:2]
        scale = DISPLAY_W / pw if pw > 0 else 1
        plate_row = cv2.resize(vis, (DISPLAY_W, max(1, int(ph * scale))),
                               interpolation=cv2.INTER_LINEAR)

    # --- Enlarged character ---
    ch_h, ch_w = char_img.shape[:2]
    s = max(1, min(10, 120 // max(ch_h, 1), 120 // max(ch_w, 1)))
    enlarged = cv2.resize(char_img, (ch_w * s, ch_h * s),
                          interpolation=cv2.INTER_NEAREST)
    if len(enlarged.shape) == 2:
        enlarged = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)

    eh, ew = enlarged.shape[:2]
    char_row = 200 * np.ones((eh + 10, DISPLAY_W, 3), dtype=np.uint8)
    x_off = (DISPLAY_W - ew) // 2
    if x_off >= 0 and x_off + ew <= DISPLAY_W:
        char_row[5:5 + eh, x_off:x_off + ew] = enlarged

    # --- Help bar ---
    help_bar = 195 * np.ones((25, DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(help_bar,
                "Space:next  d:del  s:skip  A-Z/0-9:reclassify  n/p:folder  q:quit",
                (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    # --- Compose ---
    parts = [info_bar]
    if plate_row is not None:
        parts.append(plate_row)
        parts.append(170 * np.ones((2, DISPLAY_W, 3), dtype=np.uint8))
    parts.extend([char_row, help_bar])
    return np.vstack(parts)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def gather_images(letters_dir):
    folders = []
    for entry in sorted(os.listdir(letters_dir)):
        folder_path = os.path.join(letters_dir, entry)
        if not os.path.isdir(folder_path):
            continue
        images = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        if images:
            folders.append((entry, images))
    return folders


def move_to_folder(filepath, letters_dir, target_char):
    target_folder = os.path.join(letters_dir, target_char)
    os.makedirs(target_folder, exist_ok=True)
    dest = os.path.join(target_folder, os.path.basename(filepath))
    if os.path.abspath(filepath) == os.path.abspath(dest):
        return dest
    if os.path.exists(dest):
        base, ext = os.path.splitext(os.path.basename(filepath))
        counter = 1
        while os.path.exists(dest):
            dest = os.path.join(target_folder, f"{base}_{counter}{ext}")
            counter += 1
    shutil.move(filepath, dest)
    return dest


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Review and reclassify extracted character images")
    parser.add_argument("--letters-dir", default="letters",
                        help="Path to the letters/ directory")
    parser.add_argument("--annotations", default="annotations.txt",
                        help="Path to annotations.txt from 01_detect_plates.py")
    parser.add_argument("--input-dir",
                        default="/media/ce/ce_sata/Data/Plaka/plates",
                        help="Directory containing original plate images")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(script_dir, p)

    letters_dir = resolve(args.letters_dir)
    annotations_path = resolve(args.annotations)
    input_dir = args.input_dir

    if not os.path.isdir(letters_dir):
        print(f"Error: {letters_dir} not found. Run 02_extract_digits.py first.")
        sys.exit(1)

    # Load annotations for plate look-ups
    annotations = {}
    base_to_filename = {}
    if os.path.exists(annotations_path):
        annotations = read_annotations(annotations_path)
        for fname in annotations:
            base_to_filename[os.path.splitext(fname)[0]] = fname
        print(f"Loaded annotations for {len(annotations)} images")
    else:
        print(f"Warning: {annotations_path} not found — plate context unavailable")

    plate_cache = PlateCache(annotations, input_dir, base_to_filename)

    # Progress file to resume later
    progress_path = os.path.join(script_dir, ".review_progress.json")

    folders = gather_images(letters_dir)
    if not folders:
        print("No character images found.")
        sys.exit(0)

    folder_idx = 0
    img_idx = 0

    # Restore last position if progress file exists
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as pf:
                prog = json.load(pf)
            saved_folder = prog.get("folder", "")
            saved_file = prog.get("file", "")
            for fi, (fn, imgs) in enumerate(folders):
                if fn == saved_folder:
                    folder_idx = fi
                    # Try to find the exact file
                    for ii, fp in enumerate(imgs):
                        if os.path.basename(fp) == saved_file:
                            img_idx = ii
                            break
                    break
            print(f"Resumed from [{saved_folder}] {saved_file}")
        except Exception:
            pass
    window_name = "Review Characters"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print(f"Found {len(folders)} character folders")
    print("Controls: Space/Enter=next  d=delete  s=skip  "
          "A-Z/0-9=reclassify  n/p=folder  arrows=prev/next  q=quit")

    while True:
        folders = gather_images(letters_dir)
        if not folders:
            print("No more images to review.")
            break

        folder_idx = min(folder_idx, len(folders) - 1)
        folder_name, images = folders[folder_idx]
        if not images:
            folder_idx = (folder_idx + 1) % len(folders)
            continue
        img_idx = min(img_idx, len(images) - 1)

        filepath = images[img_idx]
        char_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if char_img is None:
            img_idx += 1
            if img_idx >= len(images):
                img_idx = 0
                folder_idx = (folder_idx + 1) % len(folders)
            continue

        # Resolve plate context from filename
        plate_crop, char_rects, char_idx_val = None, None, None
        parsed = parse_char_filename(filepath)
        if parsed:
            base_name, plate_idx, char_idx_val = parsed
            plate_crop, char_rects = plate_cache.get(base_name, plate_idx)

        display = build_display(char_img, plate_crop, char_rects, char_idx_val,
                                folder_name, img_idx, len(images), filepath)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break

        elif key in (13, 32):  # Enter / Space
            img_idx += 1
            if img_idx >= len(images):
                img_idx = 0
                folder_idx = (folder_idx + 1) % len(folders)

        elif key == ord('d'):
            os.remove(filepath)
            print(f"  Deleted: {os.path.basename(filepath)}")
            if img_idx >= len(images) - 1:
                img_idx = max(0, len(images) - 2)

        elif key == ord('s'):
            move_to_folder(filepath, letters_dir, "review")
            print(f"  Moved to review/: {os.path.basename(filepath)}")
            if img_idx >= len(images) - 1:
                img_idx = max(0, len(images) - 2)

        elif key == ord('n'):
            folder_idx = (folder_idx + 1) % len(folders)
            img_idx = 0

        elif key == ord('p'):
            folder_idx = (folder_idx - 1) % len(folders)
            img_idx = 0

        elif key == 81:  # Left arrow
            img_idx = max(0, img_idx - 1)

        elif key == 83:  # Right arrow
            img_idx = min(len(images) - 1, img_idx + 1)

        else:
            char = chr(key).upper() if 0 <= key < 128 else ''
            if char and char.isalnum() and len(char) == 1:
                if char != folder_name:
                    move_to_folder(filepath, letters_dir, char)
                    print(f"  Reclassified '{folder_name}' -> '{char}': "
                          f"{os.path.basename(filepath)}")
                    if img_idx >= len(images) - 1:
                        img_idx = max(0, len(images) - 2)
                else:
                    img_idx += 1
                    if img_idx >= len(images):
                        img_idx = 0
                        folder_idx = (folder_idx + 1) % len(folders)

    # Save progress
    folders = gather_images(letters_dir)
    if folders:
        fi = min(folder_idx, len(folders) - 1)
        fn, imgs = folders[fi]
        ii = min(img_idx, len(imgs) - 1) if imgs else 0
        prog = {"folder": fn,
                "file": os.path.basename(imgs[ii]) if imgs else ""}
        with open(progress_path, "w") as pf:
            json.dump(prog, pf)
        print(f"Progress saved: [{fn}] image {ii + 1}/{len(imgs)}")

    cv2.destroyAllWindows()
    print("Review complete.")


if __name__ == "__main__":
    main()
