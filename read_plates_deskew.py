"""
License Plate Ground Truth Generator — with Auto-Deskew
Detects tilt up to ~15° and corrects it before OCR.

Usage:
    pip install easyocr opencv-python numpy
    python read_plates_deskew.py --input ./plates --output ground_truth.csv

Optional flags:
    --gpu            Use GPU if available
    --confidence     Min confidence threshold 0.0–1.0 (default: 0.2)
    --max-angle      Max tilt angle to correct in degrees (default: 15)
    --debug          Save deskewed images to ./debug/ for inspection
"""

import easyocr
import cv2
import numpy as np
import csv
import argparse
import re
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─── Deskew ───────────────────────────────────────────────────────────────────

def detect_angle(gray: np.ndarray, max_angle: float) -> float:
    """
    Detect the tilt angle of a license plate using two complementary methods:
    1. Hough line detection on edges (best for clean plates)
    2. Minimum-area bounding rectangle on the largest contour (fallback)
    Returns angle in degrees (negative = tilted left, positive = tilted right).
    """
    # --- Method 1: Hough lines ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=60, minLineLength=gray.shape[1] // 4,
                             maxLineGap=20)
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only keep near-horizontal lines
                if abs(angle) <= max_angle:
                    angles.append(angle)
        if angles:
            # Use median to ignore outlier lines
            return float(np.median(angles))

    # --- Method 2: Minimum bounding rectangle fallback ---
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        angle = rect[-1]  # angle in [-90, 0)
        # Convert to [-45, 45] range
        if angle < -45:
            angle += 90
        if abs(angle) <= max_angle:
            return float(angle)

    return 0.0


def deskew(img: np.ndarray, max_angle: float) -> tuple[np.ndarray, float]:
    """
    Rotate image to correct detected tilt.
    Returns (corrected_image, angle_applied).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = detect_angle(gray, max_angle)

    if abs(angle) < 0.5:
        return img, 0.0  # Not worth rotating

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Expand canvas so corners aren't clipped
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(image_path: Path, max_angle: float, debug_dir: Path | None) -> tuple[np.ndarray, float]:
    img = cv2.imread(str(image_path))

    # Upscale small images
    h, w = img.shape[:2]
    if w < 300:
        scale = 300 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Deskew
    img, angle = deskew(img, max_angle)

    # Optionally save debug image
    if debug_dir and abs(angle) >= 0.5:
        debug_path = debug_dir / image_path.name
        cv2.imwrite(str(debug_path), img)

    return img, angle


# ─── OCR ──────────────────────────────────────────────────────────────────────

def clean_plate_text(detections: list) -> tuple[str, float]:
    """Merge EasyOCR results, strip non-plate characters, return (text, confidence)."""
    if not detections:
        return "", 0.0

    parts, confidences = [], []
    for (_, text, conf) in detections:
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        if cleaned:
            parts.append(cleaned)
            confidences.append(conf)

    plate = " ".join(parts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return plate, avg_conf


# ─── Main ─────────────────────────────────────────────────────────────────────

def process_folder(input_dir: str, output_csv: str,
                   use_gpu: bool, min_confidence: float,
                   max_angle: float, debug: bool):

    input_path = Path(input_dir)
    image_files = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    debug_dir = None
    if debug:
        debug_dir = Path("./debug")
        debug_dir.mkdir(exist_ok=True)
        print(f"Debug mode: deskewed images saved to {debug_dir}/\n")

    print(f"Found {len(image_files)} images.")
    print("Loading EasyOCR model...\n")
    reader = easyocr.Reader(["en"], gpu=use_gpu)
    print(f"{'GPU' if use_gpu else 'CPU'} | max tilt: ±{max_angle}° | min confidence: {min_confidence}\n")

    results = []
    low_confidence = []

    for i, img_path in enumerate(image_files, 1):
        try:
            img, angle = preprocess(img_path, max_angle, debug_dir)
            detections = reader.readtext(img)
            plate_text, confidence = clean_plate_text(detections)

            angle_str = f"{angle:+.1f}°" if abs(angle) >= 0.5 else "  0.0°"
            flag = ""
            if confidence < min_confidence or not plate_text:
                flag = " ⚠️"
                low_confidence.append(img_path.name)

            results.append({
                "filename": img_path.name,
                "plate": plate_text,
                "confidence": f"{confidence:.2f}",
                "angle_corrected": f"{angle:.1f}",
            })
            print(f"[{i}/{len(image_files)}] {img_path.name}  →  "
                  f"{plate_text or '(empty)'}  (conf: {confidence:.2f}, tilt: {angle_str}){flag}")

        except Exception as e:
            print(f"[{i}/{len(image_files)}] {img_path.name}  →  ERROR: {e}")
            results.append({"filename": img_path.name, "plate": "ERROR",
                            "confidence": "0.00", "angle_corrected": "0.0"})

    # Main CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "plate", "confidence", "angle_corrected"])
        writer.writeheader()
        writer.writerows(results)

    # Low-confidence list for manual review
    review_path = Path(output_csv).with_stem(Path(output_csv).stem + "_review")
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        f.write("filename\n")
        for name in low_confidence:
            f.write(name + "\n")

    print(f"\n✅ Done!")
    print(f"   Results       → {output_csv}")
    print(f"   Manual review → {review_path}  ({len(low_confidence)} images)")
    if debug_dir:
        print(f"   Debug images  → {debug_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deskew + OCR license plates")
    parser.add_argument("--input",      default="./plates",         help="Folder with plate images")
    parser.add_argument("--output",     default="ground_truth.csv", help="Output CSV path")
    parser.add_argument("--gpu",        action="store_true",        help="Use GPU (requires CUDA)")
    parser.add_argument("--confidence", type=float, default=0.2,    help="Min confidence threshold")
    parser.add_argument("--max-angle",  type=float, default=15.0,   help="Max tilt angle to correct")
    parser.add_argument("--debug",      action="store_true",        help="Save deskewed images to ./debug/")
    args = parser.parse_args()

    process_folder(args.input, args.output, args.gpu,
                   args.confidence, args.max_angle, args.debug)
    