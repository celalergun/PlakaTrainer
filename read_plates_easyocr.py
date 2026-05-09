"""
Local License Plate Ground Truth Generator — EasyOCR version
Deep learning based, more accurate than Tesseract on difficult plates.
Fully offline after first run (models are downloaded once and cached).

Usage:
    pip install easyocr opencv-python
    python read_plates_easyocr.py --input ./plates --output ground_truth.csv

Optional flags:
    --gpu          Use GPU if available (much faster for 8000+ images)
    --confidence   Minimum confidence threshold 0.0–1.0 (default: 0.2)
"""

import easyocr
import cv2
import csv
import argparse
import re
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def preprocess(image_path: Path):
    """Light preprocessing — EasyOCR handles most cases well on its own."""
    img = cv2.imread(str(image_path))

    # Upscale very small images
    h, w = img.shape[:2]
    if w < 300:
        scale = 300 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return img


def clean_plate_text(texts: list[tuple]) -> tuple[str, float]:
    """
    Merge multi-word results and strip non-plate characters.
    Returns (plate_text, confidence).
    """
    if not texts:
        return "", 0.0

    # Each item: (bbox, text, confidence)
    parts = []
    confidences = []
    for (_, text, conf) in texts:
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        if cleaned:
            parts.append(cleaned)
            confidences.append(conf)

    plate = " ".join(parts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return plate, avg_conf


def process_folder(input_dir: str, output_csv: str, use_gpu: bool, min_confidence: float):
    input_path = Path(input_dir)
    image_files = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images.")
    print("Loading EasyOCR model (downloaded once, cached after)...\n")

    # English covers Latin-alphabet plates; add more languages if needed
    # e.g. reader = easyocr.Reader(['en', 'tr'], gpu=use_gpu)
    reader = easyocr.Reader(["en"], gpu=use_gpu)

    print(f"{'GPU' if use_gpu else 'CPU'} mode | min confidence: {min_confidence}\n")

    results = []
    low_confidence = []

    for i, img_path in enumerate(image_files, 1):
        try:
            img = preprocess(img_path)
            detections = reader.readtext(img)
            plate_text, confidence = clean_plate_text(detections)

            flag = ""
            if confidence < min_confidence or not plate_text:
                flag = " ⚠️  LOW CONFIDENCE"
                low_confidence.append(img_path.name)

            results.append({
                "filename": img_path.name,
                "plate": plate_text,
                "confidence": f"{confidence:.2f}",
            })
            print(f"[{i}/{len(image_files)}] {img_path.name}  →  {plate_text or '(empty)'}  ({confidence:.2f}){flag}")

        except Exception as e:
            print(f"[{i}/{len(image_files)}] {img_path.name}  →  ERROR: {e}")
            results.append({"filename": img_path.name, "plate": "ERROR", "confidence": "0.00"})

    # Write main CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "plate", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    # Write a separate list of low-confidence images for manual review
    review_path = Path(output_csv).with_stem(Path(output_csv).stem + "_review")
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        f.write("filename\n")
        for name in low_confidence:
            f.write(name + "\n")

    print(f"\n✅ Done!")
    print(f"   Results       → {output_csv}")
    print(f"   Manual review → {review_path}  ({len(low_confidence)} images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read license plates locally with EasyOCR")
    parser.add_argument("--input",      default="./plates",          help="Folder with plate images")
    parser.add_argument("--output",     default="ground_truth.csv",  help="Output CSV path")
    parser.add_argument("--gpu",        action="store_true",         help="Use GPU (requires CUDA)")
    parser.add_argument("--confidence", type=float, default=0.2,     help="Min confidence threshold")
    args = parser.parse_args()

    process_folder(args.input, args.output, args.gpu, args.confidence)
    