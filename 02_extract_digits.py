#!/usr/bin/env python3
"""
02_extract_digits.py — Extract individual characters from detected license plates.

Reads annotations.txt (from 01_detect_plates.py), crops plate regions,
uses contour detection to find individual characters, runs Tesseract OCR
to classify each character, and organizes cropped images into per-character folders.

Output structure:
    letters/A/  letters/B/  ...  letters/0/  letters/1/  ...  letters/unknown/
    suspicious.txt  — plates with wrong blob count or merged characters

Usage:
    python 02_extract_digits.py
    python 02_extract_digits.py --annotations annotations.txt --input-dir /path/to/images
"""

import argparse
import os
import sys
import shutil
from collections import defaultdict

import cv2
import numpy as np
import pytesseract


def read_annotations(annotations_path):
    """Read annotations.txt and group detections by filename.
    Returns dict: filename -> [(x1, y1, x2, y2, conf, cls_id), ...]"""
    detections = defaultdict(list)
    with open(annotations_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 6)
            # Format: filename x1 y1 x2 y2 conf cls_id
            # Filename may contain spaces, so rsplit from right
            filename = parts[0]
            x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            conf = float(parts[5])
            cls_id = int(parts[6])
            detections[filename].append((x1, y1, x2, y2, conf, cls_id))
    return detections


def crop_plate(img, box):
    """Crop plate region from image, clamping to image boundaries."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def enhance_plate(plate_img):
    """Convert to grayscale, apply CLAHE for contrast enhancement."""
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def find_character_contours(enhanced, plate_h, plate_w):
    """Find character contours using adaptive thresholding.
    Returns list of (x, y, w, h) bounding rects sorted left-to-right."""
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 19, 9
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    min_h = plate_h * 0.3   # character must be at least 30% of plate height
    max_h = plate_h * 0.95  # but not taller than 95%
    min_w = plate_w * 0.01  # at least 1% of plate width
    max_w = plate_w * 0.25  # no wider than 25% of plate width (single char)
    min_area = plate_h * plate_w * 0.005  # minimum area

    char_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if h < min_h or h > max_h:
            continue
        if w < min_w:
            continue
        if area < min_area:
            continue
        # Allow wider rects (up to ~50% of plate) since merged chars need detection
        if w > plate_w * 0.5:
            continue

        char_rects.append((x, y, w, h))

    # Remove overlapping rects (keep larger)
    char_rects = remove_overlapping(char_rects)

    # Sort left to right
    char_rects.sort(key=lambda r: r[0])
    return char_rects


def remove_overlapping(rects):
    """Remove significantly overlapping rectangles, keeping the larger one."""
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

            # Check overlap
            ox1 = max(xi, xj)
            oy1 = max(yi, yj)
            ox2 = min(xi + wi, xj + wj)
            oy2 = min(yi + hi, yj + hj)

            if ox2 > ox1 and oy2 > oy1:
                overlap_area = (ox2 - ox1) * (oy2 - oy1)
                area_i = wi * hi
                area_j = wj * hj
                min_area = min(area_i, area_j)

                # If overlap is > 50% of the smaller rect, remove the smaller one
                if overlap_area > 0.5 * min_area:
                    if area_i >= area_j:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

    return [r for r, k in zip(rects, keep) if k]


def check_and_split_merged(rects, plate_crop, enhanced):
    """Check for merged characters (aspect ratio too wide) and auto-split them.
    Returns (final_rects, merged_flags) where merged_flags is list of bools."""
    final_rects = []
    had_merge = False

    for x, y, w, h in rects:
        aspect = w / h if h > 0 else 0
        if aspect > 0.85:
            # Likely merged — split in half
            half_w = w // 2
            final_rects.append((x, y, half_w, h))
            final_rects.append((x + half_w, y, w - half_w, h))
            had_merge = True
        else:
            final_rects.append((x, y, w, h))

    final_rects.sort(key=lambda r: r[0])
    return final_rects, had_merge


def ocr_character(char_img):
    """Run Tesseract OCR on a single character image.
    Returns uppercase character or 'unknown'."""
    # Resize to a reasonable size for Tesseract
    resized = cv2.resize(char_img, (40, 60), interpolation=cv2.INTER_LINEAR)

    # Add padding around the character
    padded = cv2.copyMakeBorder(resized, 10, 10, 10, 10,
                                 cv2.BORDER_CONSTANT, value=255)

    # Invert if needed (Tesseract expects dark text on light background)
    mean_val = np.mean(padded)
    if mean_val < 127:
        padded = cv2.bitwise_not(padded)

    config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    try:
        result = pytesseract.image_to_string(padded, config=config).strip()
    except Exception:
        return "unknown"

    if result and len(result) == 1 and result.isalnum():
        return result.upper()
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Extract characters from detected license plates")
    parser.add_argument("--annotations", default="annotations.txt",
                        help="Path to annotations.txt from 01_detect_plates.py")
    parser.add_argument("--input-dir", default="/media/ce/ce_sata/Data/Plaka/plates",
                        help="Directory containing original plate images")
    parser.add_argument("--output-dir", default=".",
                        help="Base directory for output (letters/ folder and suspicious.txt)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotations_path = args.annotations if os.path.isabs(args.annotations) else os.path.join(script_dir, args.annotations)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(script_dir, args.output_dir)
    letters_dir = os.path.join(output_dir, "letters")
    suspicious_path = os.path.join(output_dir, "suspicious.txt")

    if not os.path.exists(annotations_path):
        print(f"Error: {annotations_path} not found. Run 01_detect_plates.py first.")
        sys.exit(1)

    # Read annotations
    detections = read_annotations(annotations_path)
    print(f"Loaded annotations for {len(detections)} images")

    # Create letters directory
    os.makedirs(letters_dir, exist_ok=True)

    total_chars = 0
    suspicious_count = 0
    char_counts = defaultdict(int)

    with open(suspicious_path, "w") as susp_f:
        for img_idx, (filename, boxes) in enumerate(detections.items()):
            img_path = os.path.join(args.input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [WARN] Cannot read: {filename}")
                continue

            for plate_idx, box in enumerate(boxes):
                plate_crop = crop_plate(img, box)
                if plate_crop is None:
                    continue

                plate_h, plate_w = plate_crop.shape[:2]
                if plate_h < 10 or plate_w < 20:
                    continue

                # Enhance
                enhanced = enhance_plate(plate_crop)

                # Find character contours
                char_rects = find_character_contours(enhanced, plate_h, plate_w)

                # Check for merged characters and auto-split
                char_rects, had_merge = check_and_split_merged(char_rects, plate_crop, enhanced)

                if had_merge:
                    susp_f.write(f"{filename} plate{plate_idx} merged_characters\n")
                    suspicious_count += 1

                # Count validation
                num_chars = len(char_rects)
                if num_chars < 6 or num_chars > 8:
                    susp_f.write(f"{filename} plate{plate_idx} blob_count={num_chars}\n")
                    suspicious_count += 1

                # Process each character
                for char_idx, (x, y, w, h) in enumerate(char_rects):
                    # Crop character from enhanced (grayscale) image
                    char_img = enhanced[y:y+h, x:x+w]
                    if char_img.size == 0:
                        continue

                    # OCR
                    char_label = ocr_character(char_img)

                    # Create folder and save
                    char_folder = os.path.join(letters_dir, char_label)
                    os.makedirs(char_folder, exist_ok=True)

                    # Filename: original_plate<N>_char<M>.png
                    base_name = os.path.splitext(filename)[0]
                    char_filename = f"{base_name}_plate{plate_idx}_char{char_idx}.png"
                    char_path = os.path.join(char_folder, char_filename)
                    cv2.imwrite(char_path, char_img)

                    char_counts[char_label] += 1
                    total_chars += 1

            if (img_idx + 1) % 100 == 0 or (img_idx + 1) == len(detections):
                print(f"  Processed {img_idx + 1}/{len(detections)} images, {total_chars} chars extracted")

    # Summary
    print(f"\nDone! Extracted {total_chars} characters into {letters_dir}/")
    print(f"Suspicious entries: {suspicious_count} (saved to {suspicious_path})")
    print("\nCharacter distribution:")
    for char, count in sorted(char_counts.items(), key=lambda x: -x[1]):
        print(f"  {char}: {count}")


if __name__ == "__main__":
    main()
