#!/usr/bin/env python3
"""
01_detect_plates.py — Detect license plates using YOLOv26 ONNX model.

Reads images from an input directory, runs plate detection using kareplaka.onnx,
and saves bounding box annotations to annotations.txt.
Images with no detected plates are logged to unknown_plates.txt.

Output format (annotations.txt):
    filename x1 y1 x2 y2 confidence class_id

Usage:
    python 01_detect_plates.py
    python 01_detect_plates.py --input-dir /path/to/images --confidence-threshold 0.3
"""

import argparse
import os
import sys
import glob

import cv2
import numpy as np
import onnxruntime as ort


def detect_angle(gray, max_angle=15.0):
    """Detect plate tilt using Hough lines first, then a contour fallback."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=60,
                             minLineLength=gray.shape[1] // 4,
                             maxLineGap=20)
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) <= max_angle:
                    angles.append(angle)
        if angles:
            return float(np.median(angles))

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        if abs(angle) <= max_angle:
            return float(angle)

    return 0.0


def deskew(img, max_angle=15.0):
    """Rotate the image to correct the detected tilt."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = detect_angle(gray, max_angle)
    if abs(angle) < 0.5:
        return img, 0.0

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(
        img,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def crop_and_deskew(img, x1, y1, x2, y2, max_angle=15.0):
    """Crop the detected plate region, upscale if small, then deskew."""
    h, w = img.shape[:2]
    cx1 = max(0, int(x1))
    cy1 = max(0, int(y1))
    cx2 = min(w, int(x2))
    cy2 = min(h, int(y2))
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return crop, 0.0
    cw = cx2 - cx1
    if cw < 300:
        scale = 300 / cw
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return deskew(crop, max_angle)


def letterbox(img, new_shape=(640, 640)):
    """Resize and pad image to new_shape maintaining aspect ratio.
    Returns the padded image, scale ratio, and (dw, dh) padding."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad_h, new_unpad_w = int(round(h * r)), int(round(w * r))
    dw = (new_shape[1] - new_unpad_w) / 2
    dh = (new_shape[0] - new_unpad_h) / 2

    if (w, h) != (new_unpad_w, new_unpad_h):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


def preprocess(img, input_size=(640, 640)):
    """Preprocess image for ONNX inference."""
    img_lb, ratio, (dw, dh) = letterbox(img, input_size)
    # BGR -> RGB, HWC -> CHW, normalize to [0, 1], add batch dim
    blob = img_lb[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    return blob, ratio, (dw, dh)


def postprocess(output, ratio, dw_dh, conf_threshold=0.5):
    """Parse YOLOv26 end-to-end output [1, 300, 6] -> list of (x1, y1, x2, y2, conf, class_id).
    Coordinates are scaled back to original image space."""
    dw, dh = dw_dh
    detections = output[0]  # shape: [300, 6]
    results = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < conf_threshold:
            continue
        # Remove padding offset, then undo scale
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio
        results.append((x1, y1, x2, y2, float(conf), int(cls_id)))
    return results


def get_image_files(input_dir):
    """Gather all image files from input directory."""
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Detect license plates using YOLOv26 ONNX model")
    parser.add_argument("--input-dir", default="/media/ce/ce_sata/Data/Plaka/plates",
                        help="Directory containing plate images")
    parser.add_argument("--model-path", default="kareplaka.onnx",
                        help="Path to the ONNX model")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Minimum confidence threshold for detections")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to write annotations.txt and unknown_plates.txt")
    parser.add_argument("--max-angle", type=float, default=15.0,
                        help="Max tilt angle to correct in degrees")
    parser.add_argument("--debug", action="store_true",
                        help="Print model metadata and first few detections")
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(script_dir, args.model_path)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    deskewed_dir = os.path.join(output_dir, "deskewed")
    os.makedirs(deskewed_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape  # [1, 3, 640, 640]
    input_size = (input_shape[2], input_shape[3])

    if args.debug:
        print(f"  Input: {input_name}, shape={input_shape}")
        for out in session.get_outputs():
            print(f"  Output: {out.name}, shape={out.shape}")
        meta = session.get_modelmeta()
        for k, v in meta.custom_metadata_map.items():
            print(f"  {k}: {v}")

    # Get image files
    image_files = get_image_files(args.input_dir)
    if not image_files:
        print(f"No images found in {args.input_dir}")
        sys.exit(1)
    print(f"Found {len(image_files)} images in {args.input_dir}")

    annotations_path = os.path.join(output_dir, "annotations.txt")
    unknown_path = os.path.join(output_dir, "unknown_plates.txt")

    total_detections = 0
    unknown_count = 0

    with open(annotations_path, "w") as ann_f, open(unknown_path, "w") as unk_f:
        for i, img_path in enumerate(image_files):
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [WARN] Cannot read: {filename}")
                unk_f.write(f"{filename}\n")
                unknown_count += 1
                continue

            # Preprocess
            blob, ratio, dw_dh = preprocess(img, input_size)

            # Inference
            outputs = session.run(None, {input_name: blob})
            output = outputs[0]  # [1, 300, 6]

            # Postprocess
            detections = postprocess(output, ratio, dw_dh, args.confidence_threshold)

            if not detections:
                print(f"  [NO PLATE] {filename}")
                unk_f.write(f"{filename}\n")
                unknown_count += 1
            else:
                stem = os.path.splitext(filename)[0]
                for det_idx, (x1, y1, x2, y2, conf, cls_id) in enumerate(detections):
                    ann_f.write(f"{filename} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {conf:.4f} {cls_id}\n")
                    total_detections += 1
                    crop, _ = crop_and_deskew(img, x1, y1, x2, y2, args.max_angle)
                    if crop.size > 0:
                        suffix = f"_{det_idx}" if len(detections) > 1 else ""
                        out_name = f"{stem}{suffix}.jpg"
                        out_path = os.path.join(deskewed_dir, out_name)
                        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                print(f"  Processed {i + 1}/{len(image_files)} images, "
                      f"{total_detections} detections, {unknown_count} unknown")

    print(f"\nDone! {total_detections} plate detections saved to {annotations_path}")
    print(f"{unknown_count} images with no plates saved to {unknown_path}")


if __name__ == "__main__":
    main()
