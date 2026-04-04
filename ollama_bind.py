
import ollama
import base64
import re
from PIL import Image
import io
import os
import cv2
import numpy as np

plates_dir = "/media/ce/ce_sata/Data/Plaka/plates"
annotations_file = "annotations.txt"

def load_annotations(ann_file):
    annotations = {}
    with open(ann_file, "r") as f:
        for line in f:
            parts = line.strip().rsplit(maxsplit=6)
            if len(parts) >= 6:
                fname = parts[0]
                x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                conf = float(parts[5])
                if fname not in annotations or conf > annotations[fname][4]:
                    annotations[fname] = (x1, y1, x2, y2, conf)
    # Strip confidence from returned values
    return {k: v[:4] for k, v in annotations.items()}

def read_license_plate(cropped_plate_img):
    """
    cropped_plate_img: numpy array or PIL Image from your YOLO detector
    """
    # Convert to PIL
    if not isinstance(cropped_plate_img, Image.Image):
        cropped_plate_img = Image.fromarray(cropped_plate_img)

    # Upscale small crops so the model can read them
    min_width = 300
    if cropped_plate_img.width < min_width:
        scale = min_width / cropped_plate_img.width
        new_size = (int(cropped_plate_img.width * scale), int(cropped_plate_img.height * scale))
        cropped_plate_img = cropped_plate_img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    cropped_plate_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    response = ollama.chat(
        model="moondream",  # or "qwen2.5vl:2b"
        options={"num_gpu": 0},
        messages=[{
            "role": "user",
            "content": "Read the license plate text in this image. Return ONLY the plate number, nothing else.",
            "images": [img_bytes]
        }]
    )

    raw = response["message"]["content"].strip()
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw).upper()
    return raw, cleaned


if __name__ == "__main__":
    annotations = load_annotations(annotations_file)

    jpg_files = ([f for f in os.listdir(plates_dir) if f.lower().endswith((".jpg", ".jpeg"))])[:10]

    for fname in jpg_files:
        img_path = os.path.join(plates_dir, fname)
        img = Image.open(img_path)

        if fname in annotations:
            x1, y1, x2, y2 = annotations[fname]
            cropped = img.crop((x1, y1, x2, y2))
        else:
            cropped = img

        cv2.imshow("Plate", cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR))
        #cv2.waitKey(0)

        try:
            raw, cleaned = read_license_plate(cropped)
            print(f"{fname}: raw='{raw}' -> cleaned='{cleaned}'")
        except Exception as e:
            print(f"{fname}: ERROR - {e}")

    cv2.destroyAllWindows()