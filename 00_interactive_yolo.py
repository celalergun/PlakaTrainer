import os
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

import ast
import cv2
import numpy as np
import onnxruntime as ort
import glob

# --- Configuration ---
IMAGE_FOLDER = '/media/ce/ce_sata/Data/Plaka/plates' 
ONNX_MODEL_PATH = '/media/ce/ce_sata/Proje/PlakaTrainer/kareplaka.onnx' # IMPORTANT: Set this path

# File to save annotations (image_path, x, y, w, h, class_id)
ANNOTATIONS_FILE = 'annotations.csv'

# Thresholds for the YOLO model
CONFIDENCE_THRESHOLD = 0.5 # Filter out weak detections
AUTO_ACCEPT_CONFIDENCE = 0.75

DEFAULT_CLASS_NAMES = {
    0: 'uzunplaka',
    1: 'kareplaka',
}

CLASS_NAMES = DEFAULT_CLASS_NAMES.copy()

# --- YOLOv26 Prediction Function ---
def letterbox(img, new_shape=(640, 640)):
    """Resize and pad image to new_shape maintaining aspect ratio."""
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


def get_class_name(class_id):
    return CLASS_NAMES.get(class_id, f'class_{class_id}')


def load_class_names(session):
    metadata = session.get_modelmeta().custom_metadata_map
    raw_names = metadata.get('names')
    if not raw_names:
        return DEFAULT_CLASS_NAMES.copy()

    try:
        parsed_names = ast.literal_eval(raw_names)
    except (ValueError, SyntaxError):
        return DEFAULT_CLASS_NAMES.copy()

    if not isinstance(parsed_names, dict):
        return DEFAULT_CLASS_NAMES.copy()

    class_names = {}
    for class_id, class_name in parsed_names.items():
        try:
            class_names[int(class_id)] = str(class_name)
        except (TypeError, ValueError):
            continue

    return class_names or DEFAULT_CLASS_NAMES.copy()


def render_preview(image, box, class_id, confidence=None):
    preview = image.copy()
    if box:
        x, y, w, h = box
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = get_class_name(class_id)
        if confidence is not None:
            label = f"{label} ({confidence:.2f})"
        cv2.putText(preview, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return preview


def render_zoomed_crop(image, box, scale=2):
    if not box:
        preview = np.zeros((160, 320, 3), dtype=np.uint8)
        cv2.putText(preview, 'No detection', (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return preview

    x, y, w, h = box
    img_h, img_w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    if x1 >= x2 or y1 >= y2:
        preview = np.zeros((160, 320, 3), dtype=np.uint8)
        cv2.putText(preview, 'Invalid crop', (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return preview

    crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_LINEAR)


def show_previews(image, clone, box):
    cv2.imshow("Annotator", clone)
    cv2.imshow("Detected Plate", render_zoomed_crop(image, box))


def predict_with_yolo(session, image, confidence_thresh, nms_thresh):
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    input_size = (input_shape[2], input_shape[3])

    # Preprocess: letterbox + BGR->RGB + normalize + CHW + batch
    img_lb, ratio, (dw, dh) = letterbox(image, input_size)
    blob = img_lb[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)

    # Inference — YOLOv26 end-to-end output: [1, 300, 6]
    outputs = session.run(None, {input_name: blob})
    detections = outputs[0][0]  # shape: [300, 6]

    # Find the best detection above threshold
    best = None
    best_conf = 0.0
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < confidence_thresh:
            continue
        if conf > best_conf:
            best_conf = conf
            # Map back to original image coordinates
            bx1 = (x1 - dw) / ratio
            by1 = (y1 - dh) / ratio
            bx2 = (x2 - dw) / ratio
            by2 = (y2 - dh) / ratio
            best = {
                'box': (int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1)),
                'class_id': int(cls_id),
                'confidence': float(conf),
            }

    return best

# --- Main Annotation Loop ---
if __name__ == "__main__":
    # Check if files and folders exist
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Error: ONNX model not found at '{ONNX_MODEL_PATH}'")
        exit()
    image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, '*.jpg')))
    if not image_paths:
        print(f"No .jpg images found in '{IMAGE_FOLDER}'. Please check the path.")
        exit()

    # Load the ONNX model into an inference session
    print(f"Loading YOLO model from {ONNX_MODEL_PATH}...")
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    CLASS_NAMES = load_class_names(ort_session)
    print("Model loaded successfully.")
    print(f"Detected classes: {', '.join(CLASS_NAMES[class_id] for class_id in sorted(CLASS_NAMES))}")
    
    annotations = []
    
    print("\n--- Starting YOLO-Assisted Annotation ---")
    print("  'y' -> Accept current box and class")
    print("  'c' -> Draw or replace the box")
    print("  't' -> Toggle class")
    print("  's' -> Skip this image")
    print("  'q' -> Quit and Save")
    print("------------------------------------------")

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            continue
        
        # --- Prediction Step ---
        print(f"\n> Image: {os.path.basename(image_path)} (Predicting...)")
        prediction = predict_with_yolo(ort_session, img, CONFIDENCE_THRESHOLD, None)
        
        if prediction:
            current_box = prediction['box']
            selected_class_id = prediction['class_id']
            current_confidence = prediction['confidence']
            print(f"  Predicted class: {get_class_name(selected_class_id)} ({current_confidence:.2f})")
        else:
            current_box = None
            selected_class_id = 0
            current_confidence = None
            print("  No license plate found by the model.")

        clone = render_preview(img, current_box, selected_class_id, current_confidence)
        show_previews(img, clone, current_box)

        if current_box and current_confidence is not None and current_confidence >= AUTO_ACCEPT_CONFIDENCE:
            x, y, w, h = current_box
            annotation_line = f"{image_path},{x},{y},{w},{h},{selected_class_id}\n"
            annotations.append(annotation_line)
            print(
                f"  Auto-accepted for {os.path.basename(image_path)} as "
                f"{get_class_name(selected_class_id)} ({current_confidence:.2f})"
            )
            continue

        # --- User Interaction Step ---
        final_box = None
        final_class_id = selected_class_id
        while True:
            show_previews(img, clone, current_box)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('y'): # Accept
                if current_box:
                    final_box = current_box
                    final_class_id = selected_class_id
                    break
                else:
                    print("No box to accept. Please draw a box ('c').")
            
            elif key == ord('c'): # Correct / Manual
                print("Draw a box around the license plate, then press ENTER or SPACE.")
                roi = cv2.selectROI("Annotator", img, fromCenter=False, showCrosshair=True)
                if roi[2] > 0 and roi[3] > 0:
                    current_box = tuple(map(int, roi))
                    current_confidence = None
                    clone = render_preview(img, current_box, selected_class_id, current_confidence)
                    show_previews(img, clone, current_box)
                    print(f"  Updated box. Current class: {get_class_name(selected_class_id)}")

            elif key == ord('t'):
                selected_class_id = (selected_class_id + 1) % len(CLASS_NAMES)
                clone = render_preview(img, current_box, selected_class_id, current_confidence)
                show_previews(img, clone, current_box)
                print(f"  Current class: {get_class_name(selected_class_id)}")

            elif key == ord('s'): # Skip
                final_box = None
                break

            elif key == ord('q'): # Quit
                cv2.destroyAllWindows()
                # Save any annotations collected so far before quitting
                if annotations:
                    with open(ANNOTATIONS_FILE, 'a') as f:
                        f.writelines(annotations)
                    print(f"\nSession ended. Saved {len(annotations)} new annotations to '{ANNOTATIONS_FILE}'.")
                exit()
        
        # --- Save Confirmed Annotation ---
        if final_box:
            x, y, w, h = final_box
            annotation_line = f"{image_path},{x},{y},{w},{h},{final_class_id}\n"
            annotations.append(annotation_line)
            print(f"  Annotation saved for {os.path.basename(image_path)} as {get_class_name(final_class_id)}")

    # --- Final Save ---
    if annotations:
        with open(ANNOTATIONS_FILE, 'a') as f:
            f.writelines(annotations)
        print(f"\nAnnotation session complete. Saved {len(annotations)} new annotations to '{ANNOTATIONS_FILE}'.")

    cv2.destroyAllWindows()