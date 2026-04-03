# PlakaTrainer

A small three-step pipeline for building license plate character training data from images.

The project uses:
- `kareplaka.onnx` to detect license plates
- OpenCV contour detection to segment characters from each detected plate
- Tesseract OCR to place characters into per-symbol folders
- An OpenCV review tool to correct OCR mistakes with visual plate context

## Files

- `01_detect_plates.py`: runs the ONNX detector and writes plate annotations
- `02_extract_digits.py`: crops plates, finds character blobs, runs OCR, saves characters into `letters/`
- `03_review_digits.py`: review and reclassify extracted character images
- `kareplaka.onnx`: trained plate detector
- `annotations.txt`: detector output
- `unknown_plates.txt`: images where no plate was found
- `suspicious.txt`: plates flagged for unusual blob count or merged characters
- `.review_progress.json`: saved review position for resuming later

## Requirements

Python packages are listed in `requirements.txt`:
- `onnxruntime`
- `opencv-python`
- `pytesseract`
- `numpy`

You also need Tesseract OCR installed on the system.

Ubuntu/Debian:

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng
```

Python setup example:

```bash
python3 -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt
```

## Input Data

Default image directory used by the scripts:

```text
/media/ce/ce_sata/Data/Plaka/plates
```

You can override this with `--input-dir`.

## Step 1: Detect Plates

Run plate detection with the ONNX model:

```bash
python3 01_detect_plates.py
```

Useful options:

```bash
python3 01_detect_plates.py --confidence-threshold 0.3 --debug
python3 01_detect_plates.py --input-dir /path/to/images --model-path kareplaka.onnx --output-dir .
```

Output:
- `annotations.txt`
- `unknown_plates.txt`

`annotations.txt` format:

```text
filename x1 y1 x2 y2 confidence class_id
```

## Step 2: Extract Characters

Run character extraction and OCR classification:

```bash
python3 02_extract_digits.py
```

Useful options:

```bash
python3 02_extract_digits.py --annotations annotations.txt --input-dir /path/to/images --output-dir .
```

What it does:
- reads plate boxes from `annotations.txt`
- crops each plate
- enhances contrast with CLAHE
- finds character contours with adaptive thresholding
- auto-splits likely merged characters
- runs Tesseract on each character crop
- saves each crop into a folder under `letters/`

Output structure:

```text
letters/
  A/
  B/
  ...
  0/
  1/
  ...
  unknown/
suspicious.txt
```

Notes:
- `unknown/` contains characters Tesseract could not classify confidently
- `suspicious.txt` contains plates with blob count outside `6..8` or auto-split merged characters
- `TESSDATA_PREFIX` is set in the script to `/usr/share/tesseract-ocr/5/tessdata`

## Step 3: Review And Correct Characters

Run the review tool:

```bash
python3 03_review_digits.py
```

Useful options:

```bash
python3 03_review_digits.py --letters-dir letters --annotations annotations.txt --input-dir /path/to/images
```

The review window shows:
- the current class folder and image index
- the original plate crop
- a rectangle around the current character on the plate
- all detected character positions with indexes
- the enlarged character crop

### Review Controls

- `Space` or `Enter`: accept current classification and go to next image
- `A-Z` or `0-9`: move the current image to that character folder
- `d`: delete current image
- `s`: move current image to `letters/review/`
- `n`: next folder
- `p`: previous folder
- `Left Arrow`: previous image
- `Right Arrow`: next image
- `q`: quit

### Resume Support

The review tool saves your last position to:

```text
.review_progress.json
```

When you start the tool again, it resumes from the saved folder and file if that item still exists.

## Typical Workflow

```bash
python3 01_detect_plates.py --confidence-threshold 0.3
python3 02_extract_digits.py
python3 03_review_digits.py
```

## Current Output Artifacts

After a full run, the project may contain:
- `annotations.txt`
- `unknown_plates.txt`
- `suspicious.txt`
- `letters/` with per-character folders
- `.review_progress.json`

## Limitations

- OCR quality depends heavily on plate crop quality and character size
- Tiny or blurry characters often end up in `letters/unknown/`
- Character segmentation is contour-based, so unusual plate layouts may need manual review
- Foreign plates and non-standard glyphs may not classify well with the current OCR whitelist

## Tips

- Lower `--confidence-threshold` in step 1 if plates are being missed
- Review `unknown/` first to improve the dataset fastest
- Review `suspicious.txt` when segmentation quality looks wrong
- If you regenerate `letters/`, old review progress may point to a file that no longer exists
