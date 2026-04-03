#!/usr/bin/env python3
"""
03_review_digits.py — Visual review tool for classified character images.

Browses the letters/ folder structure using OpenCV imshow.
Allows reclassifying, deleting, or skipping character images.

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
    python 03_review_digits.py --letters-dir letters/
"""

import argparse
import os
import sys
import shutil

import cv2
import numpy as np


def gather_images(letters_dir):
    """Gather all images grouped by folder.
    Returns list of (folder_name, [file_paths])."""
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


def display_image(img, folder_name, img_idx, total, filepath):
    """Display image enlarged with info in window title."""
    h, w = img.shape[:2]
    scale = max(1, 200 // max(w, 1), 300 // max(h, 1))
    scale = min(scale, 10)
    display = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    # Info bar at top
    bar_h = 40
    bar = 195 * np.ones((bar_h, display.shape[1], 3), dtype=np.uint8)
    fname = os.path.basename(filepath)
    if len(fname) > 50:
        fname = fname[:47] + "..."
    cv2.putText(bar, f"[{folder_name}] {img_idx+1}/{total} | {fname}",
                (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    display = np.vstack([bar, display])

    # Help bar at bottom
    help_h = 30
    help_bar = 195 * np.ones((help_h, display.shape[1], 3), dtype=np.uint8)
    cv2.putText(help_bar, "Space:next  d:del  s:skip  A-Z/0-9:reclassify  n/p:folder  q:quit",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
    display = np.vstack([display, help_bar])

    window_name = f"Review: [{folder_name}] {img_idx+1}/{total}"
    cv2.imshow(window_name, display)
    return window_name


def move_to_folder(filepath, letters_dir, target_char):
    """Move file to target character folder. Creates folder if needed."""
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


def main():
    parser = argparse.ArgumentParser(description="Review and reclassify extracted character images")
    parser.add_argument("--letters-dir", default="letters",
                        help="Path to the letters/ directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    letters_dir = args.letters_dir if os.path.isabs(args.letters_dir) else os.path.join(script_dir, args.letters_dir)

    if not os.path.isdir(letters_dir):
        print(f"Error: {letters_dir} not found. Run 02_extract_digits.py first.")
        sys.exit(1)

    folders = gather_images(letters_dir)
    if not folders:
        print("No character images found.")
        sys.exit(0)

    total_folders = len(folders)
    folder_idx = 0
    img_idx = 0
    prev_window = None

    print(f"Found {total_folders} character folders")
    print("Controls: Space/Enter=next, d=delete, s=skip, A-Z/0-9=reclassify, n/p=folder, q=quit")

    while True:
        # Refresh folder data (files may have been moved/deleted)
        folders = gather_images(letters_dir)
        if not folders:
            print("No more images to review.")
            break
        total_folders = len(folders)
        folder_idx = min(folder_idx, total_folders - 1)

        folder_name, images = folders[folder_idx]
        if not images:
            folder_idx = (folder_idx + 1) % total_folders
            continue
        img_idx = min(img_idx, len(images) - 1)

        filepath = images[img_idx]
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  Cannot read: {filepath}")
            img_idx += 1
            if img_idx >= len(images):
                img_idx = 0
                folder_idx = (folder_idx + 1) % total_folders
            continue

        # Close previous window if title changed
        window_name = f"Review: [{folder_name}] {img_idx+1}/{len(images)}"
        if prev_window and prev_window != window_name:
            cv2.destroyWindow(prev_window)
        prev_window = window_name

        display_image(img, folder_name, img_idx, len(images), filepath)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break

        elif key in (13, 32):  # Enter or Space — accept, next
            img_idx += 1
            if img_idx >= len(images):
                img_idx = 0
                folder_idx = (folder_idx + 1) % total_folders

        elif key == ord('d'):  # Delete
            os.remove(filepath)
            print(f"  Deleted: {os.path.basename(filepath)}")
            if img_idx >= len(images) - 1:
                img_idx = max(0, len(images) - 2)

        elif key == ord('s'):  # Skip / mark for review
            move_to_folder(filepath, letters_dir, "review")
            print(f"  Moved to review/: {os.path.basename(filepath)}")
            if img_idx >= len(images) - 1:
                img_idx = max(0, len(images) - 2)

        elif key == ord('n'):  # Next folder
            folder_idx = (folder_idx + 1) % total_folders
            img_idx = 0

        elif key == ord('p'):  # Previous folder
            folder_idx = (folder_idx - 1) % total_folders
            img_idx = 0

        elif key == 81:  # Left arrow
            img_idx = max(0, img_idx - 1)

        elif key == 83:  # Right arrow
            img_idx += 1
            if img_idx >= len(images):
                img_idx = len(images) - 1

        else:
            # Check if typed A-Z or 0-9 for reclassification
            char = chr(key).upper() if 0 <= key < 128 else ''
            if char and char.isalnum() and len(char) == 1:
                if char != folder_name:
                    move_to_folder(filepath, letters_dir, char)
                    print(f"  Reclassified '{folder_name}' -> '{char}': {os.path.basename(filepath)}")
                    if img_idx >= len(images) - 1:
                        img_idx = max(0, len(images) - 2)
                else:
                    # Same folder, just advance
                    img_idx += 1
                    if img_idx >= len(images):
                        img_idx = 0
                        folder_idx = (folder_idx + 1) % total_folders

    cv2.destroyAllWindows()
    print("Review complete.")


if __name__ == "__main__":
    main()
