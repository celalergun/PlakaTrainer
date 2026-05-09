#!/usr/bin/env python3
"""
debug_seg.py — visual step-by-step debug of the segment_chars pipeline.

Usage:
    python debug_seg.py <image_path>

Writes every intermediate stage as a JPEG to ./debug_seg/.
"""

import sys
import os
import cv2
import numpy as np

OUT = "./debug_seg"
os.makedirs(OUT, exist_ok=True)


def save(name: str, img: np.ndarray):
    path = os.path.join(OUT, name)
    if img.ndim == 2:
        cv2.imwrite(path, img)
    else:
        cv2.imwrite(path, img)
    print(f"  saved: {path}  shape={img.shape}")


def draw_rect(img_bgr, rx, ry, rw, rh, color=(0, 255, 0), thickness=2):
    vis = img_bgr.copy()
    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), color, thickness)
    return vis


def draw_boxes(img_bgr, boxes, color=(0, 200, 255)):
    vis = img_bgr.copy()
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    return vis


def hist_image(col_hist: np.ndarray, width: int, height: int = 120) -> np.ndarray:
    """Render column histogram as a grayscale image (black bg, white bars)."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    max_v = col_hist.max() if col_hist.max() > 0 else 1
    for x, v in enumerate(col_hist[:width]):
        bar_h = int(v / max_v * (height - 2))
        if bar_h > 0:
            cv2.line(canvas, (x, height - 1), (x, height - 1 - bar_h), 255, 1)
    return canvas


def main(image_path: str):
    # ── Load ──────────────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        sys.exit(1)

    print(f"\nImage: {image_path}  size={img.shape[1]}×{img.shape[0]}")
    save("00_original.jpg", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape
    if iw < 300:
        scale = 300 / iw
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        img  = cv2.resize(img,  None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ih, iw = gray.shape
    save("01_gray.jpg", gray)

    # ── Step 1: find inner rect (flood-fill-from-outside) ────────────────────
    blurred = cv2.GaussianBlur(gray, (0, 0),
                                sigmaX=max(5, iw // 20), sigmaY=max(5, ih // 7))
    save("02_heavy_blur.jpg", blurred)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save("03_thresh_raw.jpg", thresh)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    best_rect = None
    best_area = 0
    best_interior = None
    for pol_label, polarity in (("normal", thresh), ("inverted", 255 - thresh)):
        closed = cv2.morphologyEx(polarity, cv2.MORPH_CLOSE, kernel_close)
        padded = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        ph, pw = padded.shape
        n, labels, stats, _ = cv2.connectedComponentsWithStats(padded, connectivity=8)
        exterior = (set(labels[0, :].tolist()) | set(labels[ph-1, :].tolist())
                    | set(labels[:, 0].tolist()) | set(labels[:, pw-1].tolist()))
        for i in range(1, n):
            if i in exterior:
                continue
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < iw * ih * 0.10:
                continue
            if area > best_area:
                best_area = area
                best_rect = (
                    int(stats[i, cv2.CC_STAT_LEFT]) - 1,
                    int(stats[i, cv2.CC_STAT_TOP]) - 1,
                    int(stats[i, cv2.CC_STAT_WIDTH]),
                    int(stats[i, cv2.CC_STAT_HEIGHT]),
                )
                # build interior mask for visualisation
                interior_mask = np.zeros((ih, iw), dtype=np.uint8)
                interior_mask[labels[1:-1, 1:-1] == i] = 255
                best_interior = interior_mask
                print(f"  polarity={pol_label}  interior blob area={area}  rect={best_rect}")

    save("04_thresh_polarity.jpg", thresh)
    if best_interior is not None:
        save("05_filled.jpg", best_interior)
    else:
        save("05_filled.jpg", thresh)

    if best_rect is None:
        print("  No interior blob found — falling back to full image")
        rx, ry, rw, rh = 0, 0, iw, ih
    else:
        rx, ry, rw, rh = best_rect

    vis_contours = img.copy()
    cv2.rectangle(vis_contours, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
    save("06_inner_contours.jpg", vis_contours)

    print(f"  raw inner rect: x={rx} y={ry} w={rw} h={rh}")
    save("07_inner_rect_raw.jpg", draw_rect(img, rx, ry, rw, rh, (0, 255, 0)))

    mx = max(2, rw // 20)
    my = max(2, rh // 12)
    rx = min(rx + mx, iw - 1)
    ry = min(ry + my, ih - 1)
    rw = max(10, rw - 2 * mx)
    rh = max(10, rh - 2 * my)
    print(f"  inset inner rect: x={rx} y={ry} w={rw} h={rh}")
    save("08_inner_rect_inset.jpg", draw_rect(img, rx, ry, rw, rh, (0, 255, 255)))

    # ── Step 2: binarise crop ─────────────────────────────────────────────────
    crop_gray = cv2.GaussianBlur(gray[ry: ry + rh, rx: rx + rw], (3, 3), 0)
    save("09_crop_gray.jpg", crop_gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(crop_gray)
    save("10_crop_clahe.jpg", enhanced)

    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_mean = otsu.mean()
    print(f"  otsu mean={otsu_mean:.1f}  \u2192 {'background=255, chars=0 (invert)' if otsu_mean > 127 else 'background=0, chars=255 (keep)'}")
    binary = (255 - otsu) if otsu_mean > 127 else otsu
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    save("11_binary_crop.jpg", binary)

    # ── Step 3: column histogram ──────────────────────────────────────────────
    col_hist = binary.sum(axis=0).astype(float)
    hist_img = hist_image(col_hist, rw)
    save("12_col_histogram.jpg", hist_img)
    print(f"  histogram max={col_hist.max():.0f}  mean={col_hist.mean():.1f}")

    min_gap = max(2, rw // 40)

    # ── Step 4: contour pass on crop ─────────────────────────────────────────
    dil_k   = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, dil_k, iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  char contours found: {len(cnts)}")

    raw = []
    boxes_from_wide = []
    for cnt in cnts:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        reason = None
        if bw < 4 or bh < 4:
            reason = "too small"
        elif bh / rh < 0.25:
            reason = f"too short (bh/rh={bh/rh:.2f})"
        elif bw / bh > 4.0 or bw > rw * 0.65:
            if bh / rh >= 0.5:
                reason = f"wide+tall (aspect={bw/bh:.2f}) → histogram split"
                sub_h = binary[:, bx: bx + bw].sum(axis=0).astype(float)
                for thr2 in (0.05, 0.08, 0.12, 0.18, 0.20, 0.22, 0.25):
                    max_s = sub_h.max() if sub_h.max() > 0 else 1
                    valley_s = sub_s = sub_h <= max_s * thr2
                    sg, in_gs, gss = [], False, 0
                    for ci, v in enumerate(valley_s):
                        if v and not in_gs: gss, in_gs = ci, True
                        elif not v and in_gs:
                            if ci - gss >= min_gap: sg.append((gss, ci))
                            in_gs = False
                    if in_gs and len(valley_s) - gss >= min_gap: sg.append((gss, len(valley_s)))
                    starts2 = [0] + [g[1] for g in sg]
                    ends2   = [g[0] for g in sg] + [bw]
                    fb2 = []
                    for xs, xe in zip(starts2, ends2):
                        sw = xe - xs
                        if sw >= 4 and 0.08 <= sw / bh <= 1.8:
                            fb2.append({"x": rx + bx + xs, "y": ry + by, "w": sw, "h": bh, "char": ""})
                    if 2 <= len(fb2) <= 14:
                        print(f"    WIDE SPLIT thr={thr2:.0%} → {len(fb2)} boxes")
                        boxes_from_wide.extend(fb2)
                        break
                else:
                    reason = f"wide+tall (aspect={bw/bh:.2f}) → no split found"
            else:
                reason = f"too wide (aspect={bw/bh:.2f})"
        if reason and "→ histogram split" not in reason:
            print(f"    REJECTED x={bx} w={bw} h={bh}: {reason}")
        elif reason is None:
            print(f"    KEPT     x={bx} w={bw} h={bh} aspect={bw/bh:.2f}")
            raw.append((bx, by, bw, bh))

    vis_char_cnts = img[ry: ry + rh, rx: rx + rw].copy() if img[ry: ry + rh, rx: rx + rw].size > 0 else img.copy()
    for cnt in cnts:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(vis_char_cnts, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
    save("13_char_contours.jpg", vis_char_cnts)

    # ── Step 5: histogram gaps ────────────────────────────────────────────────
    for thr in (0.05, 0.08, 0.12, 0.18, 0.20, 0.22, 0.25):
        max_v = col_hist.max() if col_hist.max() > 0 else 1
        valley = col_hist <= max_v * thr
        gap_count = 0
        in_gap = False
        for v in valley:
            if v and not in_gap:
                in_gap = True
            elif not v and in_gap:
                gap_count += 1
                in_gap = False
        seg_count = gap_count + 1
        print(f"  hist thr={thr:.0%}: gaps={gap_count}  segments={seg_count}")

    # ── Step 6: final boxes on original ──────────────────────────────────────
    # (reproduce the same logic as segment_chars to show the final result)
    ind_boxes = []
    single_ws = [bw for _, _, bw, bh in raw if 0.25 <= bw / bh <= 1.1]
    ref_w = float(np.median(single_ws)) if single_ws else None
    for bx, by, bw, bh in raw:
        char_w = ref_w if ref_w else 0.7 * bh
        if bw > char_w * 1.6:
            sub_hist = binary[:, bx: bx + bw].sum(axis=0).astype(float)
            max_v2 = sub_hist.max() if sub_hist.max() > 0 else 1
            valley2 = sub_hist <= max_v2 * 0.08
            # simple gap scan
            sub_gaps = []
            in_g, gs = False, 0
            for ci, v in enumerate(valley2):
                if v and not in_g:
                    gs, in_g = ci, True
                elif not v and in_g:
                    if ci - gs >= min_gap:
                        sub_gaps.append((gs, ci))
                    in_g = False
            if in_g and len(valley2) - gs >= min_gap:
                sub_gaps.append((gs, len(valley2)))
            if len(sub_gaps) >= 1:
                starts = [0] + [g[1] for g in sub_gaps]
                ends   = [g[0] for g in sub_gaps] + [bw]
                for xs, xe in zip(starts, ends):
                    sw = xe - xs
                    if sw >= 4 and 0.08 <= sw / bh <= 1.8:
                        ind_boxes.append({"x": rx + bx + xs, "y": ry + by, "w": sw, "h": bh, "char": ""})
                continue
        ind_boxes.append({"x": rx + bx, "y": ry + by, "w": bw, "h": bh, "char": ""})

    # Prefer individual contour boxes; use wide-blob splits only as fallback
    if 2 <= len(ind_boxes) <= 14:
        boxes = ind_boxes
        print(f"  using individual contour boxes ({len(boxes)})")
    elif 2 <= len(boxes_from_wide) <= 14:
        boxes = boxes_from_wide
        print(f"  using wide-blob split boxes ({len(boxes)})")
    else:
        boxes = ind_boxes

    if not (2 <= len(boxes) <= 14):
        for thr in (0.05, 0.08, 0.12, 0.18, 0.20, 0.22, 0.25):
            max_v = col_hist.max() if col_hist.max() > 0 else 1
            valley = col_hist <= max_v * thr
            gaps, in_g, gs = [], False, 0
            for ci, v in enumerate(valley):
                if v and not in_g:
                    gs, in_g = ci, True
                elif not v and in_g:
                    if ci - gs >= min_gap:
                        gaps.append((gs, ci))
                    in_g = False
            if in_g and len(valley) - gs >= min_gap:
                gaps.append((gs, len(valley)))
            starts = [0] + [g[1] for g in gaps]
            ends   = [g[0] for g in gaps] + [rw]
            fb = []
            for xs, xe in zip(starts, ends):
                bw2 = xe - xs
                if bw2 >= 4 and 0.08 <= bw2 / rh <= 1.8:
                    fb.append({"x": rx + xs, "y": ry, "w": bw2, "h": rh, "char": ""})
            if 2 <= len(fb) <= 14:
                boxes = fb
                print(f"  histogram fallback thr={thr:.0%} → {len(fb)} boxes")
                break

    print(f"\n  Final boxes: {len(boxes)}")
    for b in boxes:
        print(f"    x={b['x']} y={b['y']} w={b['w']} h={b['h']}")

    save("14_final_boxes.jpg", draw_boxes(img, boxes))
    print(f"\nAll debug images written to {OUT}/")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/ce/Proje/PlakaTrainer/deskewed/01NB455-20190615-145747.jpg"
    main(path)
