import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import csv
import math


def imread_any(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)
    return p

def disk_kernel(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), np.uint8)
    d = 2*radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    k = (x*x + y*y) <= radius*radius
    return k.astype(np.uint8)



def segment_tissue_binary(
    img_bgr: np.ndarray,
    min_component_area_ratio: float = 0.0005,
    open_radius: int = 2,
    close_radius: int = 3,
) -> np.ndarray:

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    S_blur = cv2.GaussianBlur(S, (0, 0), 1.0)

    _, tissue = cv2.threshold(S_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    bright_lowS_bg = cv2.inRange(S, 0, 20) & cv2.inRange(V, 200, 255)
    tissue[bright_lowS_bg > 0] = 0


    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, disk_kernel(open_radius))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, disk_kernel(close_radius))


    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tissue, connectivity=8)
    h, w = tissue.shape[:2]
    min_area = int(min_component_area_ratio * (h * w))

    keep = np.zeros(num_labels, dtype=bool)
    for i in range(1, num_labels):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area

    cleaned = np.zeros_like(tissue)
    mask_keep = np.isin(labels, np.where(keep)[0])
    cleaned[mask_keep] = 255

    final_binary = cv2.bitwise_not(cleaned)
    return final_binary


def save_side_by_side_preview(orig_bgr: np.ndarray, mask: np.ndarray, out_path: Path, max_w: int = 1400):
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    combo = np.hstack([orig_rgb, mask_rgb])

    h, w = combo.shape[:2]
    if w > max_w:
        scale = max_w / w
        combo = cv2.resize(combo, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    Image.fromarray(combo).save(out_path)

def make_contact_sheets(img_paths, preview_dir: Path, out_dir: Path, cols: int = 5, rows: int = 4):
    thumbs = []
    for p in img_paths:
        prev = preview_dir / (p.stem + "_preview.png")
        if prev.exists():
            thumbs.append(prev)

    if not thumbs:
        return

    images = [Image.open(t) for t in thumbs]
    target_h = 200
    norm = []
    for im in images:
        w, h = im.size
        scale = target_h / h
        im = im.resize((int(w*scale), target_h), Image.LANCZOS)
        norm.append(im)

    per_sheet = cols * rows
    sheet_index = 1
    for i in range(0, len(norm), per_sheet):
        page = norm[i:i+per_sheet]
        if not page:
            break

        cell_w = max(im.size[0] for im in page)
        cell_h = max(im.size[1] for im in page)
        sheet = Image.new("RGB", (cols*cell_w, rows*cell_h), (240, 240, 240))

        for k, im in enumerate(page):
            r = k // cols
            c = k % cols
            x = c * cell_w + (cell_w - im.size[0])//2
            y = r * cell_h + (cell_h - im.size[1])//2
            sheet.paste(im, (x, y))

        ensure_dir(out_dir)
        sheet.save(out_dir / f"contact_sheet_{sheet_index:03d}.png")
        sheet_index += 1


def process_one(
    in_path: Path,
    out_mask_dir: Path,
    out_preview_dir: Path,
    params
):
    try:
        img = imread_any(in_path)
        mask = segment_tissue_binary(
            img,
            min_component_area_ratio=params.min_component_area_ratio,
            open_radius=params.open_radius,
            close_radius=params.close_radius
        )

        mask_path = out_mask_dir / f"{in_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        preview_path = None
        if params.make_previews:
            preview_path = out_preview_dir / f"{in_path.stem}_preview.png"
            save_side_by_side_preview(img, mask, preview_path)

        tissue_px = int(np.count_nonzero(mask == 0))
        background_px = int(np.count_nonzero(mask == 255))
        h, w = mask.shape[:2]
        return {
            "file": str(in_path),
            "mask": str(mask_path),
            "preview": str(preview_path) if preview_path else "",
            "width": w,
            "height": h,
            "tissue_pixels": tissue_px,
            "background_pixels": background_px,
            "tissue_ratio": tissue_px / float(h*w)
        }
    except Exception as e:
        return {"file": str(in_path), "error": repr(e)}


def main():
    ap = argparse.ArgumentParser(description="Binary tissue segmentation (tissue=black, non-tissue=white)")
    ap.add_argument("--input", required=True, help="'OCELOT Dataset/images/test/tissue'")
    ap.add_argument("--out", default="out/masks", help="Output folder for masks and reports")
    ap.add_argument("--exts", nargs="+", default=[".png", ".jpg", ".jpeg", ".tif", ".tiff"], help="Image extensions to include")
    ap.add_argument("--open-radius", type=int, default=2, help="Morphological opening radius (remove specks)")
    ap.add_argument("--close-radius", type=int, default=3, help="Morphological closing radius (fill pinholes)")
    ap.add_argument("--min-component-area-ratio", type=float, default=0.0005, help="Drop components smaller than this fraction of image area")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers")
    ap.add_argument("--make-previews", action="store_true", help="Save side-by-side previews")
    ap.add_argument("--make-contact-sheets", action="store_true", help="Assemble previews into contact sheets")
    args = ap.parse_args()

    in_dir = Path(args.input)
    if not in_dir.exists():
        print(f"Input folder not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = ensure_dir(Path(args.out))
    out_mask_dir = ensure_dir(out_dir)
    out_preview_dir = ensure_dir(out_dir / "_previews")
    out_contacts_dir = ensure_dir(out_dir / "_contact_sheets")

    exts = {e.lower() for e in args.exts}
    img_paths = [p for p in in_dir.iterdir() if p.suffix.lower() in exts]
    img_paths.sort()

    if not img_paths:
        print("No images found. Check extensions and path.", file=sys.stderr)
        sys.exit(1)

    results = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(process_one, p, out_mask_dir, out_preview_dir, args) for p in img_paths]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                results.append(f.result())
    else:
        for p in tqdm(img_paths, desc="Processing"):
            results.append(process_one(p, out_mask_dir, out_preview_dir, args))

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file","mask","preview","width","height",
            "tissue_pixels","background_pixels","tissue_ratio","error"
        ])
        w.writeheader()
        for r in results:
            if "error" not in r:
                r["error"] = ""
            w.writerow(r)
    print(f"Wrote: {csv_path}")

    if args.make_contact_sheets and args.make_previews:
        make_contact_sheets(img_paths, out_preview_dir, out_contacts_dir, cols=5, rows=4)
        print(f"Contact sheets in: {out_contacts_dir}")

    print(f"Masks in: {out_mask_dir}")
    if args.make_previews:
        print(f"Previews in: {out_preview_dir}")

if __name__ == "__main__":
    main()
