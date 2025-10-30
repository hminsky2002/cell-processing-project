import cv2
import numpy as np
import pandas as pd
from pathlib import Path


def disk_kernel(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), np.uint8)
    d = 2*radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    k = (x*x + y*y) <= radius*radius
    return k.astype(np.uint8)


def process_tissue_binary(image_path, annotations, image=None):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    S_blur = cv2.GaussianBlur(S, (0, 0), 1.0)
    _, tissue = cv2.threshold(S_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bright_lowS_bg = cv2.inRange(S, 0, 20) & cv2.inRange(V, 200, 255)
    tissue[bright_lowS_bg > 0] = 0

    open_radius = 2
    close_radius = 3
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, disk_kernel(open_radius))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, disk_kernel(close_radius))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tissue, connectivity=8)
    h, w = tissue.shape[:2]
    min_area = int(0.0005 * (h * w))
    
    filtered_labels = np.zeros_like(labels)
    valid_label = 1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_labels[labels == label] = valid_label
            valid_label += 1

    final_binary = np.where(filtered_labels > 0, 255, 0).astype('uint8')
    final_binary = cv2.bitwise_not(final_binary)

    output_path = results_dir / f"{image_path.stem}_1_result.png"
    cv2.imwrite(str(output_path), final_binary)

    overlay = image.copy()

    detected_tissue_mask = (final_binary == 0).astype('uint8') * 255
    overlay[detected_tissue_mask > 0] = [0, 0, 255]

    print('generated final tissue binary for image: ', image_path.name)
    if annotations is not None and isinstance(annotations, np.ndarray):
        annotation_mask = (annotations == 0).astype('uint8') * 255
        overlay[annotation_mask > 0] = [0, 255, 0]

    overlay_path = results_dir / f"{image_path.stem}_2_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    original_path = results_dir / f"{image_path.stem}_3_original.png"
    cv2.imwrite(str(original_path), image)

    if annotations is not None and isinstance(annotations, np.ndarray):
        annotation_path = results_dir / f"{image_path.stem}_4_annotation.png"
        cv2.imwrite(str(annotation_path), annotations)

    return None
