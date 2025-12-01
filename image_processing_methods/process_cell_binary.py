import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def process_cell_binary(image_path, annotations: pd.DataFrame, image=None):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    binary_img = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,
        C=10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    binary_img = (binary_img > 0).astype('uint8')

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    min_area = 50
    max_area = 500

    filtered_labels = np.zeros_like(labels)
    detected_centroids = []
    valid_label = 1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if min_area <= area <= max_area:
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]

            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            bbox_area = w * h
            compactness = area / bbox_area if bbox_area > 0 else 0

            if aspect_ratio < 3.0 and compactness > 0.3:
                filtered_labels[labels == label] = valid_label
                detected_centroids.append(centroids[label])
                valid_label += 1

    detected_count = valid_label - 1
    actual_count = len(annotations)

    print(f"Image: {image_path.name}, Detected: {detected_count}, Actual: {actual_count}")

    binary_vis = (binary_img * 255).astype('uint8')
    binary_path = results_dir / f"{image_path.stem}_cell_binary_1_threshold.png"
    cv2.imwrite(str(binary_path), binary_vis)

    output_path = results_dir / f"{image_path.stem}_cell_binary_2_filtered.png"
    filtered_vis = (filtered_labels > 0).astype('uint8') * 255
    cv2.imwrite(str(output_path), filtered_vis)

    original_color = cv2.imread(str(image_path))
    overlay = original_color.copy()

    for centroid in detected_centroids:
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), 2)

    if annotations is not None and len(annotations) > 0:
        for _, row in annotations.iterrows():
            cx, cy = int(row['x']), int(row['y'])
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), 2)

    overlay_path = results_dir / f"{image_path.stem}_cell_binary_3_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    original_path = results_dir / f"{image_path.stem}_cell_binary_4_original.png"
    cv2.imwrite(str(original_path), original_color)

    return {
        'image_name': image_path.name,
        'detected_components': detected_count,
        'actual_components': actual_count,
        'difference': abs(detected_count - actual_count)
    }
