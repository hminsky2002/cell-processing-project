import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def process_cell_binary(image_path, annotations: pd.DataFrame, image=None):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    intensity_threshold = 0.35 * 255
    binary_img = (image < intensity_threshold).astype('uint8')

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

    min_area = 100
    filtered_labels = np.zeros_like(labels)
    valid_label = 1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if min_area <= area:
            filtered_labels[labels == label] = valid_label
            valid_label += 1

    detected_count = valid_label - 1
    actual_count = len(annotations)

    print(f"Image: {image_path.name}, Detected: {detected_count}, Actual: {actual_count}")

    output_path = results_dir / f"{image_path.stem}_1_result.png"
    cv2.imwrite(str(output_path), filtered_labels)


    original_color = cv2.imread(str(image_path))
    overlay = original_color.copy()
    mask = (filtered_labels > 0).astype('uint8') * 255
    overlay[mask > 0] = [0, 0, 255]

    overlay_path = results_dir / f"{image_path.stem}_2_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    original_path = results_dir / f"{image_path.stem}_3_original.png"
    cv2.imwrite(str(original_path), original_color)

    return {
        'image_name': image_path.name,
        'detected_components': detected_count,
        'actual_components': actual_count,
        'difference': abs(detected_count - actual_count)
    }
