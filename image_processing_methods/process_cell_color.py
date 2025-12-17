import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from project_utils import calculate_detection_metrics


ORGAN_COLOR_RANGES = {
    'bladder': {
        'hsv_lower': np.array([130, 90, 50]),
        'hsv_upper': np.array([157, 180, 170]),
        'min_area': 200,
        'max_area': 2000,
    },
    'endometrium': {
        'hsv_lower': np.array([133, 70, 70]),
        'hsv_upper': np.array([169, 175, 200]),
        'min_area': 200,
        'max_area': 2000,
    },
    'head-and-neck': {
        'hsv_lower': np.array([120, 70, 65]),
        'hsv_upper': np.array([165, 200, 200]),
        'min_area': 200,
        'max_area': 2000,
    },
    'kidney': {
        'hsv_lower': np.array([130, 75, 50]),
        'hsv_upper': np.array([160, 170, 175]),
        'min_area': 200,
        'max_area': 2000,
    },
    'prostate': {
        'hsv_lower': np.array([122, 40, 90]),
        'hsv_upper': np.array([158, 140, 215]),
        'min_area': 200,
        'max_area': 2000,
    },
    'stomach': {
        'hsv_lower': np.array([128, 45, 85]),
        'hsv_upper': np.array([158, 150, 210]),
        'min_area': 200,
        'max_area': 2000,
    },
    'default': {
        'hsv_lower': np.array([125, 60, 60]),
        'hsv_upper': np.array([160, 180, 200]),
        'min_area': 200,
        'max_area': 2000,
    }
}


def get_organ_from_metadata(image_path, data_root='ocelot_testing_data'):
    img_id = Path(image_path).stem
    metadata_path = Path(data_root) / 'metadata.json'
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if img_id in metadata['sample_pairs']:
            return metadata['sample_pairs'][img_id].get('organ', 'default')
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
    return 'default'


def process_cell_color(image_path, annotations: pd.DataFrame, image=None, data_root='ocelot_testing_data', distance_threshold=100):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    organ = get_organ_from_metadata(image_path, data_root)
    color_params = ORGAN_COLOR_RANGES.get(organ, ORGAN_COLOR_RANGES['default'])

    print(f"Processing {Path(image_path).name} as {organ} organ")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    color_mask = cv2.inRange(image_hsv, color_params['hsv_lower'], color_params['hsv_upper'])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    binary_img = (color_mask > 0).astype('uint8')

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    min_area = color_params['min_area']
    max_area = color_params['max_area']

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

    metrics = calculate_detection_metrics(detected_centroids, annotations, distance_threshold)

    print(f"  Organ: {organ}, Detected: {detected_count}, Actual: {actual_count}")
    print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
    print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}, Accuracy: {metrics['accuracy']:.3f}")

    mask_path = results_dir / f"{Path(image_path).stem}_cell_color_1_mask.png"
    cv2.imwrite(str(mask_path), color_mask)

    filtered_path = results_dir / f"{Path(image_path).stem}_cell_color_2_filtered.png"
    filtered_vis = (filtered_labels > 0).astype('uint8') * 255
    cv2.imwrite(str(filtered_path), filtered_vis)

    overlay = image_bgr.copy()

    for centroid in detected_centroids:
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), 2)

    if annotations is not None and len(annotations) > 0:
        for _, row in annotations.iterrows():
            cx, cy = int(row['x']), int(row['y'])
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), 2)

    cv2.putText(overlay, f"Organ: {organ}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"Detected: {detected_count} | Actual: {actual_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"P: {metrics['precision']:.2f} R: {metrics['recall']:.2f} F1: {metrics['f1_score']:.2f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    overlay_path = results_dir / f"{Path(image_path).stem}_cell_color_3_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    original_path = results_dir / f"{Path(image_path).stem}_cell_color_4_original.png"
    cv2.imwrite(str(original_path), image_bgr)

    return {
        'image_name': Path(image_path).name,
        'organ': organ,
        'detected_components': detected_count,
        'actual_components': actual_count,
        'difference': abs(detected_count - actual_count),
        **metrics
    }
