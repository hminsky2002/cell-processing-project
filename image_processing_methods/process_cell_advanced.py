import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path


ORGAN_PARAMS = {
    'bladder': {
        'hsv_lower': np.array([130, 90, 50]),
        'hsv_upper': np.array([157, 180, 170]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'endometrium': {
        'hsv_lower': np.array([133, 70, 70]),
        'hsv_upper': np.array([169, 175, 200]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'head-and-neck': {
        'hsv_lower': np.array([120, 70, 65]),
        'hsv_upper': np.array([165, 200, 200]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'kidney': {
        'hsv_lower': np.array([130, 75, 50]),
        'hsv_upper': np.array([160, 170, 175]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'prostate': {
        'hsv_lower': np.array([122, 40, 90]),
        'hsv_upper': np.array([158, 140, 215]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'stomach': {
        'hsv_lower': np.array([128, 45, 85]),
        'hsv_upper': np.array([158, 150, 210]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'default': {
        'hsv_lower': np.array([125, 60, 60]),
        'hsv_upper': np.array([160, 180, 200]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
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


def detect_cells_multi_stage(image_bgr, params):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(image_hsv, params['hsv_lower'], params['hsv_upper'])

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=8
    )

    combined_mask = cv2.bitwise_and(color_mask, adaptive_thresh)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    dist_transform = cv2.distanceTransform(combined_mask, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        sure_fg, connectivity=8
    )

    detected_centroids = []
    min_area = params['min_area']
    max_area = params['max_area']

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if min_area <= area <= max_area:
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]

            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            bbox_area = w * h
            compactness = area / bbox_area if bbox_area > 0 else 0

            if aspect_ratio < 3.0 and compactness > 0.3:
                detected_centroids.append(centroids[label])

    blob_params = cv2.SimpleBlobDetector_Params()
    blob_params.filterByArea = True
    blob_params.minArea = params['min_area']
    blob_params.maxArea = params['max_area']
    blob_params.filterByCircularity = True
    blob_params.minCircularity = 0.2
    blob_params.filterByConvexity = True
    blob_params.minConvexity = 0.2
    blob_params.filterByInertia = True
    blob_params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector_create(blob_params)
    keypoints = detector.detect(color_mask)

    for kp in keypoints:
        kp_point = np.array([kp.pt[0], kp.pt[1]])
        too_close = False
        for centroid in detected_centroids:
            dist = np.linalg.norm(kp_point - centroid)
            if dist < 10:
                too_close = True
                break
        if not too_close:
            detected_centroids.append(kp_point)

    return detected_centroids, {
        'color_mask': color_mask,
        'combined_mask': combined_mask,
        'dist_transform': (dist_transform * 255).astype(np.uint8),
        'sure_fg': sure_fg
    }


def process_cell_advanced(image_path, annotations: pd.DataFrame, image=None, data_root='ocelot_testing_data'):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    organ = get_organ_from_metadata(image_path, data_root)
    params = ORGAN_PARAMS.get(organ, ORGAN_PARAMS['default'])

    print(f"Processing {Path(image_path).name} as {organ} organ (advanced)")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    detected_centroids, debug_images = detect_cells_multi_stage(image_bgr, params)

    detected_count = len(detected_centroids)
    actual_count = len(annotations)

    print(f"  Organ: {organ}, Detected: {detected_count}, Actual: {actual_count}")

    cv2.imwrite(
        str(results_dir / f"{Path(image_path).stem}_cell_adv_1_color_mask.png"),
        debug_images['color_mask']
    )
    cv2.imwrite(
        str(results_dir / f"{Path(image_path).stem}_cell_adv_2_combined.png"),
        debug_images['combined_mask']
    )
    cv2.imwrite(
        str(results_dir / f"{Path(image_path).stem}_cell_adv_3_distance.png"),
        debug_images['dist_transform']
    )
    cv2.imwrite(
        str(results_dir / f"{Path(image_path).stem}_cell_adv_4_centers.png"),
        debug_images['sure_fg']
    )

    overlay = image_bgr.copy()

    for centroid in detected_centroids:
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), 2)

    if annotations is not None and len(annotations) > 0:
        for _, row in annotations.iterrows():
            cx, cy = int(row['x']), int(row['y'])
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), 2)

    cv2.putText(overlay, f"Organ: {organ} (Advanced)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"Detected: {detected_count} | Actual: {actual_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    overlay_path = results_dir / f"{Path(image_path).stem}_cell_adv_5_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    original_path = results_dir / f"{Path(image_path).stem}_cell_adv_6_original.png"
    cv2.imwrite(str(original_path), image_bgr)

    return {
        'image_name': Path(image_path).name,
        'organ': organ,
        'detected_components': detected_count,
        'actual_components': actual_count,
        'difference': abs(detected_count - actual_count)
    }
