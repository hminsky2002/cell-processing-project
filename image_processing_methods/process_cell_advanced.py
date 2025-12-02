import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from project_utils import calculate_detection_metrics


# =============================================================================
# TUNABLE DETECTION PARAMETERS - ADJUST THESE TO IMPROVE RECALL
# =============================================================================
# These are the default tuning parameters. Modify these values to reduce false negatives.
#
# **MOST IMPORTANT FOR REDUCING FALSE NEGATIVES:**
# 1. dist_transform_threshold: LOWER = more detections (try 0.03-0.07)
# 2. use_color_mask_only: True = skip adaptive threshold if it's too restrictive
# 3. blob_min_circularity/convexity/inertia: LOWER = accept more shapes
#
DEFAULT_TUNING_PARAMS = {
    # === CRITICAL PARAMETERS (adjust these first) ===
    'dist_transform_threshold': 0.05,    # LOWER = more cells detected (try 0.03-0.08)
    'use_color_mask_only': False,         # True = skip adaptive threshold (less restrictive)

    # === Morphological operations ===
    'morph_kernel_size': 3,               # Kernel size for open/close operations
    'morph_open_iterations': 1,           # Noise removal iterations
    'morph_close_iterations': 2,          # Gap filling iterations

    # === Adaptive thresholding ===
    'adaptive_block_size': 31,            # Block size (must be odd)
    'adaptive_C': 8,                      # Constant subtracted from mean

    # === Connected components filtering ===
    # UPDATED: Balanced shape filtering - not too strict, not too loose
    'aspect_ratio_max': 4.0,              # Max elongation (balanced between 3.0 and 6.0)
    'compactness_min': 0.35,              # Min fill ratio (balanced between 0.3 and 0.4)

    # === Blob detection (supplementary method) ===
    'blob_detector_enabled': True,        # Enable/disable blob detector
    'blob_min_circularity': 0.12,         # Slightly relaxed for varied cell shapes
    'blob_min_convexity': 0.12,           # Slightly relaxed
    'blob_min_inertia': 0.12,             # Slightly relaxed
    'blob_merge_distance': 10,            # Min distance between blob and existing detection
}


# =============================================================================
# ORGAN-SPECIFIC HSV PARAMETERS
# =============================================================================
# UPDATED based on analyze_cell_colors.py output
# Key changes:
#   - Widened H/S lower bounds to catch more cells (reduce FN)
#   - Lowered V upper bounds to exclude light pink stroma (reduce FP)
#   - Raised S lower bounds slightly to exclude desaturated stroma
#
ORGAN_PARAMS = {
    'bladder': {
        # Analysis: H=143.5±8.8, S=134.3±31.0, V=109.8±38.6
        'hsv_lower': np.array([125, 70, 30]),
        'hsv_upper': np.array([162, 200, 165]),  # V upper limited to exclude light stroma
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'endometrium': {
        # Analysis: H=151.2±11.9, S=121.2±36.2, V=135.9±44.1
        'hsv_lower': np.array([127, 50, 45]),
        'hsv_upper': np.array([175, 195, 190]),  # V upper limited
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'head-and-neck': {
        # Analysis: H=142.7±14.6, S=132.9±45.1, V=132.6±45.4
        'hsv_lower': np.array([113, 45, 40]),
        'hsv_upper': np.array([172, 225, 190]),  # V upper limited
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'kidney': {
        # Analysis: H=144.6±10.1, S=122.2±32.0, V=112.8±41.1
        'hsv_lower': np.array([124, 60, 30]),
        'hsv_upper': np.array([165, 190, 165]),  # V upper limited
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'prostate': {
        # Analysis: H=140.3±11.9, S=88.7±36.1, V=151.8±42.1
        # Note: prostate has lower saturation and higher brightness than other organs
        # Widened V range to catch more cells
        'hsv_lower': np.array([116, 15, 50]),   # V floor lowered from 65 → 50
        'hsv_upper': np.array([164, 170, 215]), # S upper 165→170, V upper 200→215
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'stomach': {
        # Analysis: H=142.9±10.1, S=95.7±36.7, V=144.8±42.2
        'hsv_lower': np.array([122, 25, 60]),
        'hsv_upper': np.array([164, 170, 195]),  # V upper limited
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'default': {
        # Conservative defaults that work across organs
        # Wide H range to catch all purples, moderate S/V filtering
        'hsv_lower': np.array([113, 20, 30]),
        'hsv_upper': np.array([175, 225, 190]),  # V upper limited to exclude light pink
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


def detect_cells_multi_stage(image_bgr, params, tuning_params=None):
    """
    Multi-stage cell detection with configurable parameters.

    Args:
        image_bgr: Input image in BGR format
        params: Organ-specific parameters (HSV ranges, area thresholds)
        tuning_params: Optional dict of tuning parameters (uses DEFAULT_TUNING_PARAMS if None)

    Returns:
        Tuple of (detected_centroids, debug_images)

    TUNING GUIDE - Parameters ranked by impact on recall:

    **TIER 1 - HIGHEST IMPACT (adjust these first):**
    1. dist_transform_threshold (0.0-1.0, default: 0.05)
       - Controls threshold for distance transform peaks
       - LOWER = More detections, higher recall, more FP
       - HIGHER = Fewer detections, lower recall, fewer FP
       - Recommended for high FN: 0.03-0.05
       - Current default: 0.05

    2. use_color_mask_only (bool, default: False)
       - Skip adaptive threshold, use only color filtering
       - True = Less restrictive, better recall
       - False = More restrictive, better precision
       - Try True if combined mask is eliminating real cells

    **TIER 2 - MEDIUM IMPACT:**
    3. blob_min_circularity/convexity/inertia (0.0-1.0, defaults: 0.15)
       - Controls blob detector sensitivity
       - LOWER = Accept more varied shapes
       - Affects supplementary detections only

    4. aspect_ratio_max (default: 3.0)
       - Max elongation of detected components
       - HIGHER = Accept more elongated cells

    5. compactness_min (0.0-1.0, default: 0.4)
       - Min fill ratio of bounding box
       - LOWER = Accept more irregular shapes

    **TIER 3 - FINE TUNING:**
    6. adaptive_block_size (odd int, default: 31)
       - Local adaptive threshold neighborhood
       - SMALLER = More sensitive to variations

    7. morph_kernel_size (odd int, default: 3)
       - Morphological operation kernel size
       - SMALLER = Preserve smaller cells
    """

    # Use default tuning params if none provided
    if tuning_params is None:
        tuning_params = DEFAULT_TUNING_PARAMS
    else:
        # Merge with defaults for any missing params
        merged_params = DEFAULT_TUNING_PARAMS.copy()
        merged_params.update(tuning_params)
        tuning_params = merged_params

    # Extract tuning parameters
    morph_kernel_size = tuning_params['morph_kernel_size']
    morph_open_iterations = tuning_params['morph_open_iterations']
    morph_close_iterations = tuning_params['morph_close_iterations']
    adaptive_block_size = tuning_params['adaptive_block_size']
    adaptive_C = tuning_params['adaptive_C']
    use_color_mask_only = tuning_params['use_color_mask_only']
    dist_transform_threshold = tuning_params['dist_transform_threshold']
    aspect_ratio_max = tuning_params['aspect_ratio_max']
    compactness_min = tuning_params['compactness_min']
    blob_detector_enabled = tuning_params['blob_detector_enabled']
    blob_min_circularity = tuning_params['blob_min_circularity']
    blob_min_convexity = tuning_params['blob_min_convexity']
    blob_min_inertia = tuning_params['blob_min_inertia']
    blob_merge_distance = tuning_params['blob_merge_distance']

    # Color filtering
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(image_hsv, params['hsv_lower'], params['hsv_upper'])

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_small, iterations=morph_open_iterations)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_small, iterations=morph_close_iterations)

    # Adaptive thresholding (optional)
    if use_color_mask_only:
        combined_mask = color_mask
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=adaptive_block_size,
            C=adaptive_C
        )
        combined_mask = cv2.bitwise_and(color_mask, adaptive_thresh)

    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Distance transform
    dist_transform = cv2.distanceTransform(combined_mask, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_transform, dist_transform_threshold * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Connected components
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

            if aspect_ratio < aspect_ratio_max and compactness > compactness_min:
                detected_centroids.append(centroids[label])

    # Blob detection (supplementary)
    if blob_detector_enabled:
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.filterByArea = True
        blob_params.minArea = params['min_area']
        blob_params.maxArea = params['max_area']
        blob_params.filterByCircularity = True
        blob_params.minCircularity = blob_min_circularity
        blob_params.filterByConvexity = True
        blob_params.minConvexity = blob_min_convexity
        blob_params.filterByInertia = True
        blob_params.minInertiaRatio = blob_min_inertia

        detector = cv2.SimpleBlobDetector_create(blob_params)
        keypoints = detector.detect(color_mask)

        for kp in keypoints:
            kp_point = np.array([kp.pt[0], kp.pt[1]])
            too_close = False
            for centroid in detected_centroids:
                dist = np.linalg.norm(kp_point - centroid)
                if dist < blob_merge_distance:
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


def process_cell_advanced(image_path, annotations: pd.DataFrame, image=None, data_root='ocelot_testing_data', distance_threshold=100, tuning_params=None):
    """
    Process cell detection with advanced multi-stage algorithm.

    Args:
        image_path: Path to input image
        annotations: DataFrame with ground truth annotations
        image: Optional pre-loaded image
        data_root: Root directory for OCELOT data
        distance_threshold: Distance threshold for matching detections to ground truth
        tuning_params: Optional dict to override DEFAULT_TUNING_PARAMS

    To reduce false negatives, pass custom tuning_params:
        tuning_params = {
            'dist_transform_threshold': 0.03,  # Lower for more detections
            'use_color_mask_only': True,       # Skip adaptive threshold
        }
    """
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    organ = get_organ_from_metadata(image_path, data_root)
    params = ORGAN_PARAMS.get(organ, ORGAN_PARAMS['default'])

    print(f"Processing {Path(image_path).name} as {organ} organ (advanced)")

    # Print tuning params if custom ones are provided
    if tuning_params is not None:
        print(f"  Using custom tuning parameters:")
        for key, value in tuning_params.items():
            print(f"    {key}: {value}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    detected_centroids, debug_images = detect_cells_multi_stage(image_bgr, params, tuning_params)

    detected_count = len(detected_centroids)
    actual_count = len(annotations)

    metrics = calculate_detection_metrics(detected_centroids, annotations, distance_threshold)

    print(f"  Organ: {organ}, Detected: {detected_count}, Actual: {actual_count}")
    print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
    print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}, Accuracy: {metrics['accuracy']:.3f}")

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
    cv2.putText(overlay, f"P: {metrics['precision']:.2f} R: {metrics['recall']:.2f} F1: {metrics['f1_score']:.2f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    overlay_path = results_dir / f"{Path(image_path).stem}_cell_adv_5_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    original_path = results_dir / f"{Path(image_path).stem}_cell_adv_6_original.png"
    cv2.imwrite(str(original_path), image_bgr)

    return {
        'image_name': Path(image_path).name,
        'organ': organ,
        'detected_components': detected_count,
        'actual_components': actual_count,
        'difference': abs(detected_count - actual_count),
        **metrics
    }