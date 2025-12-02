import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from project_utils import calculate_detection_metrics
from .process_cell_advanced import (
    ORGAN_PARAMS,
    DEFAULT_TUNING_PARAMS,
    get_organ_from_metadata,
    detect_cells_multi_stage,
)

SIFT_PARAMS = {
    "min_size": 20.0,
    "max_size": 30.0,
    "min_response": 0.02,
    "merge_distance": 12.0,
}


def _create_sift_detector():
    try:
        return cv2.SIFT_create()
    except AttributeError:
        if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
            return cv2.xfeatures2d.SIFT_create()
        raise RuntimeError(
            "SIFT is not available in this OpenCV build. "
            "Install opencv-contrib-python or enable nonfree features."
        )


def _add_sift_on_color_mask(image_bgr, color_mask, base_centroids):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    masked_gray = cv2.bitwise_and(gray, gray, mask=color_mask)

    sift = _create_sift_detector()
    keypoints = sift.detect(masked_gray, None)

    min_size = SIFT_PARAMS["min_size"]
    max_size = SIFT_PARAMS["max_size"]
    min_response = SIFT_PARAMS["min_response"]
    merge_distance = SIFT_PARAMS["merge_distance"]

    final_centroids = []
    for c in base_centroids:
        final_centroids.append(np.array([float(c[0]), float(c[1])], dtype=np.float32))

    sift_keypoints_filtered = []

    for kp in keypoints:
        size = kp.size
        resp = kp.response
        x, y = kp.pt

        if size < min_size or size > max_size:
            continue
        if resp < min_response:
            continue

        ix, iy = int(round(x)), int(round(y))
        if iy < 0 or iy >= color_mask.shape[0] or ix < 0 or ix >= color_mask.shape[1]:
            continue
        if color_mask[iy, ix] == 0:
            continue

        kp_point = np.array([x, y], dtype=np.float32)

        too_close = False
        for c in final_centroids:
            if np.linalg.norm(kp_point - c) < merge_distance:
                too_close = True
                break
        if too_close:
            continue

        final_centroids.append(kp_point)
        sift_keypoints_filtered.append(kp)

    return final_centroids, sift_keypoints_filtered


def process_cell_sift_hybrid(
    image_path,
    annotations: pd.DataFrame,
    image=None,
    data_root="ocelot_testing_data",
    distance_threshold=100,
    tuning_params=None,
):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    image_path = Path(image_path)

    if image is None:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    else:
        image_bgr = image

    if image_bgr is None:
        print(f"Warning: could not read image {image_path}")
        return None

    organ = get_organ_from_metadata(image_path, data_root=data_root)
    params = ORGAN_PARAMS.get(organ, ORGAN_PARAMS["default"])

    if tuning_params is None:
        effective_tuning = DEFAULT_TUNING_PARAMS
    else:
        effective_tuning = DEFAULT_TUNING_PARAMS.copy()
        effective_tuning.update(tuning_params)

    print(f"Processing {image_path.name} (organ: {organ}) with Advanced+SIFT hybrid...")

    base_centroids, debug_images = detect_cells_multi_stage(
        image_bgr, params, tuning_params=effective_tuning
    )

    color_mask = debug_images["color_mask"]
    hybrid_centroids, sift_keypoints = _add_sift_on_color_mask(
        image_bgr, color_mask, base_centroids
    )

    detected_centroids = [(float(c[0]), float(c[1])) for c in hybrid_centroids]

    detected_count = len(detected_centroids)
    base_count = len(base_centroids)
    extra_sift = detected_count - base_count
    actual_count = len(annotations)

    metrics = calculate_detection_metrics(
        detected_centroids, annotations, distance_threshold=distance_threshold
    )

    print(
        f"  Detected (advanced): {base_count}, "
        f"Final (hybrid): {detected_count} (+{extra_sift} via SIFT), "
        f"Actual: {actual_count}"
    )
    print(
        f"  TP: {metrics['true_positives']}, "
        f"FP: {metrics['false_positives']}, "
        f"FN: {metrics['false_negatives']}"
    )
    print(
        f"  Precision: {metrics['precision']:.3f}, "
        f"Recall: {metrics['recall']:.3f}, "
        f"F1: {metrics['f1_score']:.3f}, "
        f"Accuracy: {metrics['accuracy']:.3f}"
    )

    stem = image_path.stem

    cv2.imwrite(
        str(results_dir / f"{stem}_cell_hybrid_1_color_mask.png"),
        debug_images["color_mask"],
    )
    cv2.imwrite(
        str(results_dir / f"{stem}_cell_hybrid_2_combined.png"),
        debug_images["combined_mask"],
    )
    cv2.imwrite(
        str(results_dir / f"{stem}_cell_hybrid_3_distance.png"),
        debug_images["dist_transform"],
    )
    cv2.imwrite(
        str(results_dir / f"{stem}_cell_hybrid_4_centers.png"),
        debug_images["sure_fg"],
    )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kp_vis = cv2.drawKeypoints(
        gray,
        sift_keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imwrite(
        str(results_dir / f"{stem}_cell_hybrid_5_sift_keypoints.png"),
        kp_vis,
    )

    overlay = image_bgr.copy()

    for c in detected_centroids:
        cx, cy = int(round(c[0])), int(round(c[1]))
        cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), 2)

    if annotations is not None and len(annotations) > 0:
        for _, row in annotations.iterrows():
            gx, gy = int(row["x"]), int(row["y"])
            cv2.circle(overlay, (gx, gy), 6, (0, 0, 255), 2)

    cv2.putText(
        overlay,
        f"Organ: {organ} (Adv+SIFT)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        overlay,
        f"Det: {detected_count} (base {base_count}, +{extra_sift}) | GT: {actual_count}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        overlay,
        f"P: {metrics['precision']:.2f} R: {metrics['recall']:.2f} F1: {metrics['f1_score']:.2f}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(
        str(results_dir / f"{stem}_cell_hybrid_6_overlay.png"),
        overlay,
    )

    cv2.imwrite(
        str(results_dir / f"{stem}_cell_hybrid_7_original.png"),
        image_bgr,
    )

    return {
        "image_name": image_path.name,
        "organ": organ,
        "detected_components": detected_count,
        "actual_components": actual_count,
        "difference": abs(detected_count - actual_count),
        **metrics,
    }
