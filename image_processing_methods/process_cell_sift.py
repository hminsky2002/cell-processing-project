import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from project_utils import calculate_detection_metrics


SIFT_PARAMS = {
    "min_size": 20.0,
    "max_size": 30.0,
    "min_response": 0.02,
    "contrastThreshold": 0.01,
    "edgeThreshold": 10,
    "sigma": 1.6,
}


def get_organ_from_metadata(image_path, data_root="ocelot_testing_data"):
    img_id = Path(image_path).stem
    metadata_path = Path(data_root) / "metadata.json"
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if img_id in metadata.get("sample_pairs", {}):
            return metadata["sample_pairs"][img_id].get("organ", "default")
    except Exception as e:
        print(f"Warning: Could not load metadata for {image_path}: {e}")
    return "default"


def _create_sift_detector():
    try:
        return cv2.SIFT_create(
            contrastThreshold=SIFT_PARAMS["contrastThreshold"],
            edgeThreshold=SIFT_PARAMS["edgeThreshold"],
            sigma=SIFT_PARAMS["sigma"],
        )
    except AttributeError:
        if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
            return cv2.xfeatures2d.SIFT_create(
                contrastThreshold=SIFT_PARAMS["contrastThreshold"],
                edgeThreshold=SIFT_PARAMS["edgeThreshold"],
                sigma=SIFT_PARAMS["sigma"],
            )
        raise RuntimeError(
            "SIFT is not available in this OpenCV build. "
            "Install opencv-contrib-python or enable nonfree features."
        )


def detect_cells_sift(gray: np.ndarray):
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    sift = _create_sift_detector()
    keypoints = sift.detect(gray_blur, None)

    min_size = SIFT_PARAMS["min_size"]
    max_size = SIFT_PARAMS["max_size"]
    min_response = SIFT_PARAMS["min_response"]

    filtered_kps = []
    centroids = []

    for kp in keypoints:
        size = kp.size
        resp = kp.response
        if size < min_size or size > max_size:
            continue
        if resp < min_response:
            continue

        x, y = kp.pt
        filtered_kps.append(kp)
        centroids.append((int(round(x)), int(round(y))))

    return filtered_kps, centroids


def process_cell_sift(
    image_path,
    annotations: pd.DataFrame,
    image=None,
    distance_threshold: int = 100,
    data_root: str = "ocelot_testing_data",
):
    image_path = Path(image_path)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    organ = get_organ_from_metadata(image_path, data_root=data_root)

    print(f"Processing {image_path.name} (organ: {organ}) with SIFT...")

    try:
        keypoints, detected_centroids = detect_cells_sift(gray)
    except RuntimeError as e:
        print(f"Error in SIFT detection for {image_path}: {e}")
        return None

    detected_count = len(detected_centroids)
    actual_count = len(annotations)

    metrics = calculate_detection_metrics(
        detected_centroids, annotations, distance_threshold=distance_threshold
    )

    print(
        f"  Detected: {detected_count}, Actual: {actual_count} | "
        f"TP: {metrics['true_positives']} FP: {metrics['false_positives']} "
        f"FN: {metrics['false_negatives']}"
    )

    stem = image_path.stem

    gray_path = results_dir / f"{stem}_cell_sift_1_gray.png"
    cv2.imwrite(str(gray_path), gray)

    kp_vis = cv2.drawKeypoints(
        gray,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    kp_path = results_dir / f"{stem}_cell_sift_2_keypoints.png"
    cv2.imwrite(str(kp_path), kp_vis)

    overlay = image_bgr.copy()

    for (cx, cy) in detected_centroids:
        cv2.circle(overlay, (cx, cy), 6, (0, 255, 0), 2)

    if len(annotations) > 0:
        for _, row in annotations.iterrows():
            gx, gy = int(row["x"]), int(row["y"])
            cv2.circle(overlay, (gx, gy), 6, (0, 0, 255), 2)

    cv2.putText(
        overlay,
        f"Organ: {organ}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        overlay,
        f"Detected: {detected_count} | Actual: {actual_count}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        overlay,
        f"P: {metrics['precision']:.2f}  R: {metrics['recall']:.2f}  F1: {metrics['f1_score']:.2f}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    overlay_path = results_dir / f"{stem}_cell_sift_3_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    original_path = results_dir / f"{stem}_cell_sift_4_original.png"
    cv2.imwrite(str(original_path), image_bgr)

    return {
        "image_name": image_path.name,
        "organ": organ,
        "detected_components": detected_count,
        "actual_components": actual_count,
        "difference": abs(detected_count - actual_count),
        **metrics,
    }
