import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def process_connected_components(image_path, annotations: pd.DataFrame, image=None):
    image = cv2.imread(str(image_path))

    target_color_rgb = np.array([83, 59, 118])
    target_color_bgr = target_color_rgb[::-1]
    color_threshold = 50

    color_diff = np.linalg.norm(image - target_color_bgr, axis=2)
    binary_img = (color_diff < color_threshold).astype('uint8')

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

    min_area = 50
    min_roundness = 0.3
    max_roundness = 1.0

    filtered_labels = np.zeros_like(labels)
    valid_label = 1
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        component_mask = (labels == label).astype('uint8')
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        contour = contours[0]
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        roundness = (4 * np.pi * area) / (perimeter ** 2)

        if min_roundness <= roundness <= max_roundness:
            filtered_labels[labels == label] = valid_label
            valid_label += 1

    print(f"Total components: {num_labels - 1}, After filtering: {valid_label - 1}")
    plt.figure()
    plt.title(image_path.name)
    comment = f"Annotations: {annotations.shape[0]}"
    print(comment)
    plt.imshow(filtered_labels, cmap='gray')
    plt.axis("off")
    plt.show()
