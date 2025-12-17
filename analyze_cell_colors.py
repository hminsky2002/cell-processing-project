import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from collections import defaultdict
from process_ocelot_data import OcelotDataProcessor


def analyze_cell_colors(data_root='ocelot_testing_data', folder='train', max_images=None):
    processor = OcelotDataProcessor(data_root=data_root)

    metadata_path = Path(data_root) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    pairs = processor.get_image_annotation_pairs(folder, 'cell')

    organ_stats = defaultdict(lambda: {
        'colors_bgr': [],
        'colors_hsv': [],
        'intensities': [],
        'count': 0
    })

    for img_path, ann_path in pairs:

        img_id = img_path.stem


        organ = None
        for key, value in metadata['sample_pairs'].items():
            if key == img_id:
                organ = value.get('organ', 'unknown')
                break

        if organ is None:
            continue


        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        annotations = pd.read_csv(ann_path, header=None, names=['x', 'y', 'class'])


        for _, row in annotations.iterrows():
            x, y = int(row['x']), int(row['y'])


            region_size = 5
            half = region_size // 2

            y1, y2 = max(0, y - half), min(image_bgr.shape[0], y + half + 1)
            x1, x2 = max(0, x - half), min(image_bgr.shape[1], x + half + 1)

            region_bgr = image_bgr[y1:y2, x1:x2]
            region_hsv = image_hsv[y1:y2, x1:x2]


            mean_bgr = region_bgr.mean(axis=(0, 1))
            mean_hsv = region_hsv.mean(axis=(0, 1))
            mean_intensity = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY).mean()

            organ_stats[organ]['colors_bgr'].append(mean_bgr)
            organ_stats[organ]['colors_hsv'].append(mean_hsv)
            organ_stats[organ]['intensities'].append(mean_intensity)
            organ_stats[organ]['count'] += 1

        print(f"Processed {img_path.name} ({organ}): {len(annotations)} cells")

        if max_images and organ_stats[organ]['count'] >= max_images:
            break


    print("\n" + "="*80)
    print("COLOR ANALYSIS BY ORGAN TYPE")
    print("="*80)

    for organ, stats in sorted(organ_stats.items()):
        if stats['count'] == 0:
            continue

        colors_bgr = np.array(stats['colors_bgr'])
        colors_hsv = np.array(stats['colors_hsv'])
        intensities = np.array(stats['intensities'])

        print(f"\n{organ.upper()} ({stats['count']} cells)")
        print("-" * 40)


        mean_bgr = colors_bgr.mean(axis=0)
        std_bgr = colors_bgr.std(axis=0)
        print(f"  BGR Mean: B={mean_bgr[0]:.1f}, G={mean_bgr[1]:.1f}, R={mean_bgr[2]:.1f}")
        print(f"  BGR Std:  B={std_bgr[0]:.1f}, G={std_bgr[1]:.1f}, R={std_bgr[2]:.1f}")


        mean_hsv = colors_hsv.mean(axis=0)
        std_hsv = colors_hsv.std(axis=0)
        print(f"  HSV Mean: H={mean_hsv[0]:.1f}, S={mean_hsv[1]:.1f}, V={mean_hsv[2]:.1f}")
        print(f"  HSV Std:  H={std_hsv[0]:.1f}, S={std_hsv[1]:.1f}, V={std_hsv[2]:.1f}")


        mean_int = intensities.mean()
        std_int = intensities.std()
        print(f"  Intensity Mean: {mean_int:.1f}")
        print(f"  Intensity Std:  {std_int:.1f}")


        print(f"\n  Recommended BGR range:")
        lower_bgr = np.clip(mean_bgr - 2*std_bgr, 0, 255).astype(int)
        upper_bgr = np.clip(mean_bgr + 2*std_bgr, 0, 255).astype(int)
        print(f"    Lower: {lower_bgr}")
        print(f"    Upper: {upper_bgr}")

        print(f"\n  Recommended HSV range:")
        lower_hsv = np.clip(mean_hsv - 2*std_hsv, 0, [179, 255, 255]).astype(int)
        upper_hsv = np.clip(mean_hsv + 2*std_hsv, 0, [179, 255, 255]).astype(int)
        print(f"    Lower: {lower_hsv}")
        print(f"    Upper: {upper_hsv}")

    return organ_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze color characteristics of cells by organ type"
    )
    parser.add_argument(
        '--folder',
        type=str,
        choices=['test', 'train', 'val'],
        default='train',
        help='Which dataset folder to analyze'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='ocelot_testing_data',
        help='Root directory of OCELOT dataset'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to analyze'
    )

    args = parser.parse_args()

    try:
        analyze_cell_colors(
            data_root=args.data_root,
            folder=args.folder,
            max_images=args.max_images
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
