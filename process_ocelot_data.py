import argparse
import sys
from pathlib import Path
import pandas as pd
from image_processing_methods import PROCESSING_METHODS
from project_utils import save_cell_results, analyze_cell_results
import cv2

class OcelotDataProcessor:
    def __init__(self, data_root: str = "ocelot_testing_data"):
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.annotations_dir = self.data_root / "annotations"
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")

    def get_image_annotation_pairs(self, folder: str, data_type: str):
        valid_folders = ['test', 'train', 'val']
        valid_types = ['cell', 'tissue']
        if folder not in valid_folders:
            raise ValueError(f"folder must be one of {valid_folders}, got '{folder}'")
        if data_type not in valid_types:
            raise ValueError(f"data_type must be one of {valid_types}, got '{data_type}'")

        images_path = self.images_dir / folder / data_type
        annotations_path = self.annotations_dir / folder / data_type
        if not images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {images_path}")
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations directory not found: {annotations_path}")

        image_files = sorted(images_path.glob("*.jpg"))
        pairs = []

        if data_type == 'cell':
            for image_path in image_files:
                annotation_file = annotations_path / f"{image_path.stem}.csv"
                if annotation_file.exists():
                    pairs.append((image_path, annotation_file))
        elif data_type == 'tissue':
            for image_path in image_files:
                annotation_file = annotations_path / f"{image_path.stem}.png"
                if annotation_file.exists():
                    pairs.append((image_path, annotation_file))

        return pairs

    def load_annotations(self, annotation_path: Path, data_type: str):
        if data_type == 'cell':
            try:
                return pd.read_csv(annotation_path, header=None, names=['x', 'y', 'class'])
            except pd.errors.EmptyDataError:
                return pd.DataFrame(columns=['x', 'y', 'class'])
        elif data_type == 'tissue':
            return cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)

    def process_dataset(self, folder: str, data_type: str, image_limit: int = None, processing_method: str = None):
        pairs = self.get_image_annotation_pairs(folder, data_type)
        if image_limit is not None and image_limit > 0:
            pairs = pairs[:image_limit]

        processing_func = None
        if processing_method:
            if processing_method not in PROCESSING_METHODS:
                raise ValueError(f"Unknown processing method: {processing_method}")
            processing_func = PROCESSING_METHODS[processing_method]
            
        results_list = []
        for image_path, annotation_path in pairs:
            annotations = self.load_annotations(annotation_path, data_type)
            
            if processing_func:
                
                result = processing_func(image_path, annotations)
                
                if result:
                    results_list.append(result)

        if processing_method == 'cell_binary' and results_list:
            output_path = save_cell_results(results_list)
            analyze_cell_results(output_path)   

def main():
    if len(sys.argv) < 3:
        print("Error: Missing required arguments", file=sys.stderr)
        print("Usage: python process_ocelot_data.py <folder> <type> [--image-limit N] [--method METHOD]", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Process OCELOT dataset images with annotations")
    parser.add_argument('folder', type=str, choices=['test', 'train', 'val'])
    parser.add_argument('type', type=str, choices=['cell', 'tissue'])
    parser.add_argument('--image-limit', type=int, default=None)
    parser.add_argument('--method', type=str, default=None, choices=list(PROCESSING_METHODS.keys()))
    parser.add_argument('--data-root', type=str, default='ocelot_testing_data')

    args = parser.parse_args()

    try:
        processor = OcelotDataProcessor(data_root=args.data_root)
        processor.process_dataset(args.folder, args.type, image_limit=args.image_limit, processing_method=args.method)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
