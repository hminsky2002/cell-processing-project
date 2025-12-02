from .process_cell_binary import process_cell_binary
from .process_cell_color import process_cell_color
from .process_cell_advanced import process_cell_advanced
from .process_tissue_binary import process_tissue_binary
from .process_cell_sift_hybrid import process_cell_sift_hybrid
from .process_cell_sift import process_cell_sift
from .process_cell_advanced_cnn import process_cell_advanced_cnn

PROCESSING_METHODS = {
    'cell_binary': process_cell_binary,
    'cell_color': process_cell_color,
    'cell_advanced': process_cell_advanced,
    'tissue_binary': process_tissue_binary,
    'cell_sift_hybrid': process_cell_sift_hybrid,
    'cell_sift': process_cell_sift,
    'cell_advanced_cnn': process_cell_advanced_cnn,
}

__all__ = [
    'PROCESSING_METHODS',
    'process_cell_binary',
    'process_cell_color',
    'process_cell_advanced',
    'process_tissue_binary',
    'process_cell_sift_hybrid',
    'process_cell_sift',
    'process_cell_advanced_cnn',
]
