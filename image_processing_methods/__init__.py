from .process_cell_binary import process_cell_binary
from .process_cell_color import process_cell_color
from .process_cell_advanced import process_cell_advanced
from .process_tissue_binary import process_tissue_binary


PROCESSING_METHODS = {
    'cell_binary': process_cell_binary,
    'cell_color': process_cell_color,
    'cell_advanced': process_cell_advanced,
    'tissue_binary': process_tissue_binary,
}

__all__ = [
    'PROCESSING_METHODS',
    'process_cell_binary',
    'process_cell_color',
    'process_cell_advanced',
    'process_tissue_binary',
]
