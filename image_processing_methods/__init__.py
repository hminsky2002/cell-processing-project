from .process_cell_binary import process_cell_binary
from .process_tissue_binary import process_tissue_binary


PROCESSING_METHODS = {
    'cell_binary': process_cell_binary,
    'tissue_binary': process_tissue_binary,
}

__all__ = ['PROCESSING_METHODS', 'process_cell_binary', 'process_tissue_binary']
