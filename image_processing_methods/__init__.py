from .connected_components import process_connected_components

PROCESSING_METHODS = {
    'connected_components': process_connected_components,
}

__all__ = ['PROCESSING_METHODS', 'process_connected_components']
