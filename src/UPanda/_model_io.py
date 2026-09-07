"""Decode historical emulator metadata without changing the on-disk format."""


def decode_attributes(attributes):
    """Return named metadata for the supported 7–12 element pickle layouts."""
    base = ['input_scaler', 'output_scaler', 'offset', 'log_preprocess', 'pca']
    tail = ['parameter_names', 'rescaling_factor']
    layouts = {
        7: base + tail,
        8: base + ['pca_scaler'] + tail,
        9: base + ['pca_scaler'] + tail + ['rescale_by_as'],
        10: base + ['pca_scaler'] + tail + ['rescale_by_as', 'galaxy_bias_config'],
        11: base + ['pca_scaler'] + tail + ['rescale_by_as', 'cheby_preprocess', 'cheby_meta'],
        12: base + ['pca_scaler'] + tail + ['rescale_by_as', 'cheby_preprocess', 'cheby_meta', 'galaxy_bias_config'],
    }
    if len(attributes) not in layouts:
        raise ValueError(f'Unknown model format with {len(attributes)} attributes')
    result = dict(pca_scaler=None, rescale_by_as=False, cheby_preprocess=False,
                  cheby_meta=None, galaxy_bias_config=None)
    result.update(zip(layouts[len(attributes)], attributes))
    return result
