"""TIFF-stack reading for raw microCT acquisitions."""
import time
import numpy as np


def load_stack_to_array(paths, dtype=np.uint16, verbose=True):
    """Load a list of TIFF slice paths into a contiguous
    (n_slc, H, W) array. Used by the mouse Maga pipeline to bring the
    full 6.127 um native (~38 GB) into memory for the rotation step.

    Pre-allocates the output to keep peak memory predictable
    (one copy of the volume, not two).
    """
    import tifffile
    t0 = time.time()
    if verbose:
        print(f'Loading {len(paths)} TIFFs into a {dtype.__name__} array...')
    sample = tifffile.imread(paths[0])
    raw = np.empty((len(paths), sample.shape[0], sample.shape[1]), dtype=dtype)
    raw[0] = sample
    for k in range(1, len(paths)):
        raw[k] = tifffile.imread(paths[k])
        if verbose and k % 500 == 0:
            print(f'  {k}/{len(paths)}  ({(time.time()-t0):.0f} s elapsed)')
    if verbose:
        print(f'  loaded: shape={raw.shape}, dtype={raw.dtype}, '
              f'size={raw.nbytes/1e9:.1f} GB, t={(time.time()-t0):.0f} s')
    return raw
