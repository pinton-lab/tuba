"""NRRD reading for the human Halle skull microCT.

The Halle NRRD is stored in LAS voxel order despite the NRRD header
tagging the volume as ``space=left-posterior-superior`` (LPS); this is
a known property of the Halle release (verified by slice-viewer
inspection in 2026-04-28). The species module
(:mod:`tuba.species.human`) is responsible for constructing the
correct storage-to-RAS affine and should not trust the NRRD header's
``space`` field.
"""
import numpy as np


def read_nrrd(path, verbose=True):
    """Read an NRRD volume.

    Returns
    -------
    array : 3D ndarray
        Volume data with shape ``(n0, n1, n2)`` in the NRRD's native
        storage order.
    voxel_mm : float
        Isotropic voxel size (read from the diagonal of
        ``header['space directions']``; assumed isotropic).
    header : dict
        The full NRRD header.
    """
    import nrrd
    if verbose:
        print(f'Loading {path}...')
    data, header = nrrd.read(path)
    voxel_mm = float(header['space directions'][0, 0])
    if verbose:
        print(f'  shape={data.shape}, voxel={voxel_mm} mm, '
              f'range=[{data.min()}, {data.max()}]')
    return data, voxel_mm, header
