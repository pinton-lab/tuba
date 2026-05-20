"""Apply the same cranium-CC isolation as
:mod:`tuba._macaque.isolate_cranium`, but to the RAW (un-aligned) 250 um
cache. Produces ``macaque_skull_250um_cranium.nii.gz`` which is used by
:mod:`tuba._macaque.find_cranium_alignment` to re-derive the PCA
pre-alignment angles cleanly.

Same logic as ``isolate_cranium.py`` but operating on the small raw
cache so it runs in seconds rather than minutes.
"""
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_closing, label as cc_label

from .macaque_skull_constants import REG_DIR, MACAQUE_RAS_NIFTI

CC_THRESH = 12000
CLOSE_MM = 1.0
MARGIN_MM = 5.0

OUT = os.path.join(REG_DIR, 'macaque_skull_250um_cranium.nii.gz')


def main():
    img = nib.load(MACAQUE_RAS_NIFTI)
    arr = np.asarray(img.dataobj).astype(np.uint16)
    aff = img.affine
    voxel_mm = float(abs(aff[0, 0]))
    print(f'shape={arr.shape}, voxel={voxel_mm*1000:.2f} um')

    bone = arr > CC_THRESH
    iters = max(1, int(round(CLOSE_MM / voxel_mm)))
    closed = binary_closing(bone, iterations=iters)
    print(f'bone fraction (>{CC_THRESH}): {bone.mean():.4f}; '
          f'closed fraction: {closed.mean():.4f}')

    labeled, n_cc = cc_label(closed)
    sizes = np.bincount(labeled.ravel()); sizes[0] = 0
    print(f'  {n_cc} CCs; top-5 sizes (mL): '
          f'{[float(s)*voxel_mm**3/1000 for s in sorted(sizes, reverse=True)[:5]]}')
    cranium_id = int(np.argmax(sizes))
    cranium_mask = (labeled == cranium_id)
    print(f'  cranium CC #{cranium_id}, '
          f'volume {sizes[cranium_id]*voxel_mm**3/1000:.1f} mL')

    coords = np.argwhere(cranium_mask)
    lo = np.maximum(coords.min(0) - int(round(MARGIN_MM / voxel_mm)), 0)
    hi = np.minimum(coords.max(0) + 1 + int(round(MARGIN_MM / voxel_mm)),
                      np.asarray(arr.shape))
    print(f'  cranium bbox vox: i=[{lo[0]},{hi[0]}) j=[{lo[1]},{hi[1]}) '
          f'k=[{lo[2]},{hi[2]})')
    print(f'  cranium bbox world (mm):')
    for axis, name in enumerate(['x', 'y', 'z']):
        a = lo[axis] * aff[axis, axis] + aff[axis, 3]
        b = hi[axis] * aff[axis, axis] + aff[axis, 3]
        print(f'    {name}: [{min(a,b):+.1f}, {max(a,b):+.1f}]')

    out = np.zeros_like(arr)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = arr[
        lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    # Within bbox: zero any bone CC that isn't the main cranium CC
    bbox_bone = out > CC_THRESH
    bbox_closed = binary_closing(bbox_bone, iterations=iters)
    bbox_labeled, _ = cc_label(bbox_closed)
    bbox_sizes = np.bincount(bbox_labeled.ravel()); bbox_sizes[0] = 0
    bbox_cranium_id = int(np.argmax(bbox_sizes))
    bbox_cranium_mask = (bbox_labeled == bbox_cranium_id)
    n_mandible = int((bbox_bone & ~bbox_cranium_mask).sum())
    print(f'  suppressing {n_mandible} non-cranium bone voxels '
          f'({n_mandible*voxel_mm**3/1000:.2f} mL)')
    out[bbox_bone & ~bbox_cranium_mask] = 0

    nib.save(nib.Nifti1Image(out, aff), OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
