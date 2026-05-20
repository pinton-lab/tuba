"""Isolate the cranium from the refined 250 um aligned cache.

Direct port of :mod:`tuba._macaque.isolate_cranium` (which operates on
the 11 GB native) to the 250 um cache for fast iteration. Identifies
the cranium as the largest bone connected component, computes its
bounding box plus a small margin, and zeros every voxel outside that
bbox plus any non-cranium-CC bone within the bbox.

After refinement (:mod:`tuba._macaque.refine_250um`) the cranium is in
its final orientation; this script just removes the mandible (separate
bone CC articulated to the cranium without bone-contact) and any other
loose fragments. Reads from / writes to ``MACAQUE_RAS_ALIGNED_250UM``
in place. The pre-isolation backup is preserved as
``..._with_mandible.nii.gz`` so the operation can be re-run from
scratch.
"""
import os
import shutil
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_closing, label as cc_label

from .macaque_skull_constants import MACAQUE_RAS_ALIGNED_250UM

CC_THRESH = 12000           # high enough to be cortical bone (excludes
                             # the diffuse mounting/soft-tissue residue)
CLOSE_MM  = 1.0              # bridge sutures inside the cranium so the
                             # whole vault forms one CC
MARGIN_MM = 5.0              # bbox margin around the cranium CC

PRE_ISOLATE_BACKUP = MACAQUE_RAS_ALIGNED_250UM.replace(
    '.nii.gz', '_with_mandible.nii.gz')


def main():
    if os.path.exists(PRE_ISOLATE_BACKUP):
        src = PRE_ISOLATE_BACKUP
        print(f'Loading from pre-isolation backup: {PRE_ISOLATE_BACKUP}')
    else:
        src = MACAQUE_RAS_ALIGNED_250UM
        print(f'Loading {src} (no backup yet, creating one)')
        shutil.copy(src, PRE_ISOLATE_BACKUP)

    img = nib.load(src)
    arr = np.asarray(img.dataobj).astype(np.uint16)
    aff = img.affine
    voxel_mm = float(abs(aff[0, 0]))
    print(f'  shape {arr.shape}, voxel {voxel_mm*1000:.1f} um')

    bone = arr > CC_THRESH
    iters = max(1, int(round(CLOSE_MM / voxel_mm)))
    closed = binary_closing(bone, iterations=iters)
    print(f'  bone fraction (>{CC_THRESH}): {bone.mean():.4f}; '
          f'closed ({CLOSE_MM} mm): {closed.mean():.4f}')

    print('Connected-component labelling...')
    labeled, n_cc = cc_label(closed)
    sizes = np.bincount(labeled.ravel()); sizes[0] = 0
    top5 = sorted(sizes, reverse=True)[:5]
    print(f'  {n_cc} CCs; top-5 sizes (mL): '
          f'{[float(s)*voxel_mm**3/1000 for s in top5]}')
    cranium_id = int(np.argmax(sizes))
    cranium_mask = (labeled == cranium_id)
    print(f'  cranium CC #{cranium_id}, '
          f'volume {sizes[cranium_id]*voxel_mm**3/1000:.1f} mL')

    coords = np.argwhere(cranium_mask)
    margin_vox = max(1, int(round(MARGIN_MM / voxel_mm)))
    lo = np.maximum(coords.min(0) - margin_vox, 0)
    hi = np.minimum(coords.max(0) + margin_vox + 1,
                     np.asarray(arr.shape))
    print(f'  bbox vox: i=[{lo[0]},{hi[0]}) '
          f'j=[{lo[1]},{hi[1]}) k=[{lo[2]},{hi[2]})')
    print(f'  bbox world (mm):')
    for axis, name in enumerate(['x', 'y', 'z']):
        a = lo[axis] * aff[axis, axis] + aff[axis, 3]
        b = hi[axis] * aff[axis, axis] + aff[axis, 3]
        print(f'    {name}: [{min(a,b):+.1f}, {max(a,b):+.1f}]')

    out = np.zeros_like(arr)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = arr[
        lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    bbox_bone = out > CC_THRESH
    bbox_closed = binary_closing(bbox_bone, iterations=iters)
    bbox_labeled, _ = cc_label(bbox_closed)
    bbox_sizes = np.bincount(bbox_labeled.ravel()); bbox_sizes[0] = 0
    bbox_cranium_id = int(np.argmax(bbox_sizes))
    bbox_cranium_mask = (bbox_labeled == bbox_cranium_id)
    n_mandible = int((bbox_bone & ~bbox_cranium_mask).sum())
    print(f'  suppressing {n_mandible} non-cranium-CC bone voxels in bbox '
          f'({n_mandible*voxel_mm**3/1000:.2f} mL of mandible/fragments)')
    out[bbox_bone & ~bbox_cranium_mask] = 0

    nib.save(nib.Nifti1Image(out, aff), MACAQUE_RAS_ALIGNED_250UM)
    print(f'wrote isolated cranium -> {MACAQUE_RAS_ALIGNED_250UM}')


if __name__ == '__main__':
    main()
