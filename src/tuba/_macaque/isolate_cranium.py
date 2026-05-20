"""Isolate the cranium from the AMNH M-87264 microCT.

The MorphoSource scan contains the cranium AND a separately-laid
mandible. The mandible is not articulated to the cranium and biases
PCA pre-alignment (which assumes a single rigid bone object), the
cavity extractor (which assumes the cavity is enclosed by the
largest bone CC), and the downstream SyN registration to NMT.

This script identifies the cranium as the largest bone connected
component in the aligned-native microCT, masks every voxel outside
that CC's bounding box (with a small margin) to zero, and overwrites
the aligned-native NIfTI in place. The downstream caches must be
rebuilt afterwards (downsample_aligned_to_250um, find_alignment_rotation,
align_native_to_axes for the second-pass alignment, then cavity
extractor + SyN).

Workflow:
  1. Read MACAQUE_RAS_NATIVE_ALIGNED (already PCA-aligned in the v1
     pipeline; the v1 PCA included the mandible so the rotation is
     slightly biased -- we will fix this in the next pass).
  2. Threshold to bone, close to merge cranium parts, label CCs.
  3. Pick the largest CC = cranium.
  4. Compute cranium bbox + 5 mm margin in each direction.
  5. Zero out everything outside that bbox.
  6. Overwrite the aligned-native NIfTI.
"""
import os
import sys
import time
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_closing, label as cc_label

from .macaque_skull_constants import (
    MACAQUE_RAS_NATIVE_ALIGNED, MACAQUE_RAS_ALIGNED_250UM,
    BONE_THRESH, downsample_aligned_to_250um)

CC_THRESH = 12000           # higher than BONE_THRESH so the CC is the
                             # cortical bone (not the diffuse mounting/
                             # soft-tissue residue)
CLOSE_MM = 1.0              # bridge sutures inside the cranium so the
                             # whole vault forms one CC
MARGIN_MM = 5.0             # bbox margin


def main():
    print(f'Loading {MACAQUE_RAS_NATIVE_ALIGNED}...')
    img = nib.load(MACAQUE_RAS_NATIVE_ALIGNED)
    arr = np.asarray(img.dataobj).astype(np.uint16)
    aff = img.affine
    voxel_mm = float(abs(aff[0, 0]))
    print(f'  shape={arr.shape}, voxel={voxel_mm*1000:.2f} um, '
          f'size={arr.nbytes/1e9:.1f} GB')

    # Work on a 250 um downsampled copy for CC analysis (faster).
    print('Downsample 4x for CC labelling...')
    t0 = time.time()
    factor = 4
    n_lo = tuple(s // factor for s in arr.shape)
    arr_lo = np.zeros(n_lo, dtype=np.uint16)
    for i in range(factor):
        for j in range(factor):
            for k in range(factor):
                arr_lo = np.maximum(arr_lo,
                    arr[i::factor, j::factor, k::factor]
                       [:n_lo[0], :n_lo[1], :n_lo[2]])
    vox_lo = voxel_mm * factor
    print(f'  lo-res shape {arr_lo.shape}, voxel {vox_lo*1000:.2f} um, '
          f'size {arr_lo.nbytes/1e9:.2f} GB, t={time.time()-t0:.0f} s')

    bone_lo = arr_lo > CC_THRESH
    print(f'  bone fraction (>{CC_THRESH}): {bone_lo.mean():.4f}')
    iters = max(1, int(round(CLOSE_MM / vox_lo)))
    closed_lo = binary_closing(bone_lo, iterations=iters)
    print(f'  closed at {CLOSE_MM} mm ({iters} iter), '
          f'fraction={closed_lo.mean():.4f}')

    print('Connected-component labelling...')
    labeled, n_cc = cc_label(closed_lo)
    sizes = np.bincount(labeled.ravel()); sizes[0] = 0
    print(f'  {n_cc} CCs; top-5 sizes (mL): '
          f'{[float(s)*vox_lo**3/1000 for s in sorted(sizes, reverse=True)[:5]]}')
    cranium_id = int(np.argmax(sizes))
    cranium_mask_lo = (labeled == cranium_id)
    print(f'  selected CC #{cranium_id}, '
          f'volume {sizes[cranium_id]*vox_lo**3/1000:.1f} mL')

    # bbox in lo-res voxel coords
    coords = np.argwhere(cranium_mask_lo)
    lo_idx = coords.min(0)
    hi_idx = coords.max(0) + 1
    margin_vox = max(1, int(round(MARGIN_MM / vox_lo)))
    lo_idx = np.maximum(lo_idx - margin_vox, 0)
    hi_idx = np.minimum(hi_idx + margin_vox,
                          np.asarray(cranium_mask_lo.shape))
    # Convert to native voxel coords
    lo_native = lo_idx * factor
    hi_native = np.minimum(hi_idx * factor, np.asarray(arr.shape))

    print(f'  cranium bbox lo-res: i=[{lo_idx[0]},{hi_idx[0]}) '
          f'j=[{lo_idx[1]},{hi_idx[1]}) k=[{lo_idx[2]},{hi_idx[2]})')
    print(f'  cranium bbox native: i=[{lo_native[0]},{hi_native[0]}) '
          f'j=[{lo_native[1]},{hi_native[1]}) k=[{lo_native[2]},{hi_native[2]})')

    # Convert to world mm for the manuscript
    def vox_to_world_mm(v, axis):
        return v * aff[axis, axis] + aff[axis, 3]
    print(f'  cranium bbox world (mm):')
    for axis, name in enumerate(['x', 'y', 'z']):
        a = vox_to_world_mm(lo_native[axis], axis)
        b = vox_to_world_mm(hi_native[axis], axis)
        print(f'    {name}: [{min(a,b):+.1f}, {max(a,b):+.1f}] mm')

    # Mask out everything outside the bbox. Everything inside the bbox
    # but not in cranium CC is also kept (e.g., brain cavity air,
    # mounting residue inside the bone shell) -- we only want to remove
    # the mandible, not all non-cranium voxels.
    print('Zeroing voxels outside cranium bbox...')
    out = np.zeros_like(arr)
    out[lo_native[0]:hi_native[0],
        lo_native[1]:hi_native[1],
        lo_native[2]:hi_native[2]] = arr[
            lo_native[0]:hi_native[0],
            lo_native[1]:hi_native[1],
            lo_native[2]:hi_native[2]]

    # Also, within the bbox, suppress any small bone CCs that aren't
    # connected to the main cranium CC (e.g., mandible fragments that
    # happened to fall inside the bbox).
    print('Within bbox, suppressing non-cranium bone CCs...')
    bbox_arr = out
    bbox_bone = bbox_arr > CC_THRESH
    bbox_closed = binary_closing(bbox_bone, iterations=iters)
    bbox_labeled, _ = cc_label(bbox_closed)
    bbox_sizes = np.bincount(bbox_labeled.ravel()); bbox_sizes[0] = 0
    bbox_cranium_id = int(np.argmax(bbox_sizes))
    bbox_cranium_mask = (bbox_labeled == bbox_cranium_id)
    # Anything classified as bone (>CC_THRESH) but not in the cranium CC
    # is mandible -> zero it.
    mandible_voxels = bbox_bone & ~bbox_cranium_mask
    n_mandible = int(mandible_voxels.sum())
    print(f'  suppressing {n_mandible} mandible-fragment bone voxels '
          f'({n_mandible*voxel_mm**3/1000:.2f} mL)')
    out[mandible_voxels] = 0

    print(f'Saving cropped aligned-native to {MACAQUE_RAS_NATIVE_ALIGNED}...')
    nib.save(nib.Nifti1Image(out, aff), MACAQUE_RAS_NATIVE_ALIGNED)
    print(f'  wrote ({os.path.getsize(MACAQUE_RAS_NATIVE_ALIGNED)/1e9:.2f} GB)')

    # Invalidate downstream caches by removing them so they get rebuilt.
    for stale in [MACAQUE_RAS_ALIGNED_250UM]:
        if os.path.exists(stale):
            os.remove(stale)
            print(f'  removed stale cache {stale}')

    # Re-build the 250 um cavity-extraction cache from the new aligned native
    downsample_aligned_to_250um()
    print('Done. Next steps: re-run find_alignment_rotation() on the '
          'new 250 um cache for a refined PCA, re-align native if angles '
          'changed substantially, then re-run extract_macaque_cavity.py + '
          'register_macaque_cavity_to_nmt.py.')


if __name__ == '__main__':
    main()
