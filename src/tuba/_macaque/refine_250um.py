"""Apply user-specified additional world-frame rotation + translation
to the 250 um aligned cache (analogous to :mod:`tuba._macaque.refine_native`
but operating at the coarser cache for fast iteration).

Edit the ``REFINE_*`` constants for each iteration. Re-running starts
from the saved ORIG backup so each refinement is from scratch (not
cumulative).

After this runs, :mod:`tuba._macaque.fast_iter_250um` picks up the
refined 250 um cache.
"""
import os
import shutil
import time
import numpy as np
import nibabel as nib
from scipy import ndimage

from .macaque_skull_constants import MACAQUE_RAS_ALIGNED_250UM

# === user refinement (world frame) ==========================================
REFINE_RX_DEG = -22.0   # 22 deg CW in sagittal (DV/AP plane)
REFINE_RY_DEG =  -1.5   # 1.5 deg CW in coronal (DV/RL plane)
REFINE_RZ_DEG =   0.0
REFINE_TX_MM  =  +5.0   # +5 mm in world +x (anatomical right)
REFINE_TY_MM  =   0.0
REFINE_TZ_MM  = -27.0   # 27 mm down (-z, ventral)

# Padding to add around the volume before rotation, so the rotated
# content (especially the forehead under a -22 deg sagittal rotation)
# doesn't get clipped at the volume edges.
PAD_MM = 30.0
# ============================================================================

ORIG_BACKUP = MACAQUE_RAS_ALIGNED_250UM.replace(
    '.nii.gz', '_orig.nii.gz')


def _rot_zyx_world(rx, ry, rz):
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def main():
    if os.path.exists(ORIG_BACKUP):
        src = ORIG_BACKUP
        print(f'Loading ORIG 250um cache from backup: {ORIG_BACKUP}')
    else:
        src = MACAQUE_RAS_ALIGNED_250UM
        print(f'Loading 250um cache (no backup yet): {src}')
        shutil.copy(src, ORIG_BACKUP)
        print(f'  backed up to {ORIG_BACKUP}')

    img = nib.load(src)
    aff = img.affine
    arr = np.asarray(img.dataobj, dtype=np.uint16)
    print(f'  shape {arr.shape}, voxel {abs(aff[0,0])*1000:.1f} um')

    R_world = _rot_zyx_world(REFINE_RX_DEG, REFINE_RY_DEG, REFINE_RZ_DEG)
    T_world = np.array([REFINE_TX_MM, REFINE_TY_MM, REFINE_TZ_MM])
    print(f'\nRefinement (world frame):')
    print(f'  R: Rx={REFINE_RX_DEG:+.2f}  Ry={REFINE_RY_DEG:+.2f}  '
          f'Rz={REFINE_RZ_DEG:+.2f} deg')
    print(f'  T: ({REFINE_TX_MM:+.2f}, {REFINE_TY_MM:+.2f}, '
          f'{REFINE_TZ_MM:+.2f}) mm')

    # Pad the output volume so the rotated content (which sweeps a
    # larger bbox) doesn't get clipped.
    voxel_mm = float(abs(aff[0, 0]))
    pad_vox = int(np.ceil(PAD_MM / voxel_mm))
    out_shape = tuple(s + 2 * pad_vox for s in arr.shape)
    print(f'  padding output by {pad_vox} vox ({PAD_MM} mm) on each side: '
          f'in {arr.shape} -> out {out_shape}')

    # Output affine: same voxel size, translation adjusted so the OUTPUT
    # volume centre is still at the same WORLD point as the INPUT centre
    # (the world origin = bone centroid).
    in_centre_w = aff[:3, :3] @ ((np.array(arr.shape) - 1) / 2.0) + aff[:3, 3]
    out_aff = aff.copy()
    out_centre_v = (np.array(out_shape) - 1) / 2.0
    out_aff[:3, 3] = in_centre_w - aff[:3, :3] @ out_centre_v

    A_in = aff[:3, :3]
    A_in_inv = np.linalg.inv(A_in)
    A_out = out_aff[:3, :3]
    matrix = A_in_inv @ R_world.T @ A_out
    offset = A_in_inv @ (R_world.T @ (out_aff[:3, 3] - T_world)
                          - aff[:3, 3])

    t0 = time.time()
    refined = ndimage.affine_transform(
        arr, matrix, offset=offset, output_shape=out_shape,
        order=1, mode='constant', cval=0, prefilter=False)
    print(f'  affine_transform done in {time.time()-t0:.1f} s')

    bone_idx_out = np.argwhere(refined > 8000)
    if len(bone_idx_out) > 0:
        bw = (out_aff[:3, :3] @ bone_idx_out.T).T + out_aff[:3, 3]
        print(f'  refined bone world bbox: '
              f'x [{bw[:,0].min():+.1f}, {bw[:,0].max():+.1f}], '
              f'y [{bw[:,1].min():+.1f}, {bw[:,1].max():+.1f}], '
              f'z [{bw[:,2].min():+.1f}, {bw[:,2].max():+.1f}] mm')
        out_max = np.array(out_shape) - 1
        on_edge = ((bone_idx_out.min(0) <= 0).any() or
                   (bone_idx_out.max(0) >= out_max).any())
        if on_edge:
            print('  WARNING: bone touches volume edge -- consider larger PAD_MM')
        else:
            print('  bone does not touch volume edge (good)')

    nib.save(nib.Nifti1Image(refined, out_aff), MACAQUE_RAS_ALIGNED_250UM)
    print(f'wrote refined 250um cache (shape {out_shape}) -> '
          f'{MACAQUE_RAS_ALIGNED_250UM}')


if __name__ == '__main__':
    main()
