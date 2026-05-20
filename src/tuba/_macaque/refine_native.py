"""Apply the user-specified refinement (-22 deg sagittal, -1.5 deg
coronal, +5 mm right, -27 mm down) to the NATIVE 60.6 um aligned
cache, mirroring refine_250um.py.

This brings the native cache into the same world coordinate system
as the refined 250 um cache, so the 250 um affine transform from
the cavity-fitting step can be applied directly to native bone /
cavity / acoustic-property work.

Reads from MACAQUE_RAS_NATIVE_ALIGNED_orig backup and writes to
MACAQUE_RAS_NATIVE_ALIGNED. Does NOT touch the 250 um cache (which
already has its own refinement).

Runtime: ~10-30 min for ~11 GB volume affine_transform.
"""
import os
import sys
import shutil
import subprocess
import time
import numpy as np
import nibabel as nib
from scipy import ndimage

from .macaque_skull_constants import MACAQUE_RAS_NATIVE_ALIGNED, _save_nifti_pigz

# Mirror refine_250um.py constants exactly
REFINE_RX_DEG = -22.0
REFINE_RY_DEG =  -1.5
REFINE_RZ_DEG =   0.0
REFINE_TX_MM  =  +5.0
REFINE_TY_MM  =   0.0
REFINE_TZ_MM  = -27.0
PAD_MM = 30.0

ORIG_BACKUP = MACAQUE_RAS_NATIVE_ALIGNED.replace('.nii.gz', '_orig.nii.gz')


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
        print(f'Loading ORIG native cache from backup: {ORIG_BACKUP}')
    else:
        src = MACAQUE_RAS_NATIVE_ALIGNED
        print(f'Loading native cache (no backup yet): {src}')
        shutil.copy(src, ORIG_BACKUP)
        print(f'  backed up to {ORIG_BACKUP}')

    t0 = time.time()
    img = nib.load(src)
    aff = img.affine
    voxel_mm = float(abs(aff[0, 0]))
    print(f'  shape {img.shape}, voxel {voxel_mm*1000:.2f} um')
    print(f'  loading {np.prod(img.shape)*2/1e9:.1f} GB into memory...')
    arr = np.asarray(img.dataobj, dtype=np.uint16)
    print(f'  loaded, t={time.time()-t0:.0f} s')

    R_world = _rot_zyx_world(REFINE_RX_DEG, REFINE_RY_DEG, REFINE_RZ_DEG)
    T_world = np.array([REFINE_TX_MM, REFINE_TY_MM, REFINE_TZ_MM])
    print(f'\nRefinement (world frame):')
    print(f'  R: Rx={REFINE_RX_DEG:+.2f}  Ry={REFINE_RY_DEG:+.2f}  '
          f'Rz={REFINE_RZ_DEG:+.2f} deg')
    print(f'  T: ({REFINE_TX_MM:+.2f}, {REFINE_TY_MM:+.2f}, '
          f'{REFINE_TZ_MM:+.2f}) mm')

    # Pad the output volume (same logic as refine_250um.py).
    pad_vox = int(np.ceil(PAD_MM / voxel_mm))
    out_shape = tuple(s + 2 * pad_vox for s in arr.shape)
    out_gb = np.prod(out_shape) * 2 / 1e9
    print(f'  padding by {pad_vox} vox ({PAD_MM} mm) on each side')
    print(f'  output shape: {out_shape} = {out_gb:.1f} GB')

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

    print(f'\nApplying scipy.ndimage.affine_transform...')
    t1 = time.time()
    refined = ndimage.affine_transform(
        arr, matrix, offset=offset, output_shape=out_shape,
        order=1, mode='constant', cval=0, prefilter=False)
    print(f'  affine_transform done in {time.time()-t1:.1f} s')

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

    # Write via parallel gzip (pigz) when available; falls back to
    # nibabel's single-thread zlib gzip otherwise. For an ~80 GB uint16
    # volume single-thread gzip costs ~30 min on this box, vs ~1 min
    # for pigz -p 64. Writes the uncompressed .nii to the realpath of
    # the .nii.gz target (so a symlink to NVMe staging stays honored),
    # then pigz replaces it with .nii.gz in place.
    print(f'\nWriting refined native ({out_gb:.1f} GB)...')
    out_resolved = os.path.realpath(MACAQUE_RAS_NATIVE_ALIGNED)
    tmp_nii = out_resolved[:-3] if out_resolved.endswith('.gz') else out_resolved + '.nii'
    use_pigz = shutil.which('pigz') is not None
    if use_pigz:
        t_w = time.time()
        nib.save(nib.Nifti1Image(refined, out_aff), tmp_nii)
        raw_gb = os.path.getsize(tmp_nii) / 1e9
        print(f'  uncompressed {raw_gb:.1f} GB written in {time.time()-t_w:.0f} s')
        t_p = time.time()
        n_threads = min(64, os.cpu_count() or 1)
        subprocess.run(['pigz', '-f', '-p', str(n_threads), tmp_nii],
                        check=True)
        gz_mb = os.path.getsize(out_resolved) / 1e6
        print(f'  pigz ({n_threads} threads) compressed to {gz_mb:.0f} MB '
              f'in {time.time()-t_p:.0f} s')
    else:
        nib.save(nib.Nifti1Image(refined, out_aff), MACAQUE_RAS_NATIVE_ALIGNED)
    print(f'wrote refined native -> {MACAQUE_RAS_NATIVE_ALIGNED}')
    print(f'\nDone in {time.time()-t0:.0f} s.')


if __name__ == '__main__':
    main()
