"""Re-render native_cavity_qc.png using the existing native cavity
(macaque_cavity_native.nii.gz) and a user-specified 250 um bone
backdrop. Memory-efficient: never loads the 82.8 GB refined native CT.

Use case: after fixing align_native padding, the native cavity itself
is fine but the QC backdrop (macaque_skull_250um_iter.nii.gz) inherited
the old truncation. Regenerating the iter cache via fast_iter_250um is
slow; this script swaps in a different 250 um cache (typically the
fresh block-max ``macaque_skull_aligned_native_250um.nii.gz``) so the
backdrop matches the new refined-native extents.

Run from the registration/ directory:
    python rerender_native_cavity_qc.py [--backdrop PATH]
"""
import argparse
import os
from .macaque_skull_constants import REG_DIR
import sys
import time
import numpy as np
import nibabel as nib

from . import native_cavity as nc

CAVITY_NATIVE = os.path.join(REG_DIR, 'macaque_cavity_native.nii.gz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backdrop', default=os.path.join(
        REG_DIR, 'macaque_skull_aligned_native_250um.nii.gz'),
        help='250 um bone NIfTI to paint as backdrop')
    args = ap.parse_args()

    t0 = time.time()
    print(f'Rerender native_cavity_qc.png')
    print(f'  cavity:   {CAVITY_NATIVE}')
    print(f'  backdrop: {args.backdrop}')

    # Temporarily redirect the BONE_250UM_PATH inside _render_qc by
    # symlinking the chosen backdrop to the path _render_qc expects.
    # That keeps native_cavity.py unchanged.
    backdrop_target = os.path.join(REG_DIR, 'macaque_skull_250um_iter.nii.gz')
    backup = None
    if os.path.realpath(backdrop_target) != os.path.realpath(args.backdrop):
        backup = backdrop_target + '.preempt'
        if os.path.lexists(backdrop_target):
            os.replace(backdrop_target, backup)
        os.symlink(args.backdrop, backdrop_target)
        print(f'  symlinked iter-cache slot -> {args.backdrop}')

    try:
        cav_img = nib.load(CAVITY_NATIVE)
        cavity = np.asarray(cav_img.dataobj, dtype=np.uint8)
        aff = cav_img.affine
        vox = abs(aff[0, 0])
        vol_ml = float(cavity.sum()) * vox**3 / 1000
        print(f'  cavity shape {cavity.shape}, voxel {vox*1000:.1f} um, '
              f'volume {vol_ml:.1f} mL')
        print(f'  [{time.time()-t0:.1f} s -- starting QC render]')
        nc._render_qc(cavity, aff, vol_ml)
        print(f'  done in {time.time()-t0:.0f} s')
    finally:
        if backup is not None:
            os.remove(backdrop_target)
            os.rename(backup, backdrop_target)
            print('  restored original iter-cache path')


if __name__ == '__main__':
    main()
