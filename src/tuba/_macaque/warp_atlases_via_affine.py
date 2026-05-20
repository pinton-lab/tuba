"""Re-fit ANTs Affine (bone-shell cavity -> NMT brain mask) at
250 um, save the .mat persistently, and warp NMT template, brain
mask, CHARM (5D), SARM (5D), and D99 (3D) into macaque space.

Mirrors the registration step inside fast_iter_250um.py but also
writes the affine transform to disk and applies it to the four
NMT-bundled atlases. Outputs go to registration/ with
'_in_macaque.nii.gz' suffix.

Runtime: ~10 min on 8 cores.
"""
import os
import sys
import time
import shutil
import numpy as np
import nibabel as nib
import ants

from .macaque_skull_constants import REG_DIR, MACAQUE_RAS_ALIGNED_250UM

NMT_DIR = os.path.join(REG_DIR, '..', 'templates', 'nmt_v2',
                        'NMT_v2.0_sym', 'NMT_v2.0_sym_fh')
NMT_FH_TEMPLATE  = os.path.join(NMT_DIR, 'NMT_v2.0_sym_fh.nii.gz')
NMT_FH_BRAINMASK = os.path.join(NMT_DIR, 'NMT_v2.0_sym_fh_brainmask.nii.gz')
NMT_FH_CHARM     = os.path.join(NMT_DIR, 'CHARM_in_NMT_v2.0_sym_fh.nii.gz')
NMT_FH_SARM      = os.path.join(NMT_DIR, 'SARM_in_NMT_v2.0_sym_fh.nii.gz')
NMT_FH_D99       = os.path.join(NMT_DIR, 'D99_atlas_in_NMT_v2.0_sym_fh.nii.gz')

BONESHELL_NII = os.path.join(REG_DIR, 'macaque_cavity_boneshell_250um.nii.gz')
SKULL_250UM   = os.path.join(REG_DIR, 'macaque_skull_250um_iter.nii.gz')

OUT_AFFINE_MAT = os.path.join(REG_DIR, 'cavity_to_nmt_affine_iter21.mat')
OUT_TEMPLATE   = os.path.join(REG_DIR, 'nmt_template_in_macaque.nii.gz')
OUT_BRAINMASK  = os.path.join(REG_DIR, 'nmt_brainmask_in_macaque.nii.gz')
OUT_CHARM      = os.path.join(REG_DIR, 'charm_in_macaque.nii.gz')
OUT_SARM       = os.path.join(REG_DIR, 'sarm_in_macaque.nii.gz')
OUT_D99        = os.path.join(REG_DIR, 'd99_in_macaque.nii.gz')

TARGET_SPACING_MM = 0.25   # work at NMT pitch


def main():
    t0 = time.time()
    print('=== warp atlases via Affine (250 um) ===')

    print('\nStep 1: load inputs...')
    moving = ants.image_read(BONESHELL_NII)
    nmt_bm = ants.image_read(NMT_FH_BRAINMASK)
    nmt_bm_lo = ants.resample_image(nmt_bm, (TARGET_SPACING_MM,) * 3,
                                     use_voxels=False, interp_type=1)
    fixed_mac = ants.image_read(SKULL_250UM)
    print(f'  bone-shell {moving.shape} @ {moving.spacing[0]*1000:.0f} um')
    print(f'  NMT brain mask {nmt_bm_lo.shape} @ '
          f'{nmt_bm_lo.spacing[0]*1000:.0f} um')
    print(f'  macaque CT cache (fixed grid) {fixed_mac.shape}')
    print(f'  [{time.time()-t0:.1f} s]')

    print('\nStep 2: ANTs Affine (cavity -> NMT brain mask)...')
    reg = ants.registration(
        fixed=nmt_bm_lo, moving=moving,
        type_of_transform='Affine',
        aff_metric='meansquares',
        verbose=False,
    )
    fwd = reg['fwdtransforms']
    inv = reg['invtransforms']
    print(f'  fwd transforms: {fwd}')
    print(f'  inv transforms: {inv}')

    # The Affine returned is a single .mat file. Copy it to a stable name.
    affine_mat = [t for t in fwd if t.endswith('.mat')]
    if not affine_mat:
        raise RuntimeError(f'No .mat in fwd transforms: {fwd}')
    shutil.copy(affine_mat[0], OUT_AFFINE_MAT)
    print(f'  saved Affine -> {OUT_AFFINE_MAT}')
    print(f'  [{time.time()-t0:.1f} s]')

    def _warp(atlas_path, out_path, interp='linear'):
        atlas = ants.image_read(atlas_path)
        warped = ants.apply_transforms(
            fixed=fixed_mac, moving=atlas, transformlist=inv,
            whichtoinvert=[t.endswith('.mat') for t in inv],
            interpolator=interp)
        ants.image_write(warped, out_path)
        print(f'  wrote {os.path.basename(out_path)} '
              f'(shape {warped.shape}, interp={interp})')

    print('\nStep 3: warp NMT template + brain mask...')
    _warp(NMT_FH_TEMPLATE, OUT_TEMPLATE, 'linear')
    _warp(NMT_FH_BRAINMASK, OUT_BRAINMASK, 'linear')
    print(f'  [{time.time()-t0:.1f} s]')

    print('\nStep 4: warp D99 (3D)...')
    _warp(NMT_FH_D99, OUT_D99, 'genericLabel')
    print(f'  [{time.time()-t0:.1f} s]')

    print('\nStep 5: warp CHARM (5D per-level)...')
    _warp_5d(NMT_FH_CHARM, OUT_CHARM, fixed_mac, inv)
    print(f'  [{time.time()-t0:.1f} s]')

    print('\nStep 6: warp SARM (5D per-level)...')
    _warp_5d(NMT_FH_SARM, OUT_SARM, fixed_mac, inv)
    print(f'  [{time.time()-t0:.1f} s]')

    print(f'\nDone in {time.time()-t0:.0f} s.')


def _warp_5d(atlas_path, out_path, fixed_mac, inv):
    """Warp a 5D (x,y,z,1,6) hierarchy atlas: extract each level, warp
    independently with genericLabel, re-stack to 5D."""
    img = nib.load(atlas_path)
    arr = np.asarray(img.dataobj)
    aff = img.affine
    if arr.ndim != 5 or arr.shape[3] != 1 or arr.shape[4] != 6:
        raise ValueError(f'expected (x,y,z,1,6), got {arr.shape}')
    print(f'    input 5D shape {arr.shape}')

    warped_levels = []
    tmpdir = '/tmp/_warp5d'
    os.makedirs(tmpdir, exist_ok=True)
    for level in range(6):
        level_arr = arr[:, :, :, 0, level].astype(np.int32)
        tmp_in = os.path.join(tmpdir, f'level{level}_in.nii.gz')
        tmp_out = os.path.join(tmpdir, f'level{level}_out.nii.gz')
        nib.save(nib.Nifti1Image(level_arr,
                                  aff[:4, :4] if aff.shape == (4, 4) else aff),
                 tmp_in)
        level_ants = ants.image_read(tmp_in)
        warped = ants.apply_transforms(
            fixed=fixed_mac, moving=level_ants, transformlist=inv,
            whichtoinvert=[t.endswith('.mat') for t in inv],
            interpolator='genericLabel')
        ants.image_write(warped, tmp_out)
        warped_levels.append(np.asarray(nib.load(tmp_out).dataobj))
        print(f'    level {level+1}/6 warped '
              f'(unique IDs: {len(np.unique(warped_levels[-1]))})')

    stacked = np.stack([w[..., None] for w in warped_levels], axis=-1)
    # final shape (x, y, z, 1, 6)
    print(f'    stacked 5D shape {stacked.shape}')
    out_img = nib.load(tmp_out)
    nib.save(nib.Nifti1Image(stacked.astype(np.int32), out_img.affine),
             out_path)
    print(f'    wrote {os.path.basename(out_path)}')


if __name__ == '__main__':
    main()
