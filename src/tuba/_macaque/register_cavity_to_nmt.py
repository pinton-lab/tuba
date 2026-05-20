"""ANTs SyN: register the AMNH macaque endocranial cavity to the NMT v2
brain template (cavity = moving, NMT brain mask = fixed). The inverse
warp brings any NMT-space volume back into macaque skull space.

Direct port of ``mouse_therapy/registration/register_maga_cavity_to_allen.py``
with NMT v2-specific paths. Differences vs the Maga case:
  * NMT v2 ships its own brain mask (``NMT_v2.0_sym_brainmask.nii.gz``),
    so we don't need to derive one from a thresholded template.
  * NMT v2 is at 0.25 mm isotropic; the macaque 250 um cache is at
    0.2424 mm. Both are on similar grids -- SyN converges quickly.
  * The warped products are sourced from the same 0.25 mm NMT
    template (no separate higher-resolution template like Allen 10 um).
    The CHARM/SARM/D99 atlases ship in the same 0.25 mm NMT grid so
    we warp them through the same inverse transform.

Outputs (in REG_DIR):
  macaque_cavity_to_nmt_affine.mat
  macaque_cavity_to_nmt_warp.nii.gz
  macaque_cavity_to_nmt_invwarp.nii.gz
  macaque_cavity_to_nmt_syn_fwd.txt
  macaque_cavity_to_nmt_syn_inv.txt
  nmt_template_in_macaque.nii.gz       (linear, for display)
  nmt_brainmask_in_macaque.nii.gz      (nearestNeighbor)
  charm_in_macaque.nii.gz              (5D 1-x-6 cortical hierarchy)
  sarm_in_macaque.nii.gz               (5D 1-x-6 subcortical hierarchy)
  d99_in_macaque.nii.gz                (3D D99 v2)

This is the SyN alternative to the production Affine recipe
(:mod:`tuba._macaque.fast_iter_250um` + :mod:`tuba._macaque.warp_atlases_via_affine`).
See manuscript Appendix A (cavity-extraction trials, Table 2) for why
the Affine recipe was chosen over SyN in production.
"""
import os
import shutil
import sys
import numpy as np
import nibabel as nib
import ants

from .macaque_skull_constants import REG_DIR, NMT_DIR

# Inputs
CAVITY_NII = os.path.join(REG_DIR, 'macaque_cranial_cavity.nii.gz')
MACAQUE_250UM_REF = os.path.join(REG_DIR,
                                  'macaque_skull_aligned_native_250um.nii.gz')
NMT_TEMPLATE = os.path.join(NMT_DIR, 'NMT_v2.0_sym.nii.gz')
NMT_BRAINMASK = os.path.join(NMT_DIR, 'NMT_v2.0_sym_brainmask.nii.gz')
NMT_CHARM = os.path.join(NMT_DIR, 'CHARM_in_NMT_v2.0_sym.nii.gz')
NMT_SARM = os.path.join(NMT_DIR, 'SARM_in_NMT_v2.0_sym.nii.gz')
NMT_D99 = os.path.join(NMT_DIR, 'D99_atlas_in_NMT_v2.0_sym.nii.gz')

# Stable on-disk transform paths
AFFINE_PATH = os.path.join(REG_DIR, 'macaque_cavity_to_nmt_affine.mat')
WARP_PATH = os.path.join(REG_DIR, 'macaque_cavity_to_nmt_warp.nii.gz')
INVWARP_PATH = os.path.join(REG_DIR, 'macaque_cavity_to_nmt_invwarp.nii.gz')
FWD_TXT = os.path.join(REG_DIR, 'macaque_cavity_to_nmt_syn_fwd.txt')
INV_TXT = os.path.join(REG_DIR, 'macaque_cavity_to_nmt_syn_inv.txt')

# Warped outputs in macaque space
WARPED_TEMPLATE = os.path.join(REG_DIR, 'nmt_template_in_macaque.nii.gz')
WARPED_BRAINMASK = os.path.join(REG_DIR, 'nmt_brainmask_in_macaque.nii.gz')
WARPED_CHARM = os.path.join(REG_DIR, 'charm_in_macaque.nii.gz')
WARPED_SARM = os.path.join(REG_DIR, 'sarm_in_macaque.nii.gz')
WARPED_D99 = os.path.join(REG_DIR, 'd99_in_macaque.nii.gz')


def _save_transforms(fwd, inv):
    """Copy ANTs' /tmp transform files into REG_DIR with stable names.

    Iterates over BOTH lists so the InverseWarp file (which only appears
    in `inv`) is copied. ANTs typically returns:
      fwd = [warp.nii.gz, affine.mat]      (apply moving -> fixed)
      inv = [affine.mat, invwarp.nii.gz]   (apply fixed -> moving)
    """
    def _classify(src):
        if src.endswith('.mat'):
            return AFFINE_PATH
        if 'InverseWarp' in os.path.basename(src):
            return INVWARP_PATH
        return WARP_PATH

    seen = set()
    for src in list(fwd) + list(inv):
        dst = _classify(src)
        if dst in seen:
            continue
        seen.add(dst)
        shutil.copyfile(src, dst)

    fwd_stable = [_classify(s) for s in fwd]
    inv_stable = [_classify(s) for s in inv]
    with open(FWD_TXT, 'w') as f:
        f.write('\n'.join(fwd_stable))
    with open(INV_TXT, 'w') as f:
        f.write('\n'.join(inv_stable))
    return fwd_stable, inv_stable


def _read_inv_txt():
    with open(INV_TXT) as f:
        return [ln.strip() for ln in f if ln.strip()]


def warp_nmt_to_macaque(inv=None):
    """Warp NMT template, brain mask, and the CHARM/SARM/D99 atlas
    label volumes back into macaque space using the saved inverse
    transforms. The macaque 250 um cache is the target grid."""
    if inv is None:
        inv = _read_inv_txt()
    if not os.path.exists(MACAQUE_250UM_REF):
        raise FileNotFoundError(
            f'250 um macaque reference missing: {MACAQUE_250UM_REF}. Run '
            f'macaque_skull_constants.downsample_aligned_to_250um() first.')
    ref_ants = ants.image_read(MACAQUE_250UM_REF)
    whichtoinvert = [t.endswith('.mat') for t in inv]
    print(f'apply_transforms target grid: {ref_ants.shape} @ '
          f'{ref_ants.spacing[0]*1000:.1f} um')

    print('Warping NMT template (linear)...')
    template_ants = ants.image_read(NMT_TEMPLATE)
    warped_t = ants.apply_transforms(
        fixed=ref_ants, moving=template_ants, transformlist=inv,
        whichtoinvert=whichtoinvert, interpolator='linear')
    ants.image_write(warped_t, WARPED_TEMPLATE)
    print(f'  wrote {WARPED_TEMPLATE}')

    print('Warping NMT brain mask (nearestNeighbor)...')
    bm_ants = ants.image_read(NMT_BRAINMASK)
    warped_bm = ants.apply_transforms(
        fixed=ref_ants, moving=bm_ants, transformlist=inv,
        whichtoinvert=whichtoinvert, interpolator='nearestNeighbor')
    ants.image_write(warped_bm, WARPED_BRAINMASK)
    print(f'  wrote {WARPED_BRAINMASK}')

    print('Warping D99 (genericLabel)...')
    atl_ants = ants.image_read(NMT_D99)
    warped_atl = ants.apply_transforms(
        fixed=ref_ants, moving=atl_ants, transformlist=inv,
        whichtoinvert=whichtoinvert, interpolator='genericLabel')
    ants.image_write(warped_atl, WARPED_D99)
    print(f'  wrote {WARPED_D99}')

    # CHARM and SARM are 5D (x,y,z,1,6) -- ANTs python wrapper doesn't
    # accept 5D. We loop over the 6 hierarchy levels, warping each as
    # a 3D label volume, and stack the results back into 5D so that the
    # placement helper's `arr[..., 0, 5]` indexing for L6 still works.
    for label, src, dst in [
        ('CHARM', NMT_CHARM, WARPED_CHARM),
        ('SARM',  NMT_SARM,  WARPED_SARM),
    ]:
        print(f'Warping {label} (5D, per-level genericLabel)...')
        full = nib.load(src)
        full_arr = np.asarray(full.dataobj)
        if full_arr.ndim != 5 or full_arr.shape[3] != 1:
            raise ValueError(f'{label} has unexpected shape {full_arr.shape}')
        n_levels = full_arr.shape[4]
        warped_arr = None
        for lev in range(n_levels):
            level_arr = full_arr[:, :, :, 0, lev].astype(np.int16)
            level_img = nib.Nifti1Image(level_arr, full.affine)
            tmp_path = os.path.join(REG_DIR, f'_tmp_{label}_L{lev+1}.nii.gz')
            nib.save(level_img, tmp_path)
            try:
                level_ants = ants.image_read(tmp_path)
                w = ants.apply_transforms(
                    fixed=ref_ants, moving=level_ants, transformlist=inv,
                    whichtoinvert=whichtoinvert, interpolator='genericLabel')
                w_arr = w.numpy().astype(np.int16)
                if warped_arr is None:
                    warped_arr = np.zeros(
                        w_arr.shape + (1, n_levels), dtype=np.int16)
                warped_arr[:, :, :, 0, lev] = w_arr
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            print(f'  L{lev+1}: warped, max ID = {w_arr.max()}')
        ref_img = nib.load(MACAQUE_250UM_REF)
        nib.save(nib.Nifti1Image(warped_arr, ref_img.affine), dst)
        print(f'  wrote {dst}  (5D {warped_arr.shape})')


def main():
    print('Registering macaque cavity -> NMT v2 brain mask (SyN)...')
    print(f'  fixed (NMT brain mask) = {NMT_BRAINMASK}')
    print(f'  moving (cavity)        = {CAVITY_NII}')

    fixed = ants.image_read(NMT_BRAINMASK)
    moving = ants.image_read(CAVITY_NII)
    moving_arr = moving.numpy().astype(np.float32)
    moving_img = moving.new_image_like(moving_arr)
    print(f'  moving (cavity) shape={moving.shape} '
          f'spacing={moving.spacing} voxels={int(moving_arr.sum())}')
    print(f'  fixed  (NMT)    shape={fixed.shape}  spacing={fixed.spacing}')

    print('\nRunning ANTs SyN...')
    reg = ants.registration(
        fixed=fixed, moving=moving_img,
        type_of_transform='SyN',
        aff_metric='meansquares',
        syn_metric='meansquares',
        verbose=True,
    )
    fwd_stable, inv_stable = _save_transforms(
        reg['fwdtransforms'], reg['invtransforms'])
    print(f'  saved fwd transforms: {fwd_stable}')
    print(f'  saved inv transforms: {inv_stable}')

    warp_nmt_to_macaque(inv=inv_stable)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'warp-only':
        warp_nmt_to_macaque()
    else:
        main()
