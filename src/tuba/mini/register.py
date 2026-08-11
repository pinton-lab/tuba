"""Register the ITRUSST endocranial cavity to the MNI152 brain mask, and
warp MNI-space volumes onto the skull grid.

This is the piece that gives the bare ITRUSST benchmark skull an atlas:
it reuses TUBA's frozen ``cavity_binary`` SyN convention
(``fixed = atlas brain mask``, ``moving = subject cavity mask``) -- the
same path the mouse (Allen CCFv3) and macaque (NMT v2) species bindings
use -- so any MNI-space volume (the brain mask, a parcellation, a
probabilistic nucleus, a target marker) can be resampled onto the skull.

``antspyx`` is imported lazily inside :mod:`tuba.core.warp`, so importing
this module does *not* require ANTs. ANTs is only needed to (re)generate
the transforms via :func:`register`; once the transforms are cached, the
warp/consume path likewise defers to ``tuba.core.warp`` which imports
ANTs only when a warp is actually applied. For a fully ANTs-free demo,
ship the small warped products (see :mod:`tuba.mini.demo`).
"""
import os
import numpy as np

from tuba.core import warp

PREFIX = 'mni2itrusst'


def register(cavity_path, mni_brain_mask_path, reg_dir, prefix=PREFIX,
             verbose=True):
    """SyN-register the skull cavity mask to the MNI brain mask.

    ``fixed = MNI brain mask``, ``moving = ITRUSST cavity`` (TUBA's
    ``cavity_binary`` convention). The benchmark skull is already
    MNI-aligned to within an affine, so this starts near identity and
    the deformable stage only refines the vault. Transforms are saved to
    ``reg_dir`` with ``prefix`` for later reuse. Returns
    ``(fwd_stable, inv_stable, paths)``.
    """
    os.makedirs(reg_dir, exist_ok=True)
    return warp.register_subject_to_atlas(
        subject_path=cavity_path,
        atlas_path=mni_brain_mask_path,
        reg_dir=reg_dir, prefix=prefix,
        mode='cavity_binary',
        syn_metric='meansquares', aff_metric='meansquares',
        verbose=verbose)


def warp_into_skull(mni_volume_path, skull_reference_path, reg_dir,
                    out_path, prefix=PREFIX, interpolator='linear',
                    verbose=True):
    """Resample an MNI-space volume onto the skull grid using the saved
    transforms (applies the inverse transforms, per the cavity_binary
    convention). Use ``interpolator='nearestNeighbor'`` for label maps,
    ``'linear'`` for templates / probability maps."""
    return warp.warp_atlas_into_subject(
        atlas_volume_path=mni_volume_path,
        subject_reference_path=skull_reference_path,
        reg_dir=reg_dir, prefix=prefix, out_path=out_path,
        interpolator=interpolator, verbose=verbose)


def mni_point_marker(center_ras_mm, reference_nifti, out_path,
                     radius_mm=4.0, verbose=True):
    """Write a small solid sphere at an MNI RAS-mm coordinate onto the
    grid of ``reference_nifti`` (typically the MNI T1 or brain mask).

    Warping this marker into skull space and taking its centroid is a
    transform-direction-agnostic way to locate a named MNI target on the
    skull: it rides the exact image-warp path used for the atlas, so it
    can never disagree with where the parcellation lands.
    """
    import nibabel as nib
    ref = nib.load(reference_nifti)
    inv = np.linalg.inv(ref.affine)
    c = np.asarray(center_ras_mm, dtype=np.float64)
    cijk = inv[:3, :3] @ c + inv[:3, 3]
    # grid of voxel indices within a bounding box around the centre
    r_vox = int(np.ceil(radius_mm / np.abs(np.diag(ref.affine)[:3]).min())) + 1
    ci, cj, ck = np.round(cijk).astype(int)
    sh = ref.shape[:3]
    vol = np.zeros(sh, dtype=np.uint8)
    ii, jj, kk = np.meshgrid(
        np.arange(max(ci - r_vox, 0), min(ci + r_vox + 1, sh[0])),
        np.arange(max(cj - r_vox, 0), min(cj + r_vox + 1, sh[1])),
        np.arange(max(ck - r_vox, 0), min(ck + r_vox + 1, sh[2])),
        indexing='ij')
    world = nib.affines.apply_affine(
        ref.affine, np.stack([ii, jj, kk], axis=-1))
    inside = np.linalg.norm(world - c, axis=-1) <= radius_mm
    vol[ii[inside], jj[inside], kk[inside]] = 1
    nib.save(nib.Nifti1Image(vol, ref.affine), out_path)
    if verbose:
        print(f'  wrote marker @ {tuple(np.round(c,1))} -> {out_path} '
              f'({int(vol.sum())} vox)')
    return out_path
