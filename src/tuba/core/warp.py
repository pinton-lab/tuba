"""ANTs SyN wrapper -- the only TUBA module that imports ``ants``.

Convention frozen:

    ants.registration(fixed=atlas, moving=subject, type_of_transform='SyN')

Public API exposes two operations:

* :func:`warp_atlas_into_subject` -- applies the inverse transforms to
  bring an atlas-space volume (template, annotation, mask, ...) into
  the subject's working-resolution grid.
* :func:`warp_subject_into_atlas` -- applies the forward transforms in
  the other direction (subject features into atlas space).

Two registration modes:

* ``"cavity_binary"`` -- subject cavity binary mask vs. atlas brain
  binary mask (mouse Allen CCFv3, macaque NMT v2).
* ``"intensity"`` -- subject CT intensity vs. atlas MRI template
  (human Halle vs. MNI152).

The forward/inverse transform lists are saved with stable filenames so
that ``apply_transforms`` can be called from a separate process /
session without re-running registration.
"""
import os
import shutil
import numpy as np


def _stable_paths(reg_dir, prefix, fwd_suffix='_syn_fwd', inv_suffix='_syn_inv'):
    return {
        'affine': os.path.join(reg_dir, f'{prefix}_affine.mat'),
        'warp':    os.path.join(reg_dir, f'{prefix}_warp.nii.gz'),
        'invwarp': os.path.join(reg_dir, f'{prefix}_invwarp.nii.gz'),
        'fwd_txt': os.path.join(reg_dir, f'{prefix}{fwd_suffix}.txt'),
        'inv_txt': os.path.join(reg_dir, f'{prefix}{inv_suffix}.txt'),
    }


def save_transforms(fwd, inv, reg_dir, prefix):
    """Copy ANTs' /tmp transform files into ``reg_dir`` with stable
    names and write the fwd/inv text lists.

    ``fwd`` and ``inv`` are the lists returned by ``ants.registration``
    under keys ``'fwdtransforms'`` and ``'invtransforms'``.
    """
    paths = _stable_paths(reg_dir, prefix)
    fwd_stable, inv_stable = [], []
    for src in fwd:
        if src.endswith('.mat'):
            dst = paths['affine']
        elif 'InverseWarp' in os.path.basename(src):
            dst = paths['invwarp']
        else:
            dst = paths['warp']
        shutil.copyfile(src, dst)
        fwd_stable.append(dst)
    for src in inv:
        if src.endswith('.mat'):
            inv_stable.append(paths['affine'])
        elif 'InverseWarp' in os.path.basename(src):
            inv_stable.append(paths['invwarp'])
        else:
            inv_stable.append(paths['warp'])
    with open(paths['fwd_txt'], 'w') as f:
        f.write('\n'.join(fwd_stable))
    with open(paths['inv_txt'], 'w') as f:
        f.write('\n'.join(inv_stable))
    return fwd_stable, inv_stable, paths


def load_inv_txt(reg_dir, prefix, inv_txt_path=None, inv_suffix='_syn_inv'):
    """Read saved inverse-transform list.

    ``inv_txt_path`` overrides the (reg_dir, prefix, inv_suffix)
    naming convention -- use it for species with non-default
    filenames (e.g. the human Halle pipeline saves to
    ``ants_syn_inverse.txt``, which doesn't fit the default
    ``{prefix}_syn_inv.txt`` template).
    """
    if inv_txt_path is None:
        inv_txt_path = _stable_paths(reg_dir, prefix,
                                     inv_suffix=inv_suffix)['inv_txt']
    with open(inv_txt_path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_fwd_txt(reg_dir, prefix, fwd_txt_path=None, fwd_suffix='_syn_fwd'):
    """Read saved forward-transform list (see :func:`load_inv_txt`)."""
    if fwd_txt_path is None:
        fwd_txt_path = _stable_paths(reg_dir, prefix,
                                     fwd_suffix=fwd_suffix)['fwd_txt']
    with open(fwd_txt_path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def register_subject_to_atlas(subject_path, atlas_path, reg_dir, prefix,
                              mode='cavity_binary',
                              syn_metric='meansquares',
                              aff_metric='meansquares',
                              verbose=True):
    """Run ANTs SyN with ``fixed=atlas, moving=subject``.

    For ``mode='cavity_binary'`` both ``subject_path`` and ``atlas_path``
    should be binary masks (cavity and brain-mask respectively); for
    ``mode='intensity'`` they are intensity volumes (CT and MRI
    template).

    Saves transforms to ``reg_dir`` with the given ``prefix`` and
    returns ``(fwd_stable, inv_stable, paths)``.
    """
    import ants
    fixed = ants.image_read(atlas_path)
    moving = ants.image_read(subject_path)
    if verbose:
        print(f'  fixed (atlas)   shape={fixed.shape}  spacing={fixed.spacing}')
        print(f'  moving (subject) shape={moving.shape} spacing={moving.spacing}')
        print(f'  mode={mode}, syn_metric={syn_metric}, aff_metric={aff_metric}')
        print('\nRunning ANTs SyN...')
    reg = ants.registration(
        fixed=fixed, moving=moving,
        type_of_transform='SyN',
        aff_metric=aff_metric,
        syn_metric=syn_metric,
        verbose=verbose,
    )
    return save_transforms(reg['fwdtransforms'], reg['invtransforms'],
                           reg_dir, prefix)


def warp_atlas_into_subject(atlas_volume_path, subject_reference_path,
                            reg_dir, prefix, out_path,
                            interpolator='linear', verbose=True):
    """Apply the saved inverse transforms to bring an atlas-space
    volume into the subject's reference grid.

    ``interpolator`` should be ``'linear'`` for templates and
    ``'nearestNeighbor'`` for annotation labels.
    """
    import ants
    inv = load_inv_txt(reg_dir, prefix)
    ref_ants = ants.image_read(subject_reference_path)
    whichtoinvert = [t.endswith('.mat') for t in inv]
    moving = ants.image_read(atlas_volume_path)
    if verbose:
        print(f'apply_transforms target grid: {ref_ants.shape} @ '
              f'{ref_ants.spacing[0]*1000:.1f} um')
        print(f'  warping {atlas_volume_path} -> {out_path} '
              f'({interpolator})')
    warped = ants.apply_transforms(
        fixed=ref_ants, moving=moving, transformlist=inv,
        whichtoinvert=whichtoinvert, interpolator=interpolator)
    ants.image_write(warped, out_path)
    if verbose:
        print(f'  wrote {out_path}')
    return out_path


def warp_subject_into_atlas(subject_volume_path, atlas_reference_path,
                            reg_dir, prefix, out_path,
                            interpolator='linear', verbose=True):
    """Apply the saved forward transforms to bring a subject-space
    volume into the atlas grid.
    """
    import ants
    fwd = load_fwd_txt(reg_dir, prefix)
    ref_ants = ants.image_read(atlas_reference_path)
    moving = ants.image_read(subject_volume_path)
    if verbose:
        print(f'apply_transforms target grid: {ref_ants.shape} @ '
              f'{ref_ants.spacing[0]*1000:.1f} um')
    warped = ants.apply_transforms(
        fixed=ref_ants, moving=moving, transformlist=fwd,
        interpolator=interpolator)
    ants.image_write(warped, out_path)
    if verbose:
        print(f'  wrote {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# Intensity mode: atlas (T1 MRI) registered against subject (CT) with
# fixed=subject. Used by the human Halle pipeline. The "forward" warp
# brings the atlas onto the subject grid directly (no inverse needed).
# ---------------------------------------------------------------------------
def register_atlas_to_subject_intensity(atlas_intensity_path,
                                        subject_intensity_path,
                                        reg_dir, prefix,
                                        reg_iterations=(40, 20, 10),
                                        aff_metric='mattes',
                                        syn_metric='mattes',
                                        syn_sampling=32,
                                        verbose=True):
    """ANTs SyN with ``fixed=subject_intensity, moving=atlas_intensity``.

    Used by the human pipeline where the atlas (MNI T1) is an
    intensity image rather than a binary mask. The cross-modality
    nature (CT vs T1) makes mutual-information metrics tractable.

    With this convention:
    * ``fwd`` brings the atlas onto the subject grid (atlas image
      voxels resampled at subject-frame locations);
    * ``inv`` brings the subject onto the atlas grid.

    Note that this is the *opposite* direction from
    :func:`register_subject_to_atlas` (which keeps the atlas as
    fixed). The species module declares which convention applies.
    """
    import ants
    fixed = ants.image_read(subject_intensity_path)
    moving = ants.image_read(atlas_intensity_path)
    if verbose:
        print(f'  fixed (subject CT) shape={fixed.shape}, '
              f'spacing={fixed.spacing}, origin={fixed.origin}')
        print(f'  moving (atlas T1) shape={moving.shape}, '
              f'spacing={moving.spacing}, origin={moving.origin}')
        print(f'  reg_iterations={reg_iterations}, aff_metric={aff_metric}, '
              f'syn_metric={syn_metric}')
        print('\nRunning ANTs SyN (intensity mode)...')
    result = ants.registration(
        fixed=fixed, moving=moving,
        type_of_transform='SyN',
        reg_iterations=reg_iterations,
        aff_metric=aff_metric,
        syn_metric=syn_metric,
        syn_sampling=syn_sampling,
        verbose=verbose,
    )
    return save_transforms(result['fwdtransforms'], result['invtransforms'],
                           reg_dir, prefix)


def warp_atlas_image_into_subject_via_fwd(atlas_volume_path,
                                          subject_reference_path,
                                          reg_dir, prefix, out_path,
                                          interpolator='linear',
                                          fwd_txt_path=None,
                                          verbose=True):
    """Apply saved *forward* transforms to bring an atlas-space image
    onto the subject grid. Used by the intensity mode (atlas was
    moving with subject fixed; fwd transforms atlas -> subject).

    For the cavity-binary mode, use :func:`warp_atlas_into_subject`
    instead (which applies inverse transforms).
    """
    import ants
    fwd = load_fwd_txt(reg_dir, prefix, fwd_txt_path=fwd_txt_path)
    ref_ants = ants.image_read(subject_reference_path)
    moving = ants.image_read(atlas_volume_path)
    if verbose:
        print(f'apply_transforms target grid: {ref_ants.shape} @ '
              f'{ref_ants.spacing[0]*1000:.1f} mm')
        print(f'  warping {atlas_volume_path} -> {out_path} '
              f'({interpolator})')
    warped = ants.apply_transforms(
        fixed=ref_ants, moving=moving, transformlist=fwd,
        interpolator=interpolator)
    ants.image_write(warped, out_path)
    if verbose:
        print(f'  wrote {out_path}')
    return out_path


def apply_transforms_to_points_atlas_to_subject_intensity(
        points_atlas_lps, reg_dir, prefix, inv_txt_path=None):
    """Map a list of points from atlas LPS world to subject LPS world,
    intensity-mode convention.

    ANTs' ``apply_transforms_to_points`` operates in LPS world and
    applies the *spatial inverse* of the corresponding image warp.
    For the intensity mode (subject was fixed during registration),
    image fwd transforms atlas -> subject; therefore points use the
    *inv* transform list to go atlas -> subject.

    Parameters
    ----------
    points_atlas_lps : (N, 3) array-like
        Points in the atlas's LPS world frame (ANTs convention; for
        an MNI RAS coord, negate the first two components).
    inv_txt_path : str, optional
        Explicit path to the inverse-transform text list, overriding
        the (reg_dir, prefix) convention. Used by the human Halle
        pipeline whose legacy filename is ``ants_syn_inverse.txt``.

    Returns
    -------
    points_subject_lps : (N, 3) ndarray
    """
    import ants
    import pandas as pd
    inv = load_inv_txt(reg_dir, prefix, inv_txt_path=inv_txt_path)
    pts = np.atleast_2d(np.asarray(points_atlas_lps, dtype=np.float64))
    df_in = pd.DataFrame({'x': pts[:, 0], 'y': pts[:, 1], 'z': pts[:, 2]})
    df_out = ants.apply_transforms_to_points(
        dim=3, points=df_in, transformlist=inv)
    return df_out[['x', 'y', 'z']].to_numpy()
