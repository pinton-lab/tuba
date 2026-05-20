"""Render the warped-atlas-on-macaque-skull figure for the macaque
manuscript:

  nmt_atlases_in_macaque.png   3x4 grid: macaque skull + warped NMT T1,
                                CHARM L6, SARM L6, D99. Sagittal,
                                coronal, and axial orthoslices through
                                the production-cavity centroid.

This is the warped-into-skull-space companion to ``nmt_atlases.png``
(which shows the same atlases in NMT native space). Parallels the
human ``fig4_warped_atlases_on_halle`` figure.

Inputs (paths resolved through tuba._macaque.macaque_skull_constants,
which honours $TUBA_MACAQUE_REG_DIR with a legacy-cache fallback):
  Files read from REG_DIR:
    macaque_skull_aligned_native_250um.nii.gz   bone backdrop
    nmt_template_in_macaque.nii.gz              warped NMT T1
    charm_in_macaque.nii.gz                     warped CHARM (5-D)
    sarm_in_macaque.nii.gz                      warped SARM (5-D)
    d99_in_macaque.nii.gz                       warped D99
    macaque_cranial_cavity.nii.gz               cavity (for slice picker)

Run:
    cd docs/macaque
    python make_atlas_in_macaque_figure.py
"""
import os
import sys
import time
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse tuba's REG_DIR resolution so the doc-figure script reads from
# the same cache the pipeline writes to.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))
from tuba._macaque.macaque_skull_constants import REG_DIR  # noqa: E402

SKULL_250UM = os.path.join(REG_DIR, 'macaque_skull_aligned_native_250um.nii.gz')
NMT_IN_MACAQUE   = os.path.join(REG_DIR, 'nmt_template_in_macaque.nii.gz')
CHARM_IN_MACAQUE = os.path.join(REG_DIR, 'charm_in_macaque.nii.gz')
SARM_IN_MACAQUE  = os.path.join(REG_DIR, 'sarm_in_macaque.nii.gz')
D99_IN_MACAQUE   = os.path.join(REG_DIR, 'd99_in_macaque.nii.gz')
CAVITY_PROD      = os.path.join(REG_DIR, 'macaque_cranial_cavity.nii.gz')

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
OUT = os.path.join(OUT_DIR, 'nmt_atlases_in_macaque.png')

SAVE_DPI = 200


def _extents_for_affine(arr, aff):
    ni, nj, nk = arr.shape[:3]
    sag = [aff[1, 3], aff[1, 3] + (nj - 1) * aff[1, 1],
           aff[2, 3], aff[2, 3] + (nk - 1) * aff[2, 2]]
    cor = [aff[0, 3], aff[0, 3] + (ni - 1) * aff[0, 0],
           aff[2, 3], aff[2, 3] + (nk - 1) * aff[2, 2]]
    ax_e = [aff[0, 3], aff[0, 3] + (ni - 1) * aff[0, 0],
            aff[1, 3], aff[1, 3] + (nj - 1) * aff[1, 1]]
    return sag, cor, ax_e


def _world_to_vox(aff, world):
    return np.linalg.solve(aff, np.array([*world, 1]))[:3]


def _annot_to_rgba(annot_slice, palette='gist_ncar', alpha_fg=0.7,
                   shuffle_prime=104729):
    """ID-hash to gist_ncar palette, background transparent. Lower
    alpha_fg here than in nmt_atlases.png because we want the bone
    backdrop visible underneath."""
    cmap = plt.get_cmap(palette)
    N = 4096
    rgba = np.zeros((*annot_slice.shape, 4), dtype=np.float32)
    for label_id in np.unique(annot_slice):
        if label_id == 0:
            continue
        shuf = (int(label_id) * shuffle_prime) % N
        c = cmap(shuf / N)
        m = annot_slice == label_id
        rgba[m, 0] = c[0]
        rgba[m, 1] = c[1]
        rgba[m, 2] = c[2]
        rgba[m, 3] = alpha_fg
    return rgba


def _extract_l6(vol):
    if vol.ndim == 5:
        return vol[..., 0, 5]
    if vol.ndim == 4:
        return vol[..., 5]
    return vol


def _ix(world, aff, axis, shape):
    return max(0, min(shape[axis] - 1,
        int(round((world - aff[axis, 3]) / aff[axis, axis]))))


def main():
    t0 = time.time()
    print(f'REG_DIR:        {REG_DIR}')
    print(f'Output:         {OUT}')
    os.makedirs(OUT_DIR, exist_ok=True)

    print('loading macaque 250 um skull backdrop...')
    bone_img = nib.load(SKULL_250UM)
    bone = np.asarray(bone_img.dataobj).astype(np.float32)
    bone_aff = bone_img.affine
    print(f'  shape {bone.shape}, voxel {abs(bone_aff[0,0])*1000:.1f} um')

    print('loading warped NMT T1...')
    nmt_img = nib.load(NMT_IN_MACAQUE)
    nmt = nmt_img.get_fdata().astype(np.float32)
    nmt_aff = nmt_img.affine

    print('loading warped CHARM (L6) + SARM (L6) + D99...')
    charm_img = nib.load(CHARM_IN_MACAQUE)
    charm6 = _extract_l6(charm_img.get_fdata().astype(np.int32))
    charm_aff = charm_img.affine
    sarm_img = nib.load(SARM_IN_MACAQUE)
    sarm6 = _extract_l6(sarm_img.get_fdata().astype(np.int32))
    sarm_aff = sarm_img.affine
    d99_img = nib.load(D99_IN_MACAQUE)
    d99 = d99_img.get_fdata().astype(np.int32)
    d99_aff = d99_img.affine

    n_charm = int(len(np.unique(charm6)) - 1)
    n_sarm = int(len(np.unique(sarm6)) - 1)
    n_d99 = int(len(np.unique(d99)) - 1)
    print(f'  CHARM L6: {n_charm} labels, SARM L6: {n_sarm} labels, '
          f'D99: {n_d99} labels (in macaque frame)')

    print('locating slice picker (dACC centroid) + crop bbox (cavity bbox)...')
    cav_img = nib.load(CAVITY_PROD)
    cav = np.asarray(cav_img.dataobj).astype(bool)
    cav_aff = cav_img.affine
    cav_idx = np.argwhere(cav)
    cw_all = (cav_aff[:3, :3] @ cav_idx.T).T + cav_aff[:3, 3]
    # Crop bbox = cavity bbox + 10 mm low, 10/15 mm high (matches Fig 5).
    x_min, x_max = cw_all[:, 0].min() - 10, cw_all[:, 0].max() + 10
    y_min, y_max = cw_all[:, 1].min() - 10, cw_all[:, 1].max() + 15
    z_min, z_max = cw_all[:, 2].min() - 10, cw_all[:, 2].max() + 15

    # Slice picker = dACC centroid (CHARM L6 ids {8,9,13,14}). Cutting
    # at the target region puts thick cortex + key subcortical
    # structures into every panel, vs slicing at the cavity centroid
    # which sits deep in the inferior brain where cortex is sparse and
    # the mid-sagittal cut is a thin medial-wall rim.
    DACC_IDS = {8, 9, 13, 14}
    charm6_for_slice = _extract_l6(np.asarray(nib.load(CHARM_IN_MACAQUE).dataobj))
    dacc_mask = np.isin(charm6_for_slice, list(DACC_IDS))
    if dacc_mask.any():
        dacc_idx = np.argwhere(dacc_mask)
        # use charm_aff which is the same as cav_aff (same macaque space)
        cw = (nib.load(CHARM_IN_MACAQUE).affine[:3, :3] @ dacc_idx.mean(0)
              + nib.load(CHARM_IN_MACAQUE).affine[:3, 3])
        print(f'  dACC centroid (world): '
              f'({cw[0]:+.2f}, {cw[1]:+.2f}, {cw[2]:+.2f}) mm '
              f'(slice picker)')
    else:
        cw = (cav_aff[:3, :3] @ cav_idx.mean(0)) + cav_aff[:3, 3]
        print(f'  dACC mask empty -- fallback to cavity centroid '
              f'({cw[0]:+.2f}, {cw[1]:+.2f}, {cw[2]:+.2f}) mm')
    print(f'  crop bbox (world): '
          f'x[{x_min:+.1f},{x_max:+.1f}] '
          f'y[{y_min:+.1f},{y_max:+.1f}] '
          f'z[{z_min:+.1f},{z_max:+.1f}] mm')

    # Bone display window
    valid = bone[bone > 0]
    lo, hi = np.percentile(valid, [0.5, 99.5]) if valid.size else (0, 1)

    # Extents
    e_bone = _extents_for_affine(bone, bone_aff)
    e_nmt = _extents_for_affine(nmt, nmt_aff)
    e_charm = _extents_for_affine(charm6, charm_aff)
    e_sarm = _extents_for_affine(sarm6, sarm_aff)
    e_d99 = _extents_for_affine(d99, d99_aff)

    print('rendering 3x4 layout...')
    fig, axes = plt.subplots(3, 4, figsize=(20, 13),
                              constrained_layout=True)

    # Per-row config: axis index + axes labels + per-row xlim/ylim
    # (mirrors native_cavity.py's tight crop to cavity bbox).
    rows = [
        ('sagittal x = {:+.1f} mm'.format(cw[0]), 0,
         'A-P (mm)', 'S-I (mm)', (y_min, y_max), (z_min, z_max)),
        ('coronal y = {:+.1f} mm'.format(cw[1]), 1,
         'R-L (mm)', 'S-I (mm)', (x_min, x_max), (z_min, z_max)),
        ('axial z = {:+.1f} mm'.format(cw[2]), 2,
         'R-L (mm)', 'A-P (mm)', (x_min, x_max), (y_min, y_max)),
    ]
    cols = [
        ('NMT v2 T1 (warped)', None, nmt, nmt_aff, e_nmt, 'overlay-cmap'),
        (f'CHARM L6 ({n_charm} cortical)', charm6, None, charm_aff, e_charm,
         'labels'),
        (f'SARM L6 ({n_sarm} subcortical)', sarm6, None, sarm_aff, e_sarm,
         'labels'),
        (f'D99 ({n_d99} cortical+subcortical)', d99, None, d99_aff, e_d99,
         'labels'),
    ]

    for r, (slice_title, axis, xlab, ylab, xlim, ylim) in enumerate(rows):
        # Bone slice
        bi = _ix(cw[axis], bone_aff, axis, bone.shape)
        if axis == 0:
            bone_sl = bone[bi, :, :].T
        elif axis == 1:
            bone_sl = bone[:, bi, :].T
        else:
            bone_sl = bone[:, :, bi].T

        for c, (col_title, vol, intensity, aff, ext_set, kind) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(np.clip(bone_sl, lo, hi), cmap='bone', origin='lower',
                       extent=e_bone[axis], vmin=lo, vmax=hi,
                       interpolation='nearest')
            ext = ext_set[axis]
            si = _ix(cw[axis], aff, axis,
                       intensity.shape if intensity is not None else vol.shape)
            if kind == 'overlay-cmap':
                if axis == 0:
                    sl = intensity[si, :, :].T
                elif axis == 1:
                    sl = intensity[:, si, :].T
                else:
                    sl = intensity[:, :, si].T
                t_overlay = np.where(sl > 1, sl, np.nan)
                ax.imshow(t_overlay, cmap='hot', alpha=0.45,
                            origin='lower', extent=ext,
                            interpolation='nearest')
            else:  # labels
                if axis == 0:
                    sl = vol[si, :, :].T
                elif axis == 1:
                    sl = vol[:, si, :].T
                else:
                    sl = vol[:, :, si].T
                ax.imshow(_annot_to_rgba(sl, alpha_fg=0.7), origin='lower',
                            extent=ext, interpolation='nearest')
            ax.set(xlabel=xlab, ylabel=ylab,
                   title=f'{slice_title}\n{col_title}',
                   aspect='equal')
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    fig.savefig(OUT, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB) '
          f'in {time.time()-t0:.0f} s')


if __name__ == '__main__':
    main()
