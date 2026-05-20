"""Render the NMT v2 atlas figure for the macaque manuscript:

  nmt_atlases.png   3-row x 4-col layout: NMT T1 template, CHARM L6,
                    SARM L6, D99 -- sagittal/coronal/axial through the
                    NMT brain centroid.

Parallels the Harvard-Oxford / Schaefer figure for the human manuscript
(``docs/human/make_figures.py`` :func:`fig2_mni_template_and_atlases`).

Paths read from environment variables that mirror tuba.species.macaque:
  NMT_DEST                   NMT_v2.0_sym/ root directory
                             (default: ~/.cache/tuba/macaque/atlas/nmt_v2)

Run:
    cd docs/macaque
    python make_atlas_figure.py
"""
import os
import sys
import time
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse the same env-var resolution + legacy fallback that
# tuba._macaque.macaque_skull_constants applies for the pipeline, so the
# doc-figure script can never disagree with the pipeline on NMT root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))
from tuba._macaque.macaque_skull_constants import NMT_DIR  # noqa: E402

NMT_T1  = os.path.join(NMT_DIR, 'NMT_v2.0_sym.nii.gz')
NMT_MASK = os.path.join(NMT_DIR, 'NMT_v2.0_sym_brainmask.nii.gz')
CHARM   = os.path.join(NMT_DIR, 'CHARM_in_NMT_v2.0_sym.nii.gz')
SARM    = os.path.join(NMT_DIR, 'SARM_in_NMT_v2.0_sym.nii.gz')
D99     = os.path.join(NMT_DIR, 'D99_atlas_in_NMT_v2.0_sym.nii.gz')

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
OUT = os.path.join(OUT_DIR, 'nmt_atlases.png')

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


def _percentile_clip(arr, low=0.5, high=99.5):
    valid = arr[arr > 0] if (arr > 0).any() else arr
    lo, hi = np.percentile(valid, [low, high])
    return np.clip(arr, lo, hi), lo, hi


def _annot_to_rgba(annot_slice, palette='gist_ncar', alpha_fg=0.85,
                   shuffle_prime=104729):
    """Map a 2D integer-label slice to RGBA via a deterministic ID hash so
    neighbouring labels get colour-separated. Background (id=0) is
    transparent."""
    cmap = plt.get_cmap(palette)
    N = 4096
    rgba = np.zeros((*annot_slice.shape, 4), dtype=np.float32)
    for label_id in np.unique(annot_slice):
        if label_id == 0:
            continue
        shuf = (int(label_id) * shuffle_prime) % N
        colour = cmap(shuf / N)
        mask = annot_slice == label_id
        rgba[mask, 0] = colour[0]
        rgba[mask, 1] = colour[1]
        rgba[mask, 2] = colour[2]
        rgba[mask, 3] = alpha_fg
    return rgba


def _atlas_panel(ax, vol, aff, slc_idx, axis, extent, title, xlab, ylab):
    if axis == 0:
        ann = vol[slc_idx, :, :].T
    elif axis == 1:
        ann = vol[:, slc_idx, :].T
    else:
        ann = vol[:, :, slc_idx].T
    ax.set_facecolor('black')
    ax.imshow(_annot_to_rgba(ann), origin='lower', extent=extent,
              interpolation='nearest')
    ax.set(xlabel=xlab, ylabel=ylab, title=title, aspect='equal')


def _extract_l6(vol):
    """CHARM and SARM are 5D (x, y, z, 1, 6) where the last axis indexes
    hierarchy depth (L1 coarsest .. L6 finest). Return L6 as 3D."""
    if vol.ndim == 5:
        return vol[..., 0, 5]
    if vol.ndim == 4:
        return vol[..., 5]
    return vol


def main():
    t0 = time.time()
    print(f'NMT atlas root: {NMT_DIR}')
    print(f'Output:         {OUT}')
    os.makedirs(OUT_DIR, exist_ok=True)

    print('loading NMT T1 + brain mask...')
    t1_img = nib.load(NMT_T1)
    t1 = t1_img.get_fdata().astype(np.float32)
    t1_aff = t1_img.affine
    mask = nib.load(NMT_MASK).get_fdata().astype(np.float32)

    print('loading CHARM + SARM + D99...')
    charm_img = nib.load(CHARM)
    charm6 = _extract_l6(charm_img.get_fdata().astype(np.int32))
    charm_aff = charm_img.affine
    sarm_img = nib.load(SARM)
    sarm6 = _extract_l6(sarm_img.get_fdata().astype(np.int32))
    sarm_aff = sarm_img.affine
    d99_img = nib.load(D99)
    d99 = d99_img.get_fdata().astype(np.int32)
    d99_aff = d99_img.affine

    n_charm = int(len(np.unique(charm6)) - 1)
    n_sarm = int(len(np.unique(sarm6)) - 1)
    n_d99 = int(len(np.unique(d99)) - 1)
    print(f'  CHARM L6: {n_charm} labels, SARM L6: {n_sarm} labels, '
          f'D99: {n_d99} labels')

    # Slice at brain centroid (mask CoM in world mm, then map per-volume)
    ijk = np.argwhere(mask > 0.5)
    centroid_v = ijk.mean(0)
    centroid_w = t1_aff[:3, :3] @ centroid_v + t1_aff[:3, 3]
    print(f'  NMT brain centroid (world): '
          f'({centroid_w[0]:+.1f}, {centroid_w[1]:+.1f}, '
          f'{centroid_w[2]:+.1f}) mm')

    t1_slice = tuple(int(round(v))
                     for v in _world_to_vox(t1_aff, centroid_w))
    charm_slice = tuple(int(round(v))
                        for v in _world_to_vox(charm_aff, centroid_w))
    sarm_slice = tuple(int(round(v))
                       for v in _world_to_vox(sarm_aff, centroid_w))
    d99_slice = tuple(int(round(v))
                      for v in _world_to_vox(d99_aff, centroid_w))

    t1_clip, lo, hi = _percentile_clip(t1, 0.5, 99.5)
    e_t1 = _extents_for_affine(t1, t1_aff)
    e_charm = _extents_for_affine(charm6, charm_aff)
    e_sarm = _extents_for_affine(sarm6, sarm_aff)
    e_d99 = _extents_for_affine(d99, d99_aff)

    print('rendering 3x4 layout...')
    fig, axes = plt.subplots(3, 4, figsize=(20, 13),
                              constrained_layout=True)

    rows = [
        ('sagittal x = {:+.1f} mm'.format(centroid_w[0]), 0,
         t1_slice[0], charm_slice[0], sarm_slice[0], d99_slice[0],
         e_t1[0], e_charm[0], e_sarm[0], e_d99[0],
         'A-P (mm)', 'S-I (mm)'),
        ('coronal y = {:+.1f} mm'.format(centroid_w[1]), 1,
         t1_slice[1], charm_slice[1], sarm_slice[1], d99_slice[1],
         e_t1[1], e_charm[1], e_sarm[1], e_d99[1],
         'R-L (mm)', 'S-I (mm)'),
        ('axial z = {:+.1f} mm'.format(centroid_w[2]), 2,
         t1_slice[2], charm_slice[2], sarm_slice[2], d99_slice[2],
         e_t1[2], e_charm[2], e_sarm[2], e_d99[2],
         'R-L (mm)', 'A-P (mm)'),
    ]

    for r, (slice_title, axis, t1i, ci, si, di,
            t1ext, cext, sext, dext, xlab, ylab) in enumerate(rows):
        if axis == 0:
            sl = t1_clip[t1i, :, :].T; msl = mask[t1i, :, :].T
        elif axis == 1:
            sl = t1_clip[:, t1i, :].T; msl = mask[:, t1i, :].T
        else:
            sl = t1_clip[:, :, t1i].T; msl = mask[:, :, t1i].T
        axes[r, 0].imshow(sl, cmap='gray', origin='lower', extent=t1ext,
                           vmin=lo, vmax=hi)
        axes[r, 0].contour(msl, levels=[0.5], colors=['cyan'],
                            linewidths=0.7, extent=t1ext)
        axes[r, 0].set(xlabel=xlab, ylabel=ylab,
                       title=f'{slice_title}\nNMT v2 T1 template (0.25 mm)',
                       aspect='equal')

        _atlas_panel(axes[r, 1], charm6, charm_aff, ci, axis, cext,
                      f'{slice_title}\nCHARM L6 ({n_charm} cortical labels)',
                      xlab, ylab)
        _atlas_panel(axes[r, 2], sarm6, sarm_aff, si, axis, sext,
                      f'{slice_title}\nSARM L6 ({n_sarm} subcortical labels)',
                      xlab, ylab)
        _atlas_panel(axes[r, 3], d99, d99_aff, di, axis, dext,
                      f'{slice_title}\nD99 ({n_d99} cortical+subcortical labels)',
                      xlab, ylab)

    # No suptitle: descriptive text lives in the LaTeX caption so the
    # PNG itself stays full-width and uncrowded.
    fig.savefig(OUT, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB) '
          f'in {time.time()-t0:.0f} s')


if __name__ == '__main__':
    main()
