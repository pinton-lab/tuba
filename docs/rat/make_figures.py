#!/usr/bin/env python3
"""Regenerate the rat pillar manuscript figures.

Inputs are read from $TUBA_RAT_REG_DIR (default ~/.cache/tuba/rat/registration)
and the WHS atlas cache (default ~/.cache/tuba/rat/atlas/whs_sd_v4). Run after:
  python -m tuba.data.fetch_rat
  python -m tuba.data.fetch_whs
  # plus a calibrated stage (orientation, 118 um cache, cavity, WHS warp)

Figures produced into docs/rat/figures/:
  - native_orthoslices.{png,pdf}      raw storage-frame orthoslices
  - world_orthoslices_240um.{png,pdf} world-RAS orthoslices on 240 um cache
  - intensity_histogram.{png,pdf}     aggregated histogram with thresholds
  - bone_mass_profile.{png,pdf}       per-slice bone-mass profile
  - whs_atlas.{png,pdf}               WHS T2* + v4.01 annotation overview
  - cavity_qc.{png,pdf}               9-panel cavity QC (already produced by
                                       pipeline; re-emitted here if missing)
  - registration_qc.{png,pdf}         WHS template + annotation in rat space
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'src')))
from tuba.atlases.whs import WaxholmSpace
from tuba.species import rat as r


def _save(fig, name):
    fig.savefig(os.path.join(FIG, f'{name}.png'), dpi=120, bbox_inches='tight')
    fig.savefig(os.path.join(FIG, f'{name}.pdf'), bbox_inches='tight')
    print(f'  wrote {name}.{{png,pdf}}')


def fig_intensity_histogram_and_profile():
    """Aggregated histogram + bone-mass profile from the cached probe.npz.
    Re-runs the probe if probe.npz is missing."""
    probe_path = os.path.join(r.REG_DIR, 'probe.npz')
    if not os.path.exists(probe_path):
        print('  generating probe.npz from raw TIFFs...')
        paths = sorted(glob.glob(r.TIFF_GLOB))
        agg = np.zeros(256, dtype=np.int64)
        for i in range(0, len(paths), 16):
            a = tifffile.imread(paths[i])
            h, _ = np.histogram(a.ravel(), bins=256, range=(0, 256))
            agg += h
        prof_path = os.path.join(r.REG_DIR, 'bone_profile_full.npz')
        if os.path.exists(prof_path):
            prof = np.load(prof_path)['profile']
        else:
            prof = np.array([(tifffile.imread(p) > r.BONE_THRESH).sum() for p in paths])
        np.savez(probe_path, hist=agg, bone_counts=prof)
    npz = np.load(probe_path)
    agg = npz['hist']

    # Histogram with thresholds
    fig, ax = plt.subplots(figsize=(8, 4))
    centers = np.arange(256)
    ax.semilogy(centers, agg, lw=1.2, color='k')
    for thresh, name, col in ((r.BONE_LOW, 'BONE_LOW', '#1f77b4'),
                              (r.ALIGN_PCA_THRESH, 'ALIGN_PCA_THRESH', '#ff7f0e'),
                              (r.BONE_HIGH, 'BONE_HIGH', '#d62728')):
        ax.axvline(thresh, color=col, linestyle='--', lw=1.2, label=f'{name} = {thresh}')
    ax.set_xlim(0, 255)
    ax.set_ylim(1e2, agg.max()*1.5)
    ax.set_xlabel('uint8 intensity')
    ax.set_ylabel('voxel count (log)')
    ax.set_title(f'TMM M-2272 aggregated intensity histogram '
                  f'(every 16th slice, total {agg.sum()/1e6:.1f}M voxels)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    _save(fig, 'intensity_histogram')
    plt.close(fig)

    # Bone-mass profile, with annotations for first/last bone-bearing slice
    prof_path = os.path.join(r.REG_DIR, 'bone_profile_full.npz')
    if os.path.exists(prof_path):
        prof_npz = np.load(prof_path)
        prof = prof_npz['profile']
        first, last = int(prof_npz['first']), int(prof_npz['last'])
    else:
        prof = npz['bone_counts']
        first, last = 0, len(prof) - 1
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(np.arange(len(prof)), prof, color='#aaaaaa', alpha=0.7)
    ax.plot(prof, lw=0.6, color='k')
    ax.axvline(first, color='#2ca02c', linestyle=':', label=f'first bone slice ({first})')
    ax.axvline(last, color='#2ca02c', linestyle=':', label=f'last bone slice ({last})')
    ax.set_xlabel('storage slice index (rostral $\\rightarrow$ caudal)')
    ax.set_ylabel(f'bone-voxel count (intensity > {r.BONE_THRESH})')
    ax.set_title(f'TMM M-2272 bone-mass profile -- cranium spans '
                  f'{(last-first+1)*r.NATIVE_VOXEL_MM:.1f} mm '
                  f'({last-first+1} slices at {r.NATIVE_VOXEL_MM*1000:.2f} um)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    _save(fig, 'bone_mass_profile')
    plt.close(fig)


def fig_whs_atlas():
    """WHS T2* template + v4.01 annotation, three orthoslices through the
    brain-mask centroid."""
    atlas = WaxholmSpace(atlas_dir=os.path.expanduser('~/.cache/tuba/rat/atlas/whs_sd_v4'))
    t2 = nib.load(atlas.template_path)
    annot = nib.load(atlas.annotation_path)
    t2_arr = t2.get_fdata()
    an_arr = annot.get_fdata().astype(np.int32)

    # Centroid of brain mask = nonzero label
    fg = an_arr > 0
    if not fg.any():
        print('  WHS annotation appears empty; skipping fig_whs_atlas')
        return
    ci, cj, ck = np.argwhere(fg).mean(0).astype(int)

    # Color palette via deterministic hash
    rng = np.random.default_rng(42)
    n_labels = an_arr.max() + 1
    palette = rng.uniform(0.2, 1.0, size=(n_labels, 3))
    palette[0] = (0, 0, 0)

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    for col, (cut, mid, xlab, ylab, name) in enumerate([
        ('sag', ci, 'AP', 'DV', f'sagittal i={ci}'),
        ('cor', cj, 'LR', 'DV', f'coronal j={cj}'),
        ('ax',  ck, 'LR', 'AP', f'axial k={ck}')]):
        if cut == 'sag':
            sl_t = t2_arr[mid]; sl_a = an_arr[mid]
        elif cut == 'cor':
            sl_t = t2_arr[:, mid, :]; sl_a = an_arr[:, mid, :]
        else:
            sl_t = t2_arr[:, :, mid]; sl_a = an_arr[:, :, mid]
        axs[0, col].imshow(sl_t.T, cmap='gray', origin='lower')
        axs[0, col].set_title(f'T2* {name}', fontsize=10)
        axs[0, col].set_xlabel(xlab); axs[0, col].set_ylabel(ylab)
        rgb = palette[sl_a]   # (H, W, 3)
        axs[1, col].imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
        axs[1, col].set_title(f'v4.01 labels {name}', fontsize=10)
        axs[1, col].set_xlabel(xlab); axs[1, col].set_ylabel(ylab)
    fig.suptitle('Waxholm Space SD rat brain atlas v4.01 (Kleven et al. 2023) -- '
                  f'39 um T2* template + 222 structures', fontsize=11)
    fig.tight_layout()
    _save(fig, 'whs_atlas')
    plt.close(fig)


def fig_registration_qc():
    """Rat 118 um bone (greyscale) + warped WHS T2* template (warm) + warped
    v4.01 annotation (categorical). Three orthoslices through the cavity centroid."""
    skull = nib.load(r.RAT_RAS_ALIGNED_60UM)
    skull_arr = skull.get_fdata()
    aff = skull.affine

    cav_path = os.path.join(r.REG_DIR, 'rat_cranial_cavity.nii.gz')
    cav = nib.load(cav_path).get_fdata().astype(bool)
    ci, cj, ck = np.argwhere(cav).mean(0).astype(int)

    t2 = nib.load(os.path.join(r.REG_DIR, 'whs_template_in_rat.nii.gz')).get_fdata()
    annot = nib.load(os.path.join(r.REG_DIR, 'whs_annotation_in_rat.nii.gz')).get_fdata().astype(np.int32)

    rng = np.random.default_rng(42)
    palette = rng.uniform(0.2, 1.0, size=(annot.max()+1, 3))
    palette[0] = (0, 0, 0)

    extYZ = [aff[1,3], aff[1,3]+aff[1,1]*skull_arr.shape[1], aff[2,3], aff[2,3]+aff[2,2]*skull_arr.shape[2]]
    extXZ = [aff[0,3], aff[0,3]+aff[0,0]*skull_arr.shape[0], aff[2,3], aff[2,3]+aff[2,2]*skull_arr.shape[2]]
    extXY = [aff[0,3], aff[0,3]+aff[0,0]*skull_arr.shape[0], aff[1,3], aff[1,3]+aff[1,1]*skull_arr.shape[1]]

    fig, axs = plt.subplots(3, 3, figsize=(13, 11))
    for row, (cut, midx, ext, xlab, ylab) in enumerate([
        ('sag', ci, extYZ, 'AP (mm)', 'DV (mm)'),
        ('cor', cj, extXZ, 'LR (mm)', 'DV (mm)'),
        ('ax',  ck, extXY, 'LR (mm)', 'AP (mm)')]):
        for col, what in enumerate(('skull', 't2', 'annot')):
            ax = axs[row, col]
            if cut == 'sag':
                sk, t, an, cm = skull_arr[midx], t2[midx], annot[midx], cav[midx]
            elif cut == 'cor':
                sk, t, an, cm = skull_arr[:, midx, :], t2[:, midx, :], annot[:, midx, :], cav[:, midx, :]
            else:
                sk, t, an, cm = skull_arr[:, :, midx], t2[:, :, midx], annot[:, :, midx], cav[:, :, midx]
            ax.imshow(sk.T, cmap='gray', origin='lower', extent=ext, vmin=0, vmax=200)
            if what == 't2':
                masked = np.ma.masked_where(t.T <= 0, t.T)
                ax.imshow(masked, cmap='hot', origin='lower', extent=ext, alpha=0.6)
                ax.set_title(f'{cut.upper()}  +  warped T2*', fontsize=10)
            elif what == 'annot':
                rgb = palette[an]   # (H, W, 3)
                rgb_im = np.transpose(rgb, (1, 0, 2))
                alpha = (an.T > 0).astype(float) * 0.65
                ax.imshow(np.concatenate([rgb_im, alpha[..., None]], axis=-1),
                          origin='lower', extent=ext)
                ax.set_title(f'{cut.upper()}  +  warped v4.01 labels', fontsize=10)
            else:
                ax.set_title(f'{cut.upper()}  skull only', fontsize=10)
            ax.contour(cm.T, levels=[0.5], colors='cyan', linewidths=0.9, extent=ext)
            ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    fig.suptitle(f'WHS v4.01 atlas warped into rat skull space via Affine (cavity in cyan, {cav.sum()*abs(aff[0,0])**3/1000:.2f} mL)', fontsize=11)
    fig.tight_layout()
    _save(fig, 'registration_qc')
    plt.close(fig)


def main():
    print('Generating rat manuscript figures into', FIG)
    print('--- intensity histogram + bone-mass profile ---')
    fig_intensity_histogram_and_profile()
    print('--- WHS atlas overview ---')
    fig_whs_atlas()
    print('--- registration QC ---')
    fig_registration_qc()
    print('done.')


if __name__ == '__main__':
    main()
