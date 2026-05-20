"""Probe the AMNH M-87264 macaque microCT for orientation + intensity QC.

Produces (in REG_DIR):
  orientation_profile.png   per-slice bone-mass + bone bbox profile
  intensity_histogram.png   log-binned histogram with provisional anchors
  orthoslice_qc.png         three orthogonal mid-stack views to disambiguate
                              SLICE_FLIP / ROW_FLIP / COL_FLIP from the
                              bone-mass profile alone.

Mirrors the orientation determination described in
``docs/macaque/manuscript.tex``, Section 3.

Usage
-----
  python3 -m tuba._macaque.orientation_probe
"""
import os
import time
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .macaque_skull_constants import (
    REG_DIR, _tiff_paths, NATIVE_VOXEL_MM, BONE_THRESH)

OUT_PROFILE = os.path.join(REG_DIR, 'orientation_profile.png')
OUT_HIST    = os.path.join(REG_DIR, 'intensity_histogram.png')
OUT_ORTHO   = os.path.join(REG_DIR, 'orthoslice_qc.png')

N_SAMPLE = 21
HIST_STRIDE = 16            # histogram from every 16th slice (~134 slices)


def per_slice_bone_profile(paths, n_sample=N_SAMPLE, thresh=BONE_THRESH):
    """Sample n_sample evenly spaced slices; report per-slice bone count
    and bone bounding box in (row, col)."""
    idx = np.linspace(0, len(paths) - 1, n_sample, dtype=int)
    counts = np.zeros(n_sample, dtype=np.int64)
    bbox = np.full((n_sample, 4), -1)        # row_lo, row_hi, col_lo, col_hi
    centroids = np.full((n_sample, 2), np.nan)
    for i, k in enumerate(idx):
        a = tifffile.imread(paths[k])
        m = a > thresh
        counts[i] = m.sum()
        if counts[i] > 0:
            rs, cs = np.where(m)
            bbox[i] = [rs.min(), rs.max(), cs.min(), cs.max()]
            centroids[i] = [rs.mean(), cs.mean()]
    return idx, counts, bbox, centroids


def intensity_histogram(paths, stride=HIST_STRIDE, bins=400, vmax=50000):
    bin_edges = np.linspace(0, vmax, bins + 1)
    hist = np.zeros(bins, dtype=np.int64)
    for k in range(0, len(paths), stride):
        a = tifffile.imread(paths[k]).ravel()
        h, _ = np.histogram(np.clip(a, 0, vmax), bins=bin_edges)
        hist += h
    centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return centres, hist


def render_orthoslices(paths, out_path):
    """Sample mid-stack slice + middle-row + middle-col views."""
    n_slc = len(paths)
    mid_slc = n_slc // 2
    n_row, n_col = tifffile.imread(paths[0]).shape
    print(f'  rendering orthoslices at slc={mid_slc}, row={n_row//2}, '
          f'col={n_col//2}')

    ax = tifffile.imread(paths[mid_slc])

    sag = np.empty((n_slc, n_row), dtype=np.uint16)
    for k in range(n_slc):
        a = tifffile.imread(paths[k])
        sag[k] = a[:, n_col // 2]
        if k % 200 == 0:
            print(f'    sagittal slice {k}/{n_slc}')

    cor = np.empty((n_slc, n_col), dtype=np.uint16)
    for k in range(n_slc):
        a = tifffile.imread(paths[k])
        cor[k] = a[n_row // 2, :]
        if k % 200 == 0:
            print(f'     coronal slice {k}/{n_slc}')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    p_lo, p_hi = np.percentile(ax, [0.5, 99.5])
    for arr, axis_obj, title in [
        (ax,  axes[0], f'storage axial (slc={mid_slc})'),
        (cor, axes[1], f'storage coronal (row={n_row//2})'),
        (sag, axes[2], f'storage sagittal (col={n_col//2})'),
    ]:
        axis_obj.imshow(arr, cmap='bone', vmin=p_lo, vmax=p_hi,
                          origin='lower', aspect='equal')
        axis_obj.set_title(title)
    fig.suptitle('AMNH M-87264 macaque microCT - storage-axis orthoslices '
                  '(no flips applied yet)')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  wrote {out_path}')


def main():
    paths = _tiff_paths()
    n_slc = len(paths)
    print(f'Probing {n_slc} TIFFs at {NATIVE_VOXEL_MM*1000:.2f} um, '
          f'BONE_THRESH={BONE_THRESH}')

    t0 = time.time()
    print('--- intensity histogram ---')
    centres, hist = intensity_histogram(paths)
    nz = hist > 0
    cum = hist.cumsum() / hist.sum()
    p50, p90, p99 = (np.searchsorted(cum, p) for p in [0.5, 0.9, 0.99])
    print(f'  P50={centres[p50]:.0f}, P90={centres[p90]:.0f}, '
          f'P99={centres[p99]:.0f}, t={(time.time()-t0):.0f} s')

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(centres[nz], hist[nz])
    ax.axvline(BONE_THRESH, color='C1', linestyle='--',
               label=f'BONE_THRESH={BONE_THRESH}')
    ax.axvline(centres[p99], color='C2', linestyle=':',
               label=f'P99 = {centres[p99]:.0f}')
    ax.set_xlabel('uint16 intensity'); ax.set_ylabel('voxel count (log)')
    ax.set_title(f'AMNH M-87264 macaque skull intensity histogram '
                  f'(every {HIST_STRIDE}th slice, {len(paths)//HIST_STRIDE} '
                  f'slices)')
    ax.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(OUT_HIST, dpi=120); plt.close(fig)
    print(f'  wrote {OUT_HIST}')

    print('--- per-slice bone-mass profile ---')
    idx, counts, bbox, centroids = per_slice_bone_profile(paths)
    print(f'  t={(time.time()-t0):.0f} s')
    for i, k in enumerate(idx):
        print(f'  slc={k:5d}: bone={counts[i]:7d}  '
              f'bbox r=[{bbox[i,0]:4d},{bbox[i,1]:4d}] '
              f'c=[{bbox[i,2]:4d},{bbox[i,3]:4d}]  '
              f'centroid r={centroids[i,0]:.0f}, c={centroids[i,1]:.0f}')

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True,
                              sharex=True)
    axes[0].plot(idx, counts, 'o-')
    axes[0].set_ylabel('bone voxels per slice')
    axes[0].set_title(f'AMNH M-87264 per-slice bone mass '
                       f'(thresh > {BONE_THRESH})')
    axes[1].plot(idx, bbox[:, 0], 'o-', label='row_lo')
    axes[1].plot(idx, bbox[:, 1], 'o-', label='row_hi')
    axes[1].plot(idx, bbox[:, 2], 's-', label='col_lo')
    axes[1].plot(idx, bbox[:, 3], 's-', label='col_hi')
    axes[1].set_xlabel('storage slice index')
    axes[1].set_ylabel('bone bbox (rows / cols)')
    axes[1].legend(loc='upper right')
    fig.savefig(OUT_PROFILE, dpi=120); plt.close(fig)
    print(f'  wrote {OUT_PROFILE}')

    print('--- mid-stack orthoslice render ---')
    render_orthoslices(paths, OUT_ORTHO)
    print(f'\nDone in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
