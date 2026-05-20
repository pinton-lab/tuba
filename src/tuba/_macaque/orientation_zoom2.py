"""Targeted orthoslice re-render: pick coordinates that actually pass
through the brain cavity (not just the volume midpoint).

From the bone-mass + bbox probe, the cranium's brain box is centred
around slice ~350 (mid-cranium) with col centroid ~900-1000 and row
centroid ~525. So slc=350 is a coronal-equivalent cut through the brain
box; col=950 is a sagittal through the cavity; row=400 is an axial
horizontal slice that should intersect the cavity.

Produces ``orthoslice_qc_brainbox.png`` in REG_DIR. Referenced from
``docs/macaque/manuscript.tex`` Section 3 (orientation determination
D-V disambiguation).
"""
import os
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .macaque_skull_constants import REG_DIR, _tiff_paths

OUT = os.path.join(REG_DIR, 'orthoslice_qc_brainbox.png')

SLC = 350
COL = 950
ROW = 400


def main():
    paths = _tiff_paths()
    n_slc = len(paths)
    n_row, n_col = tifffile.imread(paths[0]).shape
    print(f'N TIFFs={n_slc}, in-plane {n_row} x {n_col}')

    print(f'Loading axial slice slc={SLC} (one TIFF)...')
    ax = tifffile.imread(paths[SLC])

    print(f'Building coronal at row={ROW} over slices [0, 1100]...')
    cor = np.empty((1100, n_col), dtype=np.uint16)
    for k in range(1100):
        cor[k] = tifffile.imread(paths[k])[ROW, :]
    print(f'Building sagittal at col={COL} over slices [0, 1100]...')
    sag = np.empty((1100, n_row), dtype=np.uint16)
    for k in range(1100):
        sag[k] = tifffile.imread(paths[k])[:, COL]

    p_lo, p_hi = np.percentile(ax, [0.5, 99.5])
    fig, axes = plt.subplots(1, 3, figsize=(28, 12), constrained_layout=True)

    axes[0].imshow(ax, cmap='bone', vmin=p_lo, vmax=p_hi,
                    origin='lower', aspect='equal')
    axes[0].set_title(f'AXIAL (one TIFF) slc={SLC}\n'
                       f'x = col idx (0..{n_col-1})  y = row idx (0..{n_row-1})')
    axes[0].set_xlabel('col index')
    axes[0].set_ylabel('row index')
    axes[0].axvline(COL, color='cyan', linestyle='--',
                     label=f'col={COL} (sagittal cut)')
    axes[0].axhline(ROW, color='magenta', linestyle='--',
                     label=f'row={ROW} (coronal cut)')
    axes[0].legend(loc='upper right')

    axes[1].imshow(cor, cmap='bone', vmin=p_lo, vmax=p_hi,
                    origin='lower', aspect='equal')
    axes[1].set_title(f'CORONAL  row={ROW}\n'
                       f'x = col idx, y = storage slice idx (0..1099)')
    axes[1].set_xlabel('col index')
    axes[1].set_ylabel('storage slice index')
    axes[1].axhline(SLC, color='cyan', linestyle='--', label=f'slice={SLC}')
    axes[1].axvline(COL, color='magenta', linestyle='--', label=f'col={COL}')
    axes[1].legend(loc='upper right')

    axes[2].imshow(sag, cmap='bone', vmin=p_lo, vmax=p_hi,
                    origin='lower', aspect='equal')
    axes[2].set_title(f'SAGITTAL  col={COL}\n'
                       f'x = row idx, y = storage slice idx (0..1099)')
    axes[2].set_xlabel('row index')
    axes[2].set_ylabel('storage slice index')
    axes[2].axhline(SLC, color='cyan', linestyle='--')
    axes[2].axvline(ROW, color='magenta', linestyle='--')
    axes[2].legend(loc='upper right')

    fig.suptitle(f'AMNH M-87264 macaque brain-box orthoslices  '
                  f'(slc={SLC}, col={COL}, row={ROW})')
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
