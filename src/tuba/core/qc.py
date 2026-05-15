"""Shared QC plot templates used by validation reports."""
import os
import numpy as np


def cavity_orthoslice_qc(arr, cavity, affine, voxel_mm, out_path, title=None):
    """3-panel orthoslice (sagittal / coronal / axial) at the cavity
    centroid with a cyan contour overlay on the skull intensity.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    coords = np.argwhere(cavity)
    if len(coords) > 0:
        ci, cj, ck = coords.mean(0).astype(int)
    else:
        ci, cj, ck = (s // 2 for s in arr.shape)

    ni, nj, nk = arr.shape
    aff = affine
    extents = (
        [aff[1, 3], aff[1, 3] + nj * aff[1, 1],
         aff[2, 3], aff[2, 3] + nk * aff[2, 2]],
        [aff[0, 3] + ni * aff[0, 0], aff[0, 3],
         aff[2, 3], aff[2, 3] + nk * aff[2, 2]],
        [aff[0, 3] + ni * aff[0, 0], aff[0, 3],
         aff[1, 3], aff[1, 3] + nj * aff[1, 1]],
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    axes[0].imshow(arr[ci, :, :].T, cmap='bone', origin='lower', extent=extents[0])
    axes[0].contour(cavity[ci, :, :].T, levels=[0.5], colors=['cyan'],
                    linewidths=1.2, extent=extents[0])
    axes[0].set_title(f'sagittal i={ci}')
    axes[0].set_xlabel('A-P (mm)')
    axes[0].set_ylabel('D-V (mm)')
    axes[1].imshow(arr[:, cj, :].T, cmap='bone', origin='lower', extent=extents[1])
    axes[1].contour(cavity[:, cj, :].T, levels=[0.5], colors=['cyan'],
                    linewidths=1.2, extent=extents[1])
    axes[1].set_title(f'coronal j={cj}')
    axes[1].set_xlabel('R-L (mm)')
    axes[2].imshow(arr[:, :, ck].T, cmap='bone', origin='lower', extent=extents[2])
    axes[2].contour(cavity[:, :, ck].T, levels=[0.5], colors=['cyan'],
                    linewidths=1.2, extent=extents[2])
    axes[2].set_title(f'axial k={ck}')
    axes[2].set_xlabel('R-L (mm)')
    axes[2].set_ylabel('A-P (mm)')
    if title:
        fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'wrote {out_path}')
    return out_path
