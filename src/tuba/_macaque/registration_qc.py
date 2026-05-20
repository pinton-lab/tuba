"""Render the macaque cavity -> NMT v2 SyN registration QC.

Three orthoslices through the cavity centroid, in two columns:
  Left: macaque skull (greyscale) + warped NMT template overlaid (warm)
  Right: macaque skull + warped CHARM L6 annotation (saturated colours)

Plus a sanity check: locate the dACC area 24 (CHARM L6 IDs {8,9,13,14})
in macaque world coords and report its centroid + voxel count.
"""
import os
from .macaque_skull_constants import REG_DIR
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


SKULL_NII = os.path.join(REG_DIR, 'macaque_skull_aligned_native_250um.nii.gz')
CAVITY_NII = os.path.join(REG_DIR, 'macaque_cranial_cavity.nii.gz')
WARPED_TEMPLATE = os.path.join(REG_DIR, 'nmt_template_in_macaque.nii.gz')
WARPED_BRAINMASK = os.path.join(REG_DIR, 'nmt_brainmask_in_macaque.nii.gz')
WARPED_CHARM = os.path.join(REG_DIR, 'charm_in_macaque.nii.gz')

OUT = os.path.join(REG_DIR, 'registration_qc.png')

# CHARM L6 area-24 (a/b/a'/b') ids
DACC_IDS = {8, 9, 13, 14}


def main():
    # Load skull (background)
    skull_img = nib.load(SKULL_NII)
    skull = np.asarray(skull_img.dataobj).astype(np.float32)
    aff = skull_img.affine
    voxel_mm = float(abs(aff[0, 0]))
    ni, nj, nk = skull.shape

    # Load cavity (contour)
    cav_img = nib.load(CAVITY_NII)
    cavity = np.asarray(cav_img.dataobj).astype(np.uint8)

    # Load warped NMT template
    tmpl_img = nib.load(WARPED_TEMPLATE)
    template = np.asarray(tmpl_img.dataobj).astype(np.float32)

    # Load warped CHARM L6
    charm_img = nib.load(WARPED_CHARM)
    charm5 = np.asarray(charm_img.dataobj)
    if charm5.ndim == 5:
        charm6 = charm5[..., 0, 5]
    elif charm5.ndim == 4:
        charm6 = charm5[..., 5]
    else:
        charm6 = charm5

    # dACC sanity check
    dacc_mask = np.isin(charm6, list(DACC_IDS))
    n_dacc = int(dacc_mask.sum())
    print(f'dACC (CHARM L6 ids {sorted(DACC_IDS)}): {n_dacc} voxels')
    if n_dacc > 0:
        ijk = np.argwhere(dacc_mask)
        ras = (aff[:3, :3] @ ijk.T).T + aff[:3, 3]
        ci, cj, ck = ijk.mean(0).astype(int)
        cx, cy, cz = ras.mean(axis=0)
        print(f'  centroid voxel ({ci}, {cj}, {ck})')
        print(f'  centroid world ({cx:+.1f}, {cy:+.1f}, {cz:+.1f}) mm')
    else:
        ci, cj, ck = (s // 2 for s in skull.shape)

    # Slicing through the dACC centroid (or cavity centroid as fallback)
    if n_dacc == 0:
        coords = np.argwhere(cavity > 0.5)
        ci, cj, ck = coords.mean(0).astype(int)
        print(f'fallback to cavity centroid ({ci}, {cj}, {ck})')

    extents = (
        [aff[1, 3], aff[1, 3] + nj * aff[1, 1],
         aff[2, 3], aff[2, 3] + nk * aff[2, 2]],
        [aff[0, 3] + ni * aff[0, 0], aff[0, 3],
         aff[2, 3], aff[2, 3] + nk * aff[2, 2]],
        [aff[0, 3] + ni * aff[0, 0], aff[0, 3],
         aff[1, 3], aff[1, 3] + nj * aff[1, 1]],
    )

    # Build a per-region colormap for CHARM L6
    np.random.seed(0)
    max_id = int(charm6.max())
    cmap_arr = np.random.rand(max_id + 1, 3)
    cmap_arr[0] = [0, 0, 0]    # background black
    region_cmap = ListedColormap(cmap_arr)

    p_lo, p_hi = np.percentile(skull, [0.5, 99.5])
    fig, axes = plt.subplots(3, 2, figsize=(14, 18), constrained_layout=True)

    for row, (s_slice, t_slice, c_slice, contour, title, ext) in enumerate([
        (skull[ci, :, :].T, template[ci, :, :].T, charm6[ci, :, :].T,
         cavity[ci, :, :].T, f'sagittal i={ci}', extents[0]),
        (skull[:, cj, :].T, template[:, cj, :].T, charm6[:, cj, :].T,
         cavity[:, cj, :].T, f'coronal j={cj}', extents[1]),
        (skull[:, :, ck].T, template[:, :, ck].T, charm6[:, :, ck].T,
         cavity[:, :, ck].T, f'axial k={ck}', extents[2]),
    ]):
        # Left col: skull + warped template overlay
        axes[row, 0].imshow(s_slice, cmap='bone', vmin=p_lo, vmax=p_hi,
                              origin='lower', extent=ext, aspect='equal')
        # Mask template where it's zero (outside brain)
        t_overlay = np.where(t_slice > 1, t_slice, np.nan)
        axes[row, 0].imshow(t_overlay, cmap='hot', alpha=0.45,
                              origin='lower', extent=ext, aspect='equal')
        axes[row, 0].contour(contour, levels=[0.5], colors=['cyan'],
                               linewidths=0.8, extent=ext)
        axes[row, 0].set_title(f'{title} -- skull + warped NMT template')

        # Right col: skull + CHARM L6 colour overlay
        axes[row, 1].imshow(s_slice, cmap='bone', vmin=p_lo, vmax=p_hi,
                              origin='lower', extent=ext, aspect='equal')
        c_overlay = np.where(c_slice > 0, c_slice, np.nan)
        axes[row, 1].imshow(c_overlay, cmap=region_cmap, alpha=0.6,
                              vmin=0, vmax=max_id, origin='lower',
                              extent=ext, aspect='equal',
                              interpolation='nearest')
        # Highlight dACC ids in green outline
        dacc_slice = np.isin(c_slice, list(DACC_IDS)).astype(np.float32)
        axes[row, 1].contour(dacc_slice, levels=[0.5], colors=['lime'],
                               linewidths=1.5, extent=ext)
        axes[row, 1].contour(contour, levels=[0.5], colors=['cyan'],
                               linewidths=0.8, extent=ext)
        axes[row, 1].set_title(f'{title} -- CHARM L6 (lime: dACC area 24)')

    fig.suptitle(f'Macaque cavity -> NMT v2 SyN registration QC. '
                  f'dACC area 24 voxels: {n_dacc}; centroid world '
                  f'({cx:+.1f}, {cy:+.1f}, {cz:+.1f}) mm')
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
