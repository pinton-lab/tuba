"""Render the 250 um RAS cache with anatomical labels to verify the
orientation flags in macaque_skull_constants.py.

Expected, post-flip:
  Sagittal slice through midline (i=mid)  : skull profile with snout
    at +y (right-of-figure), occiput at -y (left-of-figure), vault at
    +z (top), basicranium at -z (bottom). The brain cavity (a void)
    appears in the dorsal half.
  Coronal slice (j=through brain box)    : basicranium at -z, vault at
    +z, midline at x=0, brain cavity void in the dorsal half.
  Axial slice (k=just dorsal of basicranium): symmetric L-R bone outline
    with the foramen magnum (caudal) and the orbits (rostral).
"""
import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .macaque_skull_constants import REG_DIR, MACAQUE_RAS_NIFTI

OUT = os.path.join(REG_DIR, 'cache_qc_250um.png')


def main():
    print(f'Loading {MACAQUE_RAS_NIFTI}...')
    img = nib.load(MACAQUE_RAS_NIFTI)
    arr = np.asarray(img.dataobj).astype(np.float32)
    aff = img.affine
    voxel_mm = float(abs(aff[0, 0]))
    print(f'  shape={arr.shape}, voxel={voxel_mm*1000:.1f} um, '
          f'intensity range [{arr.min():.0f}, {arr.max():.0f}]')
    print(f'  affine =\n{aff}')

    n_i, n_j, n_k = arr.shape

    # World coords of voxel (0,0,0) and (n-1, n-1, n-1)
    p0 = aff[:3, :3] @ np.array([0, 0, 0]) + aff[:3, 3]
    pN = aff[:3, :3] @ np.array([n_i-1, n_j-1, n_k-1]) + aff[:3, 3]
    print(f'  world (0,0,0) -> {p0}')
    print(f'  world (Nx-1, Ny-1, Nz-1) -> {pN}')

    # Pick mid-volume orthoslices.
    ci, cj, ck = n_i // 2, n_j // 2, n_k // 2
    print(f'  slicing at midpoints i={ci}, j={cj}, k={ck}')

    p_lo, p_hi = np.percentile(arr, [0.5, 99.5])

    extents = (
        # sagittal (i fixed): horizontal = y (AP), vertical = z (DV)
        [aff[1, 3], aff[1, 3] + n_j * aff[1, 1],
         aff[2, 3], aff[2, 3] + n_k * aff[2, 2]],
        # coronal  (j fixed): horizontal = x (RL), vertical = z (DV)
        [aff[0, 3] + n_i * aff[0, 0], aff[0, 3],
         aff[2, 3], aff[2, 3] + n_k * aff[2, 2]],
        # axial    (k fixed): horizontal = x (RL), vertical = y (AP)
        [aff[0, 3] + n_i * aff[0, 0], aff[0, 3],
         aff[1, 3], aff[1, 3] + n_j * aff[1, 1]],
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)

    axes[0].imshow(arr[ci, :, :].T, cmap='bone', vmin=p_lo, vmax=p_hi,
                    origin='lower', extent=extents[0], aspect='equal')
    axes[0].set_title(f'SAGITTAL  i={ci} (x={ci*aff[0,0]+aff[0,3]:.1f} mm)\n'
                       f'horizontal=y (AP, +y=anterior=right of figure)\n'
                       f'vertical=z (+z=dorsal=top of figure)')
    axes[0].set_xlabel('y world-mm (anterior +)')
    axes[0].set_ylabel('z world-mm (dorsal +)')

    axes[1].imshow(arr[:, cj, :].T, cmap='bone', vmin=p_lo, vmax=p_hi,
                    origin='lower', extent=extents[1], aspect='equal')
    axes[1].set_title(f'CORONAL   j={cj} (y={cj*aff[1,1]+aff[1,3]:.1f} mm)\n'
                       f'horizontal=x (RL, +x=right=right of figure)\n'
                       f'vertical=z (+z=dorsal=top of figure)')
    axes[1].set_xlabel('x world-mm (right +)')
    axes[1].set_ylabel('z world-mm (dorsal +)')

    axes[2].imshow(arr[:, :, ck].T, cmap='bone', vmin=p_lo, vmax=p_hi,
                    origin='lower', extent=extents[2], aspect='equal')
    axes[2].set_title(f'AXIAL     k={ck} (z={ck*aff[2,2]+aff[2,3]:.1f} mm)\n'
                       f'horizontal=x (RL, +x=right=right of figure)\n'
                       f'vertical=y (+y=anterior=top of figure)')
    axes[2].set_xlabel('x world-mm (right +)')
    axes[2].set_ylabel('y world-mm (anterior +)')

    fig.suptitle(f'AMNH M-87264 macaque 250 um RAS cache QC '
                  f'(SLICE_FLIP, ROW_FLIP, COL_FLIP, SWAP applied; '
                  f'NO PCA rotation yet)')
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
