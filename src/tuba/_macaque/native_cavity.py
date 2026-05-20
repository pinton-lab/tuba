"""Native (60.6 um) production cavity, derived from the 250 um
atlas-affine cavity by:

  1. Resample 250 um cavity onto the refined native CT grid
     (nearest-neighbor; same affine = same world coordinates).
  2. 2 mm closing to clean stair-step interpolation jaggedness.
  3. Bone clip with native bone mask (refined_native > BONE_LOW)
     so the cavity respects native-resolution cortical boundaries.

Input:  macaque_skull_aligned_native.nii.gz (refined orientation,
        from refine_native.py)
        macaque_cavity_250um.nii.gz (atlas-affine cavity, iter21)
Output: macaque_cavity_native.nii.gz (~5 GB binary at 60.6 um)
        Plus a QC PNG with sagittal/coronal/axial slices.

Runtime: ~5-10 min (memory-dominated).
Memory: ~35-45 GB peak.
"""
import os
import sys
import time
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .macaque_skull_constants import REG_DIR, MACAQUE_RAS_NATIVE_ALIGNED

CAVITY_250UM = os.path.join(REG_DIR, 'macaque_cavity_250um.nii.gz')
OUT_CAVITY_NATIVE = os.path.join(REG_DIR, 'macaque_cavity_native.nii.gz')
OUT_QC = os.path.join(REG_DIR, 'native_cavity_qc.png')

BONE_LOW = 8000
SMOOTH_MM = 2.0


def main():
    """Memory-efficient native cavity build.

    The previous ANTs-based path loaded the 82.8 GB refined-native CT
    + a 165 GB float32 resample intermediate + a second native copy
    for the QC, peaking near 500 GB and OOM-killing the user.slice
    cgroup (terminal included). This version:
      * reads the native header only (no data load) -- we only need
        its affine + shape, since the QC backdrop is the 250 um cache,
        not the native CT;
      * resamples the 250 um cavity into the native grid with
        scipy.ndimage.affine_transform (NN, order=0) keeping uint8
        throughout -- peak working set ~50 GB.
    """
    t0 = time.time()
    print('=== native cavity (60.6 um) [memory-efficient] ===\n')

    print('Step 1: read native refined CT header (no data load)...')
    native_img = nib.load(MACAQUE_RAS_NATIVE_ALIGNED)
    native_aff = native_img.affine
    native_shape = tuple(int(s) for s in native_img.shape)
    voxel_mm = float(abs(native_aff[0, 0]))
    print(f'  native shape {native_shape}, voxel {voxel_mm*1000:.1f} um')
    print(f'  [{time.time()-t0:.1f} s]')

    print('\nStep 2: load 250 um atlas-affine cavity (uint8)...')
    cav_img = nib.load(CAVITY_250UM)
    cav_arr = np.asarray(cav_img.dataobj, dtype=np.uint8)
    cav_aff = cav_img.affine
    print(f'  cavity shape {cav_arr.shape}, '
          f'voxel {abs(cav_aff[0,0])*1000:.1f} um, '
          f'size {cav_arr.nbytes/1e6:.0f} MB')
    print(f'  [{time.time()-t0:.1f} s]')

    # DIRECT-TRANSLATION POLICY: lossless upsample of the 250 um cavity
    # validated in Figure 4. No bone clip, no closing at native -- both
    # were applied at 250 um and validated there.
    #
    # scipy.ndimage.affine_transform maps output[i,j,k] <- input(M @ [i,j,k] + offset).
    # We want output (native voxel) at world W = native_aff @ [i,j,k,1];
    # the corresponding input (cavity voxel) is inv(cav_aff) @ W
    # = (inv(cav_aff) @ native_aff) @ [i,j,k,1].
    print('\nStep 3: NN-resample 250 um cavity to native grid '
          '(scipy.ndimage.affine_transform, order=0)...')
    M = np.linalg.inv(cav_aff) @ native_aff
    cav_native = affine_transform(
        cav_arr,
        matrix=M[:3, :3],
        offset=M[:3, 3],
        output_shape=native_shape,
        order=0,
        mode='constant',
        cval=0,
        prefilter=False,
    )
    cav_native = cav_native.astype(np.uint8, copy=False)
    vol_ml = float(cav_native.sum()) * voxel_mm**3 / 1000
    print(f'  native cavity volume: {vol_ml:.1f} mL '
          f'(should ~ match 250 um source within sub-mL)')
    print(f'  cav_native size: {cav_native.nbytes/1e9:.1f} GB uint8')
    print(f'  [{time.time()-t0:.1f} s]')
    del cav_arr

    print('\nStep 4: write native cavity NIfTI...')
    nib.save(nib.Nifti1Image(cav_native, native_aff), OUT_CAVITY_NATIVE)
    print(f'  wrote {OUT_CAVITY_NATIVE}')
    print(f'  [{time.time()-t0:.1f} s]')

    print('\nStep 5: render QC (250 um bone backdrop, native cavity overlay)...')
    _render_qc(cav_native, native_aff, vol_ml)
    print(f'\nDone in {time.time()-t0:.0f} s.')


def _render_qc(cavity, aff, vol_ml):
    """QC render for the native cavity.

    The bone backdrop uses the 250 um block-max cache (which has full
    3-axis 4x downsampling -- bone wall appears continuous in every
    direction, matching Figure 4 visually). The cavity overlay is at
    native resolution. The native CT is intentionally NOT loaded for
    this QC (would cost ~80 GB RAM for an unused array)."""
    voxel_um = abs(aff[0, 0]) * 1000
    ni, nj, nk = cavity.shape

    # Load 250 um bone backdrop = exactly the cache that produced the
    # 250 um cavity (Figure 4). Using a different 250 um cache (e.g.,
    # a re-downsample) would misalign the bone display vs the cavity
    # overlay because the world coords differ.
    BONE_250UM_PATH = os.path.join(
        os.path.dirname(__file__), 'macaque_skull_250um_iter.nii.gz')
    if os.path.exists(BONE_250UM_PATH):
        bone_img_250 = nib.load(BONE_250UM_PATH)
        bone_arr_250 = np.asarray(bone_img_250.dataobj)
        bone_aff_250 = bone_img_250.affine
        print(f'  using 250 um bone backdrop (shape {bone_arr_250.shape}) '
              f'for QC display')
    else:
        bone_arr_250 = arr
        bone_aff_250 = aff
        print('  WARNING: 250 um cache not found, using native bone')
    # Use CAVITY bbox (always cranial by construction) for the display
    # range, not bone bbox. The native CT can still contain the
    # mandible (isolate_cranium's 1 mm closing may fuse mandible to
    # cranium into one CC at lo-res), and that would compress the
    # cranium into the bottom half of the display.
    cav_idx_for_bbox = np.argwhere(cavity)
    if len(cav_idx_for_bbox) > 0:
        cw_bbox = (aff[:3, :3] @ cav_idx_for_bbox.T).T + aff[:3, 3]
        x_min, x_max = cw_bbox[:, 0].min() - 10, cw_bbox[:, 0].max() + 10
        y_min, y_max = cw_bbox[:, 1].min() - 10, cw_bbox[:, 1].max() + 15
        z_min, z_max = cw_bbox[:, 2].min() - 10, cw_bbox[:, 2].max() + 15
    else:
        x_min, x_max = -50, 50; y_min, y_max = -70, 70; z_min, z_max = -45, 50

    cav_idx = np.argwhere(cavity)
    cw = ((aff[:3, :3] @ cav_idx.mean(0)) + aff[:3, 3]
          if len(cav_idx) > 0 else np.array([0.0, -35.0, +10.0]))
    sag_x = [-15, float(cw[0]), +15]
    cor_y = [-55, float(cw[1]), -10]
    axi_z = [-5, float(cw[2]), +25]

    def _ext_sag(): return [aff[1,3], aff[1,3]+nj*aff[1,1], aff[2,3], aff[2,3]+nk*aff[2,2]]
    def _ext_cor(): return [aff[0,3]+ni*aff[0,0], aff[0,3], aff[2,3], aff[2,3]+nk*aff[2,2]]
    def _ext_axi(): return [aff[0,3]+ni*aff[0,0], aff[0,3], aff[1,3], aff[1,3]+nj*aff[1,1]]
    def _ix(world, axis):
        return max(0, min(cavity.shape[axis] - 1,
            int(round((world - aff[axis, 3]) / aff[axis, axis]))))

    # Companion bone-display extent + indexing (250 um cache may
    # have different shape / affine than the native cavity).
    nb = bone_arr_250.shape
    def _ext_sag_bone():
        return [bone_aff_250[1,3], bone_aff_250[1,3]+nb[1]*bone_aff_250[1,1],
                bone_aff_250[2,3], bone_aff_250[2,3]+nb[2]*bone_aff_250[2,2]]
    def _ext_cor_bone():
        return [bone_aff_250[0,3]+nb[0]*bone_aff_250[0,0], bone_aff_250[0,3],
                bone_aff_250[2,3], bone_aff_250[2,3]+nb[2]*bone_aff_250[2,2]]
    def _ext_axi_bone():
        return [bone_aff_250[0,3]+nb[0]*bone_aff_250[0,0], bone_aff_250[0,3],
                bone_aff_250[1,3], bone_aff_250[1,3]+nb[1]*bone_aff_250[1,1]]
    def _ix_bone(world, axis):
        return max(0, min(bone_arr_250.shape[axis] - 1,
            int(round((world - bone_aff_250[axis, 3]) / bone_aff_250[axis, axis]))))

    def _overlay(ax, sl, ext):
        ax.contour(sl, levels=[0.5], colors=['magenta'], linewidths=0.6,
                    extent=ext)
        ax.imshow(np.where(sl, 1.0, np.nan), cmap='spring',
                   alpha=0.30, origin='lower', extent=ext, aspect='equal',
                   vmin=0, vmax=1, interpolation='nearest')

    # Bone display uses 250 um block-max cache (same as Figure 4).
    # Cavity overlay is at native resolution.
    p_lo, p_hi = (np.percentile(bone_arr_250[bone_arr_250 > 0], [0.5, 99.5])
                  if (bone_arr_250 > 0).any() else (0, 1))
    fig, axes = plt.subplots(3, 3, figsize=(18, 18), constrained_layout=True)
    for c, x in enumerate(sag_x):
        ib = _ix_bone(x, 0); i = _ix(x, 0); ax = axes[0, c]
        ax.imshow(bone_arr_250[ib, :, :].T, cmap='bone',
                   vmin=p_lo, vmax=p_hi,
                   origin='lower', extent=_ext_sag_bone(), aspect='equal')
        _overlay(ax, cavity[i, :, :].T, _ext_sag())
        ax.set_xlim(y_min, y_max); ax.set_ylim(z_min, z_max)
        ax.set_title(f'sagittal x={x:+.1f} mm '
                      f'(bone i={ib} @ 250 um, cavity i={i} @ 60 um)')
        ax.set_xlabel('y A-P (mm)'); ax.set_ylabel('z D-V (mm)')
    for c, y in enumerate(cor_y):
        jb = _ix_bone(y, 1); j = _ix(y, 1); ax = axes[1, c]
        ax.imshow(bone_arr_250[:, jb, :].T, cmap='bone',
                   vmin=p_lo, vmax=p_hi,
                   origin='lower', extent=_ext_cor_bone(), aspect='equal')
        _overlay(ax, cavity[:, j, :].T, _ext_cor())
        ax.set_xlim(x_min, x_max); ax.set_ylim(z_min, z_max)
        ax.set_title(f'coronal y={y:+.1f} mm '
                      f'(bone j={jb} @ 250 um, cavity j={j} @ 60 um)')
        ax.set_xlabel('x R-L (mm)'); ax.set_ylabel('z D-V (mm)')
    for c, z in enumerate(axi_z):
        kb = _ix_bone(z, 2); k = _ix(z, 2); ax = axes[2, c]
        ax.imshow(bone_arr_250[:, :, kb].T, cmap='bone',
                   vmin=p_lo, vmax=p_hi,
                   origin='lower', extent=_ext_axi_bone(), aspect='equal')
        _overlay(ax, cavity[:, :, k].T, _ext_axi())
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_title(f'axial z={z:+.1f} mm '
                      f'(bone k={kb} @ 250 um, cavity k={k} @ 60 um)')
        ax.set_xlabel('x R-L (mm)'); ax.set_ylabel('y A-P (mm)')
    fig.suptitle(f'native cavity @ {voxel_um:.1f} um (direct upsample of '
                  f'250 um result). Bone backdrop: 250 um block-max cache '
                  f'(matches Fig.~4); cavity ({vol_ml:.1f} mL) at native.')
    fig.savefig(OUT_QC, dpi=120)
    plt.close(fig)
    print(f'  wrote {OUT_QC}')


if __name__ == '__main__':
    main()
