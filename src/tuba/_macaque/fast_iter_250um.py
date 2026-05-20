"""Fast iteration loop at 1 mm resolution -- reproduces the mouse
pipeline approach (extract cavity from bone shell + SyN to NMT) at a
coarse-but-fast resolution for rapid debugging.

Pipeline (~30 s end-to-end):
  1. Block-max downsample the 250 um aligned cache to 1 mm.
  2. Extract cavity by the mouse-style recipe:
       BONE_LOW threshold + closing + caudal seal +
       outside-region-grow + largest CC.
  3. SyN: cavity (moving, binary) -> NMT brain mask (fixed, also
     resampled to 1 mm).
  4. Apply inverse warp to push NMT template into macaque space.
  5. Render a QC PNG at 1 mm.

All intermediate files have a `_1mm` suffix so they don't conflict
with the 250 um production caches. Once the parameter set works at
1 mm, scale up by editing TARGET_VOXEL_MM and re-running.

Edit the constants at the top of the script for each iteration:
  TARGET_VOXEL_MM, BONE_LOW, CLOSE_MM, CAUDAL_SEAL_MM, etc.
"""
import os
import sys
import shutil
import time
import numpy as np
import nibabel as nib
import ants
from scipy.ndimage import (binary_closing, binary_fill_holes,
                            label as cc_label)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .macaque_skull_constants import (REG_DIR,
    MACAQUE_RAS_ALIGNED_250UM, _block_max)

# === iteration parameters ===========================================
TARGET_VOXEL_MM = 0.25      # production resolution (250 um, multi-min runtime)
BONE_LOW        = 8000      # cavity-shell seed
CLOSE_MM        = 5.0       # morphological-closing radius
CAUDAL_SEAL_MM  = 0.0       # disabled (curved per-row caudal plug
                              # below already handles the foramen
                              # magnum without flat truncation)

# Per-slice cranial-hull bounds: convex hull of bone restricted to the
# cranium-only AP region (y <= Y_HULL_MAX_MM) and lateral half-width
# (x in +/- X_HULL_HALF_MM). The lateral cutoff used to exclude the
# zygomatic arches but cut into the parietal walls; widened to 35.
# A z-range restriction (Z_HULL_MIN_MM) prevents hull from filling
# the maxilla/face region below the basicranium.
Y_HULL_MAX_MM_LOWZ  = -3.0   # rostral cutoff at LOW z (basicranium
                              # range, where snout/maxilla bone exists
                              # rostral of the cribriform plate)
Y_HULL_MAX_MM_HIGHZ = +20.0  # rostral cutoff at HIGH z (vault range,
                              # where the only bone is the frontal
                              # bone -- safe to include).
Z_YCUTOFF_BREAK_MM  =  0.0   # z value where the cutoff transitions
# (back-compat alias used in floor computation, keep at LOWZ value
# to avoid expanding the basicranium-floor computation rostrally)
Y_HULL_MAX_MM   = Y_HULL_MAX_MM_LOWZ
X_HULL_HALF_MM  = 35.0       # lateral cutoff (cranium width is ~70 mm,
                              # so half-width 35 mm covers the parietals)
Z_HULL_MIN_MM   = -25.0      # bottom of brain-box cavity in z; below
                              # this is the deepest cerebellum/basicranium.
                              # Setting too high crops the cerebellum.
USE_HULL        = True       # if False, just closing+caudal_seal (mouse style)
# Basicranium floor (kept as backup): plug below z=z_bone_min+Z_FLOOR_OFFSET
Z_FLOOR_OFFSET_MM = +2.0     # 2 mm into the basicranium
# ====================================================================

NMT_DIR = os.path.join(REG_DIR, '..', 'templates', 'nmt_v2',
                        'NMT_v2.0_sym', 'NMT_v2.0_sym_fh')
NMT_FH_TEMPLATE = os.path.join(NMT_DIR, 'NMT_v2.0_sym_fh.nii.gz')
NMT_FH_BRAINMASK = os.path.join(NMT_DIR, 'NMT_v2.0_sym_fh_brainmask.nii.gz')

OUT_SKULL_1MM      = os.path.join(REG_DIR, 'macaque_skull_250um_iter.nii.gz')
OUT_CAVITY_1MM     = os.path.join(REG_DIR, 'macaque_cavity_250um.nii.gz')
OUT_BONESHELL      = os.path.join(REG_DIR, 'macaque_cavity_boneshell_250um.nii.gz')
OUT_TEMPLATE_1MM   = os.path.join(REG_DIR, 'nmt_template_in_macaque_250um.nii.gz')
OUT_BRAINMASK      = os.path.join(REG_DIR, 'nmt_brainmask_in_macaque_250um.nii.gz')
QC_PATH            = os.path.join(REG_DIR, 'fast_iter_250um_qc.png')
# Iteration-tagged copy is also written so the user can compare runs.
ITER_TAG = os.environ.get('FAST_ITER_TAG', '').strip()
QC_PATH_TAGGED = (os.path.join(REG_DIR, f'fast_iter_250um_qc_{ITER_TAG}.png')
                  if ITER_TAG else None)


def _downsample_250um_to_target():
    img = nib.load(MACAQUE_RAS_ALIGNED_250UM)
    arr = np.asarray(img.dataobj, dtype=np.uint16)
    aff = img.affine
    voxel_in = float(abs(aff[0, 0]))
    factor = max(1, int(round(TARGET_VOXEL_MM / voxel_in)))
    actual_voxel = voxel_in * factor
    print(f'  in: shape {arr.shape} @ {voxel_in*1000:.1f} um')
    print(f'  factor {factor} -> out {actual_voxel*1000:.1f} um')
    arr = _block_max(arr, factor, axis=0)
    arr = _block_max(arr, factor, axis=1)
    arr = _block_max(arr, factor, axis=2)
    out_aff = aff.copy()
    out_aff[0, 0] = np.sign(aff[0, 0]) * actual_voxel
    out_aff[1, 1] = np.sign(aff[1, 1]) * actual_voxel
    out_aff[2, 2] = np.sign(aff[2, 2]) * actual_voxel
    nib.save(nib.Nifti1Image(arr, out_aff), OUT_SKULL_1MM)
    print(f'  wrote {OUT_SKULL_1MM} (shape {arr.shape})')
    return arr, out_aff


def _extract_cavity_mouse_style(arr, aff):
    voxel_mm = float(abs(aff[0, 0]))
    bone = arr > BONE_LOW
    print(f'  bone fraction: {bone.mean():.4f}')

    # Caudal seal: at the most-caudal cortical-bone slab, fill the
    # bone bbox with shell over CAUDAL_SEAL_MM thickness.
    bone_high = arr > (BONE_LOW * 1.5)
    bone_j = np.where(bone_high.any(axis=(0, 2)))[0]
    seal = np.zeros_like(bone)
    if len(bone_j) > 0:
        j_caudal = int(bone_j.max())
        slab_n = max(1, int(round(CAUDAL_SEAL_MM / voxel_mm)))
        i_idxs, k_idxs = np.where(
            bone_high[:, max(0, j_caudal - slab_n):j_caudal + 1, :].any(axis=1))
        if len(i_idxs) > 0:
            seal[i_idxs.min():i_idxs.max() + 1,
                 max(0, j_caudal - slab_n + 1):j_caudal + 1,
                 k_idxs.min():k_idxs.max() + 1] = True
        print(f'  caudal seal: j_caudal={j_caudal}, slab={slab_n}, '
              f'voxels={int(seal.sum())}')

    iters = max(1, int(round(CLOSE_MM / voxel_mm)))
    closed_bone = binary_closing(bone, iterations=iters)

    # Per-column curved basicranium floor: for each (i, j) column with
    # closed-bone in the brain-box AP region, plug everything below the
    # lowest closed-bone in that column. Naturally follows the
    # basicranium curve (cerebellum lower than cerebrum), no flat
    # z-plane edge.
    j_lo = int(round((Y_HULL_MAX_MM - aff[1, 3]) / aff[1, 1]))
    j_lo = max(0, min(arr.shape[1], j_lo))
    floor = np.zeros_like(closed_bone)
    cranial_subvol = closed_bone[:, j_lo:, :]
    bone_so_far_below = np.cumsum(
        cranial_subvol.astype(np.int8), axis=2) > 0
    col_has_bone = cranial_subvol.any(axis=2, keepdims=True)
    floor[:, j_lo:, :] = col_has_bone & ~bone_so_far_below
    print(f'  per-column basicranium floor (curved): '
          f'{floor.sum()} voxels = {float(floor.sum())/floor.size*100:.1f}%')

    # Per-row curved ROSTRAL plug: for each (i, k) row in the brain-box
    # x and z range, plug ROSTRAL of the most-rostral closed-bone in
    # that row. Follows the cribriform plate / frontal bone curve,
    # avoiding a flat y-plane edge at the rostral cavity boundary.
    # In our convention +y is anterior (j=0 is most anterior), so
    # "rostral" = small j; "most-rostral closed-bone" = smallest j with
    # closed_bone True. We plug j < that.
    rostral_plug = np.zeros_like(closed_bone)
    bone_so_far_rostral = np.cumsum(
        closed_bone.astype(np.int8), axis=1) > 0   # cumsum from j=0 up
    row_has_bone = closed_bone.any(axis=1, keepdims=True)
    rostral_plug[:, :, :] = row_has_bone & ~bone_so_far_rostral
    print(f'  per-row curved rostral plug: {rostral_plug.sum()} voxels '
          f'= {float(rostral_plug.sum())/rostral_plug.size*100:.1f}%')

    # Per-row curved CAUDAL plug: mirror -- for each (i, k) row, plug
    # CAUDAL of the most-caudal closed-bone (largest j with bone).
    bone_so_far_caudal = np.flip(
        np.cumsum(np.flip(closed_bone.astype(np.int8), axis=1), axis=1),
        axis=1) > 0
    caudal_plug = np.zeros_like(closed_bone)
    caudal_plug[:, :, :] = row_has_bone & ~bone_so_far_caudal
    print(f'  per-row curved caudal plug: {caudal_plug.sum()} voxels '
          f'= {float(caudal_plug.sum())/caudal_plug.size*100:.1f}%')

    # Combine: floor + rostral + caudal
    floor = floor | rostral_plug | caudal_plug
    print(f'  combined floor + rostral + caudal: {floor.sum()} voxels '
          f'= {float(floor.sum())/floor.size*100:.1f}%')

    if USE_HULL:
        from scipy.spatial import ConvexHull
        from matplotlib.path import Path
        ni_, nj_, nk_ = arr.shape
        i_x_pos = int(round((+X_HULL_HALF_MM - aff[0, 3]) / aff[0, 0]))
        i_x_neg = int(round((-X_HULL_HALF_MM - aff[0, 3]) / aff[0, 0]))
        i_lo, i_hi = sorted([max(0, i_x_pos), min(ni_, i_x_neg)])
        k_z_min = int(round((Z_HULL_MIN_MM - aff[2, 3]) / aff[2, 2]))
        k_z_min = max(0, min(nk_, k_z_min))

        def _slice_hull(slice_2d, axis_lo_offsets, out_shape):
            """Compute per-slice convex hull of slice_2d (binary).
            axis_lo_offsets shifts the hull point coords back to global.
            Returns a binary mask of shape out_shape."""
            pts = np.argwhere(slice_2d)
            if len(pts) < 3:
                return None
            pts[:, 0] += axis_lo_offsets[0]
            pts[:, 1] += axis_lo_offsets[1]
            try:
                hull = ConvexHull(pts)
            except Exception:
                return None
            grid_a, grid_b = np.meshgrid(
                np.arange(out_shape[0]), np.arange(out_shape[1]),
                indexing='ij')
            coords = np.column_stack([grid_a.ravel(), grid_b.ravel()])
            inside = Path(pts[hull.vertices]).contains_points(
                coords).reshape(out_shape)
            return inside

        # Per-AXIAL-slice hull (varies in z), bounds the cavity in
        # (i=LR, j=AP) per slice. Y cutoff is z-dependent: at high z
        # (vault region, no snout/maxilla bone present), include all
        # bone (Y_HULL_MAX_HIGHZ_MM); at low z (basicranium range,
        # snout bone present), restrict to cranium AP only.
        hull_axial = np.zeros_like(closed_bone)
        n_axial = 0
        j_hi_lowz = int(round((Y_HULL_MAX_MM_LOWZ - aff[1, 3]) / aff[1, 1]))
        j_hi_highz = int(round((Y_HULL_MAX_MM_HIGHZ - aff[1, 3]) / aff[1, 1]))
        j_hi_lowz = max(0, min(nj_, j_hi_lowz))
        j_hi_highz = max(0, min(nj_, j_hi_highz))
        k_break = int(round(
            (Z_YCUTOFF_BREAK_MM - aff[2, 3]) / aff[2, 2]))
        k_break = max(0, min(nk_, k_break))
        for k in range(k_z_min, nk_):
            # j_lo (rostral cutoff for the hull input) depends on z
            j_lo_k = j_hi_lowz if k < k_break else j_hi_highz
            sl = closed_bone[i_lo:i_hi, j_lo_k:, k]
            if sl.sum() < 10:
                continue
            inside = _slice_hull(
                sl, axis_lo_offsets=(i_lo, j_lo_k),
                out_shape=(ni_, nj_))
            if inside is None:
                continue
            hull_axial[:, :, k] = inside
            n_axial += 1

        # Per-CORONAL-slice hull (varies in y), bounds the cavity in
        # (i=LR, k=DV) per slice.
        hull_coronal = np.zeros_like(closed_bone)
        n_coronal = 0
        for j in range(j_lo, nj_):
            sl = closed_bone[i_lo:i_hi, j, k_z_min:]
            if sl.sum() < 10:
                continue
            inside = _slice_hull(
                sl, axis_lo_offsets=(i_lo, k_z_min),
                out_shape=(ni_, nk_))
            if inside is None:
                continue
            hull_coronal[:, j, :] = inside
            n_coronal += 1

        # Cavity is inside the axial hull (single-direction; intersection
        # with coronal hull was too restrictive and created mid-cavity
        # holes). The per-column basicranium floor handles the ventral
        # boundary.
        cranial_hull = hull_axial
        outside_hull = ~cranial_hull
        shell = closed_bone | seal | floor | outside_hull
        print(f'  hull restricted to '
              f'i=[{i_lo},{i_hi}) (x +/- {X_HULL_HALF_MM} mm), '
              f'j>={j_lo} (y <= {Y_HULL_MAX_MM} mm), '
              f'k>={k_z_min} (z >= {Z_HULL_MIN_MM} mm); '
              f'{n_axial} axial slice hulls; shell={shell.mean():.4f}')
    else:
        shell = closed_bone | seal | floor
        print(f'  shell fraction (closing only): {shell.mean():.4f}')

    not_shell = ~shell
    out_seed = np.zeros_like(not_shell)
    out_seed[0, :, :] = True; out_seed[-1, :, :] = True
    out_seed[:, 0, :] = True; out_seed[:, -1, :] = True
    out_seed[:, :, 0] = True; out_seed[:, :, -1] = True
    out_seed &= not_shell
    labeled_ns, _ = cc_label(not_shell)
    edge_ids = np.unique(labeled_ns[out_seed])
    edge_ids = edge_ids[edge_ids > 0]
    outside = np.isin(labeled_ns, edge_ids)
    cavity_raw = not_shell & ~outside
    print(f'  raw cavity volume: '
          f'{cavity_raw.sum() * voxel_mm**3 / 1000:.1f} mL')

    labeled, n_cc = cc_label(cavity_raw)
    sizes = np.bincount(labeled.ravel()); sizes[0] = 0
    if len(sizes) <= 1:
        print('  no cavity CCs found')
        return cavity_raw, 0.0
    main = int(np.argmax(sizes))
    cavity = (labeled == main)
    vol_ml_pre = cavity.sum() * voxel_mm**3 / 1000
    print(f'  largest pre-fill CC: {vol_ml_pre:.1f} mL')

    # Geodesic propagation: grow the largest bone-shell CC into all
    # bone-bounded pockets (petrous / sphenoid / ear-bulla isolated
    # regions) without crossing bone or the outside envelope. The
    # boundary stays bone-traced everywhere -- never tracks soft
    # tissue. scipy.ndimage.binary_propagation does morphological
    # reconstruction in a single C call (equivalent to dilation-to-
    # convergence inside a mask), so this is fast even at 250 um.
    from scipy.ndimage import binary_propagation as _bp
    inside_hull_mask = (~outside_hull) if 'outside_hull' in dir() else (~outside)
    allowed = (~bone) & (~outside) & inside_hull_mask & (~floor) & (~seal)
    cavity = _bp(cavity, mask=allowed)
    vol_ml = cavity.sum() * voxel_mm**3 / 1000
    print(f'  geodesic propagation (bone-constrained): {vol_ml:.1f} mL')
    return cavity, vol_ml


def _syn_cavity_to_nmt(cavity, aff_macaque):
    cavity_path = os.path.join(REG_DIR, '_cavity_250um.nii.gz')
    nib.save(nib.Nifti1Image(cavity.astype(np.float32), aff_macaque),
             cavity_path)
    moving = ants.image_read(cavity_path)

    # Resample NMT brain mask to TARGET_VOXEL_MM for fast SyN.
    nmt_bm = ants.image_read(NMT_FH_BRAINMASK)
    target_spacing = (TARGET_VOXEL_MM,) * 3
    nmt_bm_lo = ants.resample_image(nmt_bm, target_spacing,
                                     use_voxels=False, interp_type=1)
    print(f'  cavity {moving.shape} @ {moving.spacing[0]*1000:.0f} um, '
          f'NMT brain mask resampled '
          f'{nmt_bm.shape}->{nmt_bm_lo.shape} @ '
          f'{nmt_bm_lo.spacing[0]*1000:.0f} um')

    # Use AFFINE-only (12 DOF: translation + rotation + scale + shear).
    # SyN warps the brain mask non-rigidly to match the bone-shell
    # cavity, which propagates soft-tissue features (longitudinal
    # fissure, falx) of the cavity into the warped mask. Affine
    # cannot deform locally -- the warped mask retains its smooth
    # atlas topology while still finding the best global pose+scale.
    print('  ANTs Affine cavity -> NMT brain mask (12 DOF, MeanSquares)...')
    reg = ants.registration(
        fixed=nmt_bm_lo, moving=moving,
        type_of_transform='Affine',
        aff_metric='meansquares',
        verbose=False,
    )
    inv = reg['invtransforms']

    print('  warping NMT template -> macaque space (linear)...')
    nmt_t = ants.image_read(NMT_FH_TEMPLATE)
    fixed_mac = ants.image_read(OUT_SKULL_1MM)
    warped_t = ants.apply_transforms(
        fixed=fixed_mac, moving=nmt_t,
        transformlist=inv,
        whichtoinvert=[t.endswith('.mat') for t in inv],
        interpolator='linear')
    ants.image_write(warped_t, OUT_TEMPLATE_1MM)
    print(f'  wrote {OUT_TEMPLATE_1MM}')

    # Halle/human-pipeline lesson: the warped NMT brain MASK has the
    # correct brain topology by construction (atlas-derived). Use it
    # as the production cavity -- the bone-shell extraction was just
    # a rough target to drive SyN, but the atlas-warped mask captures
    # the brain shape without internal-bone holes (petrous, sphenoid
    # wings) that fragment the bone-shell cavity into separate CCs.
    print('  warping NMT brain mask -> macaque space (linear)...')
    nmt_bm_full = ants.image_read(NMT_FH_BRAINMASK)
    warped_bm = ants.apply_transforms(
        fixed=fixed_mac, moving=nmt_bm_full,
        transformlist=inv,
        whichtoinvert=[t.endswith('.mat') for t in inv],
        interpolator='linear')
    bm_arr = (warped_bm.numpy() > 0.5)

    # Light smoothing (2 mm) to clean up NN-interpolation jaggedness,
    # then bone-as-constraint clip. With affine-only registration the
    # warped mask retains its smooth atlas topology (no soft-tissue
    # tracking from non-rigid warp), so we don't need aggressive
    # closing to fill warp-induced indentations.
    from scipy.ndimage import binary_closing as _bc_bm
    smooth_iters = max(1, int(round(2.0 / TARGET_VOXEL_MM)))
    bm_arr = _bc_bm(bm_arr, iterations=smooth_iters)
    ct_img = nib.load(OUT_SKULL_1MM)
    ct_arr = np.asarray(ct_img.dataobj, dtype=np.uint16)
    bone_mask = ct_arr > BONE_LOW
    bm_arr = bm_arr & ~bone_mask
    nib.save(nib.Nifti1Image(bm_arr.astype(np.uint8), aff_macaque),
             OUT_BRAINMASK)
    bm_vol = float(bm_arr.sum()) * (TARGET_VOXEL_MM ** 3) / 1000
    print(f'  wrote {OUT_BRAINMASK}; atlas cavity volume {bm_vol:.1f} mL '
          f'(after {smooth_iters}-iter / 2 mm smoothing + bone clip)')

    if os.path.exists(cavity_path):
        os.remove(cavity_path)
    return bm_arr > 0.5  # atlas-warped cavity for downstream / QC


def _render_qc(arr, cavity, aff, vol_ml):
    template_arr = (np.asarray(nib.load(OUT_TEMPLATE_1MM).dataobj)
                    .astype(np.float32) if os.path.exists(OUT_TEMPLATE_1MM)
                    else None)

    ni, nj, nk = arr.shape
    bone_idx = np.argwhere(arr > 8000)
    if len(bone_idx) > 0:
        bw = (aff[:3, :3] @ bone_idx.T).T + aff[:3, 3]
        x_min, x_max = bw[:, 0].min() - 5, bw[:, 0].max() + 5
        y_min, y_max = bw[:, 1].min() - 5, bw[:, 1].max() + 5
        z_min, z_max = bw[:, 2].min() - 5, bw[:, 2].max() + 10
    else:
        x_min, x_max = -50, 50
        y_min, y_max = -70, 70
        z_min, z_max = -45, 50

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
        return max(0, min(arr.shape[axis] - 1,
            int(round((world - aff[axis, 3]) / aff[axis, axis]))))

    def _overlay(ax, sl, ext, color='magenta', alpha=0.45):
        ax.imshow(np.where(sl, 1.0, np.nan), cmap='spring',
                   alpha=alpha, origin='lower', extent=ext, aspect='equal',
                   vmin=0, vmax=1, interpolation='nearest')
        ax.contour(sl, levels=[0.5], colors=[color], linewidths=1.0,
                    extent=ext)

    p_lo, p_hi = (np.percentile(arr[arr > 0], [0.5, 99.5])
                  if (arr > 0).any() else (0, 1))
    fig, axes = plt.subplots(3, 3, figsize=(18, 18), constrained_layout=True)
    for c, x in enumerate(sag_x):
        i = _ix(x, 0); ax = axes[0, c]
        ax.imshow(arr[i, :, :].T, cmap='bone', vmin=p_lo, vmax=p_hi,
                   origin='lower', extent=_ext_sag(), aspect='equal')
        if template_arr is not None:
            tov = np.where(template_arr[i, :, :].T > 1, template_arr[i, :, :].T, np.nan)
            ax.imshow(tov, cmap='hot', alpha=0.4, origin='lower',
                       extent=_ext_sag(), aspect='equal')
        _overlay(ax, cavity[i, :, :].T, _ext_sag())
        ax.set_xlim(y_min, y_max); ax.set_ylim(z_min, z_max)
        ax.set_title(f'sagittal i={i} (x={x:+.1f} mm)')
        ax.set_xlabel('y A-P (mm)'); ax.set_ylabel('z D-V (mm)')
    for c, y in enumerate(cor_y):
        j = _ix(y, 1); ax = axes[1, c]
        ax.imshow(arr[:, j, :].T, cmap='bone', vmin=p_lo, vmax=p_hi,
                   origin='lower', extent=_ext_cor(), aspect='equal')
        if template_arr is not None:
            tov = np.where(template_arr[:, j, :].T > 1, template_arr[:, j, :].T, np.nan)
            ax.imshow(tov, cmap='hot', alpha=0.4, origin='lower',
                       extent=_ext_cor(), aspect='equal')
        _overlay(ax, cavity[:, j, :].T, _ext_cor())
        ax.set_xlim(x_min, x_max); ax.set_ylim(z_min, z_max)
        ax.set_title(f'coronal j={j} (y={y:+.1f} mm)')
        ax.set_xlabel('x R-L (mm)'); ax.set_ylabel('z D-V (mm)')
    for c, z in enumerate(axi_z):
        k = _ix(z, 2); ax = axes[2, c]
        ax.imshow(arr[:, :, k].T, cmap='bone', vmin=p_lo, vmax=p_hi,
                   origin='lower', extent=_ext_axi(), aspect='equal')
        if template_arr is not None:
            tov = np.where(template_arr[:, :, k].T > 1, template_arr[:, :, k].T, np.nan)
            ax.imshow(tov, cmap='hot', alpha=0.4, origin='lower',
                       extent=_ext_axi(), aspect='equal')
        _overlay(ax, cavity[:, :, k].T, _ext_axi())
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_title(f'axial k={k} (z={z:+.1f} mm)')
        ax.set_xlabel('x R-L (mm)'); ax.set_ylabel('y A-P (mm)')
    fig.suptitle(f'FAST iter @ {TARGET_VOXEL_MM} mm  (BONE_LOW={BONE_LOW}, '
                  f'CLOSE_MM={CLOSE_MM}, CAUDAL_SEAL_MM={CAUDAL_SEAL_MM}). '
                  f'Magenta: cavity ({vol_ml:.1f} mL); '
                  f'red overlay: warped NMT template.')
    fig.savefig(QC_PATH, dpi=120)
    if QC_PATH_TAGGED:
        fig.savefig(QC_PATH_TAGGED, dpi=120)
        print(f'wrote {QC_PATH_TAGGED}')
    plt.close(fig)
    print(f'wrote {QC_PATH}')


def main():
    t0 = time.time()
    print(f'=== Fast iteration @ {TARGET_VOXEL_MM} mm ===')
    print('Step 1: downsample to working resolution...')
    arr, aff = _downsample_250um_to_target()
    print(f'  [{time.time()-t0:.1f} s]')

    print('\nStep 2: bone-bounded cavity (SyN driver, intermediate)...')
    boneshell_cavity, vol_ml = _extract_cavity_mouse_style(arr, aff)
    nib.save(nib.Nifti1Image(boneshell_cavity.astype(np.uint8), aff),
             OUT_BONESHELL)
    print(f'  wrote bone-shell intermediate -> {OUT_BONESHELL}')
    print(f'  [{time.time()-t0:.1f} s]')

    cavity = boneshell_cavity
    if vol_ml > 1.0:
        print('\nStep 3: Affine NMT mask -> macaque (production cavity)...')
        atlas_cavity = _syn_cavity_to_nmt(boneshell_cavity, aff)
        cavity = atlas_cavity
        nib.save(nib.Nifti1Image(cavity.astype(np.uint8), aff),
                 OUT_CAVITY_1MM)
        vol_ml = float(cavity.sum()) * (TARGET_VOXEL_MM ** 3) / 1000
        print(f'  wrote production atlas-affine cavity ({vol_ml:.1f} mL) '
              f'-> {OUT_CAVITY_1MM}')
        print(f'  [{time.time()-t0:.1f} s]')
    else:
        print('\nStep 3: SKIPPED (cavity too small)')
        nib.save(nib.Nifti1Image(cavity.astype(np.uint8), aff),
                 OUT_CAVITY_1MM)

    print('\nStep 4: render QC...')
    _render_qc(arr, cavity, aff, vol_ml)
    print(f'\nDone in {time.time()-t0:.1f} s.')


if __name__ == '__main__':
    main()
