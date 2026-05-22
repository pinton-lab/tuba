"""Endocranial cavity extraction from an aligned skull microCT volume.

Hysteresis-threshold strategy (mouse manuscript §6, macaque extends
with optional basicranium-floor and rostral seals via
:class:`SealRegion`):

1. ``bone_low = vol > bone_low_thresh`` -- bone + adherent mounting
   material; seeds the shell that closing dilates and then erodes.
2. Caudal seal: fill the (i, k) bone bounding box at the most-caudal
   cortical-bone slice over the caudal ``caudal_seal_mm`` slab to plug
   the foramen magnum.
3. ``shell = closing(bone_low, close_iters) | caudal_seal | extra_seals``
   -- morphological closing (dilation followed by erosion at the same
   radius) seals foramina up to ~2x the structuring-element radius
   without the net inward boundary shift that pure dilation would
   impose at this working resolution.
4. ``outside = `` connected components of ``~shell`` that touch the
   volume border (region-grow from the volume edges).
5. ``cavity_raw = ~shell & ~outside`` -- every voxel inside the
   bone+mounting envelope that is not classified as shell.
6. Keep the largest CC. For the mouse Maga at 25 um this is the
   cerebrum + cerebellum + brainstem compartment (~450 mm^3); the
   olfactory-bulb cavity (~50 mm^3) appears as a separate CC once
   the cribriform plate is sealed -- it is deliberately NOT retained,
   so topology matches the connected Allen brain mask, and the OB
   region is recovered post-hoc via the SyN deformation field.

Manuscript §6 documents two non-obvious design points that cost
several debug cycles in the mouse migration:

* **Atlas-matching resolution**: run the extractor at the atlas
  template's grid (Allen: 25 um, derived from the *aligned native* via
  factor-4 block-max -- NOT from raw TIFFs followed by independent
  PCA, which biases the bone centroid ventrally by ~1 mm because
  block-max smears the 4-8 kcts mounting-material plateau into
  apparent bone). At a coarse 200 um cache, the cavity contour shows
  visible 0.2 mm staircase on native-resolution overlays and a 0.4 mm
  inward boundary shift from block-max smearing.

* **Stop at bone-low, not bone-high**: extending the cavity to the
  cortical (bone-high) inner edge via CC-selection on
  ``~closing(bone_high) & ~outside`` leaks through sutures and thin
  spots that even a 1 mm closing of bone-high cannot bridge -- the
  test run produced 1963 mm^3 (engulfing the mounting-material region)
  instead of the expected 450 mm^3. The bone-low surface sits ~1
  cortical-bone thickness (~0.1 mm for mouse) inside the true inner
  cortical surface -- well below visual scale and reliable
  topologically.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.ndimage import binary_closing, label as cc_label


@dataclass
class SealRegion:
    """A rectangular ROI to add to the shell mask before region-growing.

    Used by the macaque pipeline for the basicranium floor seal (large
    foramen magnum, choanae, jugular foramina) and the rostral/orbital
    plug. Mouse uses only the auto-derived caudal seal.

    The ROI is in **storage-index** space (not world-mm); the species
    module is responsible for converting from atlas regions if needed.
    """
    i_lo: int
    i_hi: int
    j_lo: int
    j_hi: int
    k_lo: int
    k_hi: int


def extract_cavity(vol, voxel_mm, bone_high, bone_low,
                   close_mm, caudal_seal_mm=0.6,
                   extra_seals=None, verbose=True):
    """Extract the endocranial cavity binary mask.

    Parameters
    ----------
    vol : 3D ndarray
        Intensity volume at the atlas-matching resolution.
    voxel_mm : float
        Isotropic voxel size in mm.
    bone_high : float
        Cortical-bone threshold (used to locate the foramen-magnum slice
        for the caudal seal).
    bone_low : float
        Bone + mounting-material threshold (the cavity stops here, NOT
        at ``bone_high``).
    close_mm : float
        Morphological-closing radius in mm.
    caudal_seal_mm : float, optional
        Thickness of the auto-derived foramen-magnum plug, in mm. Set
        to None or 0 to disable.
    extra_seals : list[SealRegion], optional
        Additional rectangular seal regions added to the shell
        (basicranium floor for macaque, etc.).

    Returns
    -------
    cavity : 3D bool ndarray, same shape as ``vol``.
    info : dict with diagnostic fields (volumes, CC counts, ...).
    """
    close_iters = max(1, int(round(close_mm / voxel_mm)))
    info = {'close_iters': close_iters}
    if verbose:
        print(f'  voxel-scaled: close_iters={close_iters} ({close_mm} mm)')
        print(f'Hysteresis thresholds: BONE_HIGH={bone_high} (cortical), '
              f'BONE_LOW={bone_low} (shell)')
    bone_high_mask = vol > bone_high
    bone_low_mask = vol > bone_low
    info['bone_high_fraction'] = float(bone_high_mask.mean())
    info['bone_low_fraction'] = float(bone_low_mask.mean())
    if verbose:
        print(f'  bone_high fraction: {info["bone_high_fraction"]:.4f}')
        print(f'  bone_low  fraction: {info["bone_low_fraction"]:.4f}')

    seal = np.zeros_like(bone_low_mask)
    if caudal_seal_mm:
        caudal_seal_voxels = max(1, int(round(caudal_seal_mm / voxel_mm)))
        info['caudal_seal_voxels'] = caudal_seal_voxels
        if verbose:
            print(f'  caudal_seal_voxels={caudal_seal_voxels} '
                  f'({caudal_seal_mm} mm)')
        bone_j = np.where(bone_high_mask.any(axis=(0, 2)))[0]
        if len(bone_j) == 0:
            raise RuntimeError('No cortical bone found -- check bone_high')
        j_caudal = int(bone_j.max())
        info['j_caudal'] = j_caudal
        if verbose:
            print(f'  caudal-most cortical-bone slice: j={j_caudal}')
        i_idxs, k_idxs = np.where(
            bone_high_mask[
                :, max(0, j_caudal - close_iters):j_caudal + 1, :
            ].any(axis=1))
        if len(i_idxs) > 0:
            i_lo, i_hi = int(i_idxs.min()), int(i_idxs.max()) + 1
            k_lo, k_hi = int(k_idxs.min()), int(k_idxs.max()) + 1
            j_seal_lo = max(0, j_caudal - caudal_seal_voxels + 1)
            seal[i_lo:i_hi, j_seal_lo:j_caudal + 1, k_lo:k_hi] = True
            if verbose:
                print(f'  caudal seal i=[{i_lo},{i_hi}) '
                      f'j=[{j_seal_lo},{j_caudal+1}) k=[{k_lo},{k_hi})  '
                      f'({seal.sum()} voxels)')

    if extra_seals:
        for r in extra_seals:
            seal[r.i_lo:r.i_hi, r.j_lo:r.j_hi, r.k_lo:r.k_hi] = True

    if verbose:
        print(f'Closing bone_low ({close_iters} iter) | seal ...')
    shell = binary_closing(bone_low_mask, iterations=close_iters) | seal
    info['shell_fraction'] = float(shell.mean())
    if verbose:
        print(f'  shell fraction: {info["shell_fraction"]:.4f}')

    if verbose:
        print('Region-grow outside-of-skull via shell ...')
    not_shell = ~shell
    out_seed = np.zeros_like(not_shell)
    out_seed[0, :, :] = True;  out_seed[-1, :, :] = True
    out_seed[:, 0, :] = True;  out_seed[:, -1, :] = True
    out_seed[:, :, 0] = True;  out_seed[:, :, -1] = True
    out_seed &= not_shell
    labeled_ns, _ = cc_label(not_shell)
    edge_ids = np.unique(labeled_ns[out_seed])
    edge_ids = edge_ids[edge_ids > 0]
    outside = np.isin(labeled_ns, edge_ids)
    info['outside_fraction'] = float(outside.mean())
    if verbose:
        print(f'  outside-of-skull fraction: {outside.mean():.4f}')

    cavity_raw = not_shell & ~outside
    raw_vol_mm3 = float(cavity_raw.sum()) * voxel_mm**3
    info['cavity_raw_mm3'] = raw_vol_mm3
    if verbose:
        print(f'  raw cavity (after shell + outside): {raw_vol_mm3:.0f} mm^3')

    if verbose:
        print('Keeping the largest connected component ...')
    labeled, n_cc = cc_label(cavity_raw)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    main_label = int(np.argmax(sizes))
    cavity = (labeled == main_label)
    final_vol_mm3 = float(cavity.sum()) * voxel_mm**3
    info['n_cc'] = n_cc
    info['main_label'] = main_label
    info['cavity_mm3'] = final_vol_mm3
    top5 = np.argsort(sizes)[::-1][:5]
    if verbose:
        print(f'  top-5 CC sizes (mm^3): '
              f'{[f"{int(sizes[i])*voxel_mm**3:.0f}" for i in top5]}  '
              f'(out of {n_cc} CCs total)')
        print(f'  selected CC label={main_label}, '
              f'volume={final_vol_mm3:.0f} mm^3')
    return cavity, info


def bone_shell_cavity(arr, aff, *, bone_low,
                       close_mm, x_hull_half_mm,
                       y_hull_max_lowz_mm, y_hull_max_highz_mm,
                       z_hull_min_mm, z_ycutoff_break_mm,
                       caudal_seal_mm=0.0, use_hull=True,
                       use_geodesic_propagation=True,
                       smooth_plug_mm=0.0,
                       verbose=True):
    """Bone-shell-bounded cavity extractor (macaque recipe).

    For species whose cranium has anatomical openings wider than a
    practical morphological closing radius (orbits, foramen magnum,
    temporal fossa, choanae), the pure-hysteresis approach in
    :func:`extract_cavity` leaks. This recipe instead constructs an
    anatomically-tight bone-shell envelope from:

    * per-column curved basicranium floor: for each (i, j) column with
      bone in the brain-box AP region (j >= j_lo from
      ``y_hull_max_lowz_mm``), plug everything ventral of the lowest
      closed-bone in that column;
    * per-row curved rostral plug + per-row curved caudal plug
      (mirror), following the cribriform plate and foramen magnum
      curves;
    * per-axial-slice convex hull of bone with z-dependent rostral
      cutoff (``y_hull_max_lowz_mm`` at z < ``z_ycutoff_break_mm``,
      ``y_hull_max_highz_mm`` above), lateral half-width
      ``x_hull_half_mm``;
    * shell = closing(bone_low) | floor | rostral_plug | caudal_plug |
      ~axial_hull.

    Cavity = NOT shell AND NOT (region-grown outside from boundary),
    largest CC, optional geodesic propagation
    (:func:`scipy.ndimage.binary_propagation`) into bone-bounded
    pockets (petrous / sphenoid / ear-bulla isolated regions) without
    crossing bone or the outside envelope.

    This is the macaque seed for ANTs Affine to an atlas brain mask
    (see :mod:`tuba._macaque.fast_iter_250um._syn_cavity_to_nmt`);
    rat reuses it with rat-scale constants.

    All ``*_mm`` parameters are species-specific. World convention is
    RAS: +x right, +y anterior, +z dorsal; the volume affine ``aff``
    converts (i, j, k) voxel indices to world mm.

    Parameters
    ----------
    arr : 3D ndarray
        Intensity volume aligned in RAS world frame.
    aff : (4, 4) ndarray
        World affine of ``arr``.
    bone_low : float
        Intensity threshold for the bone seed.
    close_mm : float
        Morphological closing radius for the bone seed.
    x_hull_half_mm : float
        Lateral half-width of the per-axial hull (excludes zygomatic
        flares).
    y_hull_max_lowz_mm, y_hull_max_highz_mm : float
        Rostral cutoffs: include bone with ``y <= y_hull_max`` at
        z < z_ycutoff_break_mm and z >= z_ycutoff_break_mm
        respectively. Low-z value excludes the snout; high-z value
        includes the frontal vault.
    z_hull_min_mm : float
        Ventral cutoff for the axial hull (below this is below the
        basicranium plate).
    z_ycutoff_break_mm : float
        z value where the rostral cutoff transitions from low-z to
        high-z.
    caudal_seal_mm : float, default 0.0
        Thickness of a flat foramen-magnum seal slab (mouse-style).
        The per-row curved caudal plug below already handles the
        foramen magnum without flat truncation, so the default is
        0.0 (disabled).
    use_hull : bool, default True
        If False, skip the per-axial hull (mouse-style: closing +
        seal + floor + plugs only).
    use_geodesic_propagation : bool, default True
        If True, grow the largest CC into bone-bounded pockets via
        :func:`scipy.ndimage.binary_propagation`.
    smooth_plug_mm : float, default 0.0
        2D median-filter radius applied to the per-column basicranium
        floor and the per-row rostral / caudal plug boundary positions
        before the plug masks are built. Default 0.0 keeps the original
        macaque behaviour (jagged per-row stepped boundary). Values
        2-4 mm give a smoothly curved plug surface, which is useful
        for the rat at 118 um voxel pitch where the per-row plug's
        stepping is visible in the cavity contour.

    Returns
    -------
    cavity : 3D bool ndarray
        The extracted cavity mask.
    info : dict
        Diagnostic info: ``bone_fraction``, ``shell_fraction``,
        ``cavity_raw_mm3``, ``cavity_mm3``, etc.
    """
    from scipy.ndimage import (binary_closing as _bc,
                                binary_propagation as _bp,
                                label as _label)
    voxel_mm = float(abs(aff[0, 0]))
    bone = arr > bone_low
    info = {'bone_fraction': float(bone.mean())}
    if verbose:
        print(f'  bone fraction: {info["bone_fraction"]:.4f}')

    # Caudal seal (optional, default disabled).
    seal = np.zeros_like(bone)
    if caudal_seal_mm > 0:
        bone_high = arr > (bone_low * 1.5)
        bone_j = np.where(bone_high.any(axis=(0, 2)))[0]
        if len(bone_j) > 0:
            j_caudal = int(bone_j.max())
            slab_n = max(1, int(round(caudal_seal_mm / voxel_mm)))
            ix, kx = np.where(bone_high[:,
                max(0, j_caudal - slab_n):j_caudal + 1, :].any(axis=1))
            if len(ix) > 0:
                seal[ix.min():ix.max() + 1,
                     max(0, j_caudal - slab_n + 1):j_caudal + 1,
                     kx.min():kx.max() + 1] = True
            if verbose:
                print(f'  caudal seal: j_caudal={j_caudal}, '
                      f'slab={slab_n}, voxels={int(seal.sum())}')

    iters = max(1, int(round(close_mm / voxel_mm)))
    closed_bone = _bc(bone, iterations=iters)

    # Per-column basicranium floor + per-row rostral / caudal plugs.
    # If smooth_plug_mm > 0, the boundary positions (per-column lowest
    # bone, per-row most-rostral bone, per-row most-caudal bone) are
    # median-smoothed in their respective 2D planes before the plug
    # masks are built. This replaces the jagged per-row stepped
    # boundary with a curved surface that follows the average bone
    # extent, eliminating the visible stepping in cavity contours at
    # high voxel resolution (rat 118 um).
    j_lo = int(round((y_hull_max_lowz_mm - aff[1, 3]) / aff[1, 1]))
    j_lo = max(0, min(arr.shape[1], j_lo))
    ni_, nj_, nk_ = arr.shape
    n_smooth = max(0, int(round(smooth_plug_mm / voxel_mm)))
    smooth_size = 2 * n_smooth + 1 if n_smooth > 0 else 1

    def _smooth2d(arr2d, size):
        if size <= 1:
            return arr2d
        from scipy.ndimage import median_filter
        return median_filter(arr2d.astype(np.float32), size=size)

    # ---- Per-column basicranium floor ----
    # k_first[i, j] = smallest k where closed_bone[i, j, k] is True
    # (the basicranium-plate inner surface in column (i,j)). When
    # smoothed, the resulting surface follows the average basicranium
    # curve. Floor = everything ventral of that surface.
    cranial_subvol = closed_bone[:, j_lo:, :]
    col_has_bone = cranial_subvol.any(axis=2)
    # argmax returns index of FIRST True along axis (k=0 ventral side
    # after the COL_FLIP=True convention; smallest k = most ventral).
    k_first = np.argmax(cranial_subvol, axis=2)
    # Where col_has_bone == False, argmax returns 0 -- mark those as nk_
    # (no plug there) so they don't drag the smoothed surface down.
    k_first = np.where(col_has_bone, k_first, nk_)
    k_first_smooth = _smooth2d(k_first, smooth_size)
    k_grid = np.arange(nk_)[None, None, :]
    floor = np.zeros_like(closed_bone)
    floor[:, j_lo:, :] = col_has_bone[..., None] & (k_grid < k_first_smooth[..., None])
    if verbose:
        print(f'  per-column basicranium floor: {int(floor.sum())} voxels'
              f'{" (smoothed)" if smooth_size > 1 else ""}')

    # ---- Per-row rostral plug ----
    # j_first[i, k] = smallest j where closed_bone[i, j, k] is True
    # (most-rostral bone in row (i, k)). Plug = everything rostral.
    j_first = np.argmax(closed_bone, axis=1)
    row_has_bone = closed_bone.any(axis=1)
    j_first = np.where(row_has_bone, j_first, nj_)
    j_first_smooth = _smooth2d(j_first, smooth_size)
    j_grid = np.arange(nj_)[None, :, None]
    rostral_plug = (row_has_bone[:, None, :] & (j_grid < j_first_smooth[:, None, :]))
    if verbose:
        print(f'  per-row rostral plug:        {int(rostral_plug.sum())} voxels'
              f'{" (smoothed)" if smooth_size > 1 else ""}')

    # ---- Per-row caudal plug (mirror) ----
    j_last = nj_ - 1 - np.argmax(closed_bone[:, ::-1, :], axis=1)
    j_last = np.where(row_has_bone, j_last, -1)
    j_last_smooth = _smooth2d(j_last, smooth_size)
    caudal_plug = (row_has_bone[:, None, :] & (j_grid > j_last_smooth[:, None, :]))
    if verbose:
        print(f'  per-row caudal plug:         {int(caudal_plug.sum())} voxels'
              f'{" (smoothed)" if smooth_size > 1 else ""}')

    floor = floor | rostral_plug | caudal_plug

    if use_hull:
        from scipy.spatial import ConvexHull
        from matplotlib.path import Path as _Path
        ni_, nj_, nk_ = arr.shape
        i_x_pos = int(round((+x_hull_half_mm - aff[0, 3]) / aff[0, 0]))
        i_x_neg = int(round((-x_hull_half_mm - aff[0, 3]) / aff[0, 0]))
        i_lo, i_hi = sorted([max(0, i_x_pos), min(ni_, i_x_neg)])
        k_z_min = int(round((z_hull_min_mm - aff[2, 3]) / aff[2, 2]))
        k_z_min = max(0, min(nk_, k_z_min))

        def _slice_hull(slice_2d, axis_lo_offsets, out_shape):
            pts = np.argwhere(slice_2d)
            if len(pts) < 3:
                return None
            pts[:, 0] += axis_lo_offsets[0]
            pts[:, 1] += axis_lo_offsets[1]
            try:
                hull = ConvexHull(pts)
            except Exception:
                return None
            ga, gb = np.meshgrid(np.arange(out_shape[0]),
                                  np.arange(out_shape[1]), indexing='ij')
            coords = np.column_stack([ga.ravel(), gb.ravel()])
            return _Path(pts[hull.vertices]).contains_points(
                coords).reshape(out_shape)

        hull_axial = np.zeros_like(closed_bone)
        j_hi_lowz = int(round((y_hull_max_lowz_mm - aff[1, 3]) / aff[1, 1]))
        j_hi_highz = int(round((y_hull_max_highz_mm - aff[1, 3]) / aff[1, 1]))
        j_hi_lowz = max(0, min(nj_, j_hi_lowz))
        j_hi_highz = max(0, min(nj_, j_hi_highz))
        k_break = int(round((z_ycutoff_break_mm - aff[2, 3]) / aff[2, 2]))
        k_break = max(0, min(nk_, k_break))
        for k in range(k_z_min, nk_):
            j_lo_k = j_hi_lowz if k < k_break else j_hi_highz
            sl = closed_bone[i_lo:i_hi, j_lo_k:, k]
            if sl.sum() < 10:
                continue
            inside = _slice_hull(sl, (i_lo, j_lo_k), (ni_, nj_))
            if inside is None:
                continue
            hull_axial[:, :, k] = inside
        outside_hull = ~hull_axial
        shell = closed_bone | seal | floor | outside_hull
        if verbose:
            print(f'  hull: i=[{i_lo},{i_hi}) x={x_hull_half_mm:+}mm '
                  f'j>={j_lo} y<={y_hull_max_lowz_mm}/'
                  f'{y_hull_max_highz_mm}mm '
                  f'k>={k_z_min} z>={z_hull_min_mm}mm  '
                  f'shell={shell.mean():.4f}')
    else:
        outside_hull = np.zeros_like(closed_bone)
        shell = closed_bone | seal | floor
        if verbose:
            print(f'  shell (no hull): {shell.mean():.4f}')

    info['shell_fraction'] = float(shell.mean())

    # Region-grow outside from volume boundary.
    not_shell = ~shell
    out_seed = np.zeros_like(not_shell)
    out_seed[0,:,:] = True;  out_seed[-1,:,:] = True
    out_seed[:,0,:] = True;  out_seed[:,-1,:] = True
    out_seed[:,:,0] = True;  out_seed[:,:,-1] = True
    out_seed &= not_shell
    labeled_ns, _ = _label(not_shell)
    edge_ids = np.unique(labeled_ns[out_seed])
    edge_ids = edge_ids[edge_ids > 0]
    outside = np.isin(labeled_ns, edge_ids)
    cavity_raw = not_shell & ~outside
    info['cavity_raw_mm3'] = float(cavity_raw.sum()) * voxel_mm**3
    if verbose:
        print(f'  raw cavity: {info["cavity_raw_mm3"]:.1f} mm^3')

    # Largest CC.
    labeled, n_cc = _label(cavity_raw)
    sizes = np.bincount(labeled.ravel()); sizes[0] = 0
    if len(sizes) <= 1:
        if verbose:
            print('  no cavity CCs found')
        info['cavity_mm3'] = 0.0
        return cavity_raw, info
    main = int(np.argmax(sizes))
    cavity = (labeled == main)
    info['cavity_mm3_pre_propagation'] = float(cavity.sum()) * voxel_mm**3
    if verbose:
        print(f'  largest pre-fill CC: '
              f'{info["cavity_mm3_pre_propagation"]:.1f} mm^3')

    # Geodesic propagation into bone-bounded pockets.
    if use_geodesic_propagation:
        allowed = (~bone) & (~outside) & (~outside_hull) & (~floor) & (~seal)
        cavity = _bp(cavity, mask=allowed)
        if verbose:
            print(f'  after geodesic propagation: '
                  f'{cavity.sum() * voxel_mm**3:.1f} mm^3')

    info['cavity_mm3'] = float(cavity.sum()) * voxel_mm**3
    info['n_cc'] = int(n_cc)
    return cavity, info


def cavity_centroid_world_mm(cavity_path):
    """Return the world-mm centroid of a saved cavity NIfTI, or
    [0, 0, 0] if the file is missing/empty."""
    import os
    import nibabel as nib
    if not os.path.exists(cavity_path):
        return np.array([0.0, 0.0, 0.0])
    img = nib.load(cavity_path)
    arr = np.asarray(img.dataobj)
    aff = img.affine
    ijk = np.argwhere(arr > 0.5)
    if len(ijk) == 0:
        return np.array([0.0, 0.0, 0.0])
    return (aff[:3, :3] @ ijk.mean(axis=0)) + aff[:3, 3]
