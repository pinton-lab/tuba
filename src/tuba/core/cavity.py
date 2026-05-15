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
