"""Macaque (AMNH M-87264 dry-skull microCT + NMT v2 atlas) species
binding for TUBA.

Reference: Pinton, *A 500 kHz transcranial focused-ultrasound pipeline
for the rhesus macaque dorsal anterior cingulate cortex*, in
``tuba/docs/macaque/manuscript.tex`` (sections cited below as §N).

Scan (manuscript §1, Table 1)
-----------------------------
* Source: AMNH:Mammals:M-87264 (adult Macaca mulatta), Delson Primate
  Scans collection, MorphoSource media id 18083 (DICOM) / 18084
  (TIFF). General Electric phoenix v|tome|x microCT at the AMNH
  Microscopy and Imaging Facility.
* 2148 axial TIFFs of 1548 x 1359 uint16 at 60.61 um isotropic;
  ~9 GB uint16 stack covering 93.8 x 82.4 x 130.2 mm.
* Dry skull + articulated mandible (not anatomically attached);
  ``tuba._macaque.isolate_cranium`` removes the mandible CC before
  cavity extraction.
* License: CC BY-NC (no commercial use), MorphoSource standard.

Wavelength criterion (manuscript §1)
------------------------------------
At 500 kHz, cortical-bone wavelength is ~5.8 mm (c_bone = 2900 m/s).
The 60.6 um native cache samples ~96 voxels per wavelength; even
after factor-4 block-max to the 242 um working cache, ~24 voxels
per wavelength remains -- comparable to the 28 voxels/lambda the
mouse 15 MHz pipeline (Maga 4K) achieves.

Atlas (manuscript §2)
---------------------
NIMH Macaque Template v2.0-sym (NMT v2), population-averaged T1 from
31 adult rhesus macaque brains. Ships at 0.25 mm isotropic bundled
with the CHARM (cortical hierarchy, 6 levels, finest = 134 labels),
SARM (subcortical hierarchy, 6 levels, finest = ~205 labels), and
D99 v2 (combined cortical+subcortical, 197 labels) parcellations.
See :mod:`tuba.data.fetch_nmt` for the staging script.

Pipeline stages (manuscript §3-8)
---------------------------------
* §3  Orientation determination from per-slice bone-mass profile +
      log-binned intensity histogram + mid-stack orthoslice renders.
      ``SLICE_FLIP=False``, ``COL_FLIP=False``, ``ROW_FLIP=False``;
      mapping cranial/mandibular bone arrangement disambiguated by
      direct slice renders.
* §4  Streaming block-max downsampling from the native 60.6 um cache
      to the 242 um (factor 4) working cache. Mandible removal via
      :mod:`tuba._macaque.isolate_cranium`: keep largest bone CC after
      1 mm closing, mask outside its bbox + 5 mm margin.
* §5  Constrained PCA pre-alignment on a cranium-only point cloud
      (mandible removed first to avoid AP-axis bias). The two smaller
      PCA eigenvectors are degenerate; D-V axis is fixed by a
      basicranium-vs-vault asymmetry sweep.
* §6  Endocranial cavity extraction at 242 um (
      :mod:`tuba._macaque.fast_iter_250um`): bone-shell intermediate
      via per-axial hull + curved basicranium floor + rostral/caudal
      plugs (~92.8 mL), then ANTs Affine fit to the NMT brain mask,
      then inverse-warp the NMT mask back into macaque space (~88.4
      mL after 2 mm closing + bone clip). Native upsample via
      :mod:`tuba._macaque.native_cavity` (NN affine_transform).
* §7  NMT-template + CHARM 5-D + SARM 5-D + D99 atlases pushed back
      into macaque space using the same Affine matrix
      (:mod:`tuba._macaque.warp_atlases_via_affine`).
* §8  Acoustic-property mapping: piecewise-linear intensity to
      (c, rho) ramp anchored on cortical-bone histogram mode
      (I_high=32000 -> c=2900 m/s, rho=1900 kg/m^3); cavity infill
      with parenchymal (1540, 1040). Forward simulation via
      angular-spectrum at 880 um lateral pitch + 3-cycle 500 kHz
      tone burst, MI=1.0 drive calibration.

Environment variables
---------------------
* ``TUBA_MACAQUE_REG_DIR`` -- output cache directory (default
  ``~/.cache/tuba/macaque/registration``).
* ``TUBA_MACAQUE_SOURCE_DIR`` -- AMNH M-87264 raw TIFF directory.
* ``TUBA_MACAQUE_TIFF_GLOB`` -- override the per-slice TIFF glob.
* ``NMT_DEST`` -- NMT v2 atlas root (mirrors
  :mod:`tuba.data.fetch_nmt`).

Public API
----------
* :data:`NATIVE_VOXEL_MM`, :data:`DOWNSAMPLE_VOXEL_MM`
* :data:`BONE_LOW`, :data:`BONE_HIGH`, :data:`ALIGN_PCA_THRESH`
* :data:`ALIGN_RX_DEG`, :data:`ALIGN_RY_DEG`, :data:`ALIGN_RZ_DEG`
* :data:`NATIVE_PAD_VOXELS` (= 400; manuscript §5 fix)
* :func:`place_on_skull` -- thin wrapper over
  :func:`tuba._macaque.macaque_placement.place_on_macaque_skull` with
  the Maga / Halle signature.
* :func:`load_slab` -- thin wrapper over
  :func:`tuba._macaque.macaque_skull_slab_3d.load_macaque_skull_slab_3d`.
"""
from __future__ import annotations

from .._macaque import macaque_skull_constants as _c
from .._macaque import macaque_placement as _placement
from .._macaque import macaque_skull_slab_3d as _slab

# ---------------------------------------------------------------------------
# Re-export module-level constants that downstream consumers may want.
# Paths --------------------------------------------------------------------
REG_DIR                       = _c.REG_DIR
SOURCE_DIR                    = _c.SOURCE_DIR
TIFF_GLOB                     = _c.TIFF_GLOB
NMT_DEST                      = _c.NMT_DEST
NMT_DIR                       = _c.NMT_DIR

MACAQUE_RAS_NIFTI             = _c.MACAQUE_RAS_NIFTI
MACAQUE_RAS_NIFTI_ALIGNED     = _c.MACAQUE_RAS_NIFTI_ALIGNED
MACAQUE_RAS_NATIVE_ALIGNED    = _c.MACAQUE_RAS_NATIVE_ALIGNED
MACAQUE_RAS_ALIGNED_250UM     = _c.MACAQUE_RAS_ALIGNED_250UM

# Physical params -----------------------------------------------------------
NATIVE_VOXEL_MM     = _c.NATIVE_VOXEL_MM
DOWNSAMPLE_VOXEL_MM = 0.24245           # factor-4 block-max from native

# Intensity thresholds ------------------------------------------------------
# AMNH M-87264 microCT histogram (manuscript §3, Fig.~\ref{fig:histogram}):
# air <100, mounting/soft-tissue peak ~5k, cortical bone 10k-45k with
# trabecular mode ~32k and dental-enamel tail at 40-44k. A single
# threshold cleanly separates bone from mounting material -- different
# convention from the mouse (BONE_LOW/BONE_HIGH split) because the
# macaque histogram is simpler.
BONE_THRESH      = _c.BONE_THRESH
ALIGN_PCA_THRESH = _c.ALIGN_PCA_THRESH

# Alignment Euler angles (intrinsic ZYX). Derived from the constrained-PCA
# recipe in :func:`tuba._macaque.find_cranium_alignment.find_cranium_alignment`.
ALIGN_RX_DEG = _c.ALIGN_RX_DEG
ALIGN_RY_DEG = _c.ALIGN_RY_DEG
ALIGN_RZ_DEG = _c.ALIGN_RZ_DEG

# Output padding (manuscript §5). 400 vox at 60.6 um = 24 mm cushion;
# gives refine_native.py's -27 mm sagittal translation enough source-FOV
# headroom that the inferior basicranium + forehead survive without
# cval=0 truncation (the bug fixed in 2026-05-19).
NATIVE_PAD_VOXELS = 400

# ---------------------------------------------------------------------------
# Public callables. Thin wrappers to keep the signatures aligned with the
# species API used by mouse + human modules; they delegate to the legacy
# implementations under :mod:`tuba._macaque`.
# ---------------------------------------------------------------------------
def place_on_skull(target_name='dACC',
                    focal_length_mm=45.0,
                    scalp_clearance_mm=None,
                    beam_tilt_deg_yz=0.0,
                    lateral_search_mm=10.0,
                    verbose=False):
    """Compute apex / target / beam for a single-element bowl aimed at
    a CHARM-defined macaque target.

    Defaults match the validation report bowl: f0 = 500 kHz, D = 45 mm,
    ROC = 45 mm (f/1). ``focal_length_mm`` is the geometric focal
    distance from the bowl apex to the target -- equals the ROC for a
    geometric-focus bowl.

    Parameters
    ----------
    target_name : {'dACC', 'dACC_broad', 'cavity_centre'}, default 'dACC'
    focal_length_mm : float, default 45.0
        Apex-to-target Euclidean distance. ``45 mm`` matches the
        bowl's geometric focus.
    scalp_clearance_mm : float, optional
        Water-column distance above the outer bone surface. Ignored
        when ``focal_length_mm`` is set.
    beam_tilt_deg_yz : float, default 0.0
        Sagittal-plane tilt about the cavity centroid.
    lateral_search_mm : float, default 10.0
        Radius around the target lateral position within which to pick
        the highest-z outer-bone surface voxel as apex candidate.
    verbose : bool

    Returns
    -------
    placement : dict
        Same shape as :func:`tuba._macaque.macaque_placement.place_on_macaque_skull`:
        ``target_lps``, ``xdc_center_lps``, ``focus_lps``, ``beam_dir_3d``,
        ``scalp_clearance_mm``, ``apex_on_bone_voxel_mm``, etc.
    """
    return _placement.place_on_macaque_skull(
        target_name=target_name,
        apex_to_target_mm=focal_length_mm,
        scalp_clearance_mm=scalp_clearance_mm,
        beam_tilt_deg_yz=beam_tilt_deg_yz,
        lateral_search_mm=lateral_search_mm,
        debug=verbose,
    )


def load_slab(apex_world_mm, beam_3d, slab_size_m, dx_target_m,
               cache_path=None, cavity_mask_path=None, chatter=False):
    """Sample a beam-aligned (c, rho) slab from the macaque skull at
    ``apex_world_mm``. Same signature as :func:`tuba.species.mouse.load_slab`
    and :func:`tuba.species.human.load_slab`.

    Parameters
    ----------
    apex_world_mm : array-like (3,)
        Slab origin in macaque aligned-native world-mm (origin at the
        skull bone centroid; axes RAS after the constrained-PCA
        rotation).
    beam_3d : array-like (3,)
        Unit beam direction in the same world frame.
    slab_size_m : tuple (lateral, elevation, depth) in metres.
    dx_target_m : float
        Target voxel pitch in metres for the slab.
    cache_path, cavity_mask_path : str, optional
        Override the default 242 um intensity cache / cavity mask
        paths if a non-default rebuild is being tested.
    chatter : bool

    Returns
    -------
    c_map, rho_map : (n_lat, n_elev, n_depth) float32
    dz_slab : float (m)
    frame : dict with 'beam', 't1', 't2', 'apex_mm'.
    """
    kw = {}
    if cache_path is not None:
        kw['cache_path'] = cache_path
    if cavity_mask_path is not None:
        kw['cavity_mask_path'] = cavity_mask_path
    return _slab.load_macaque_skull_slab_3d(
        apex_world_mm, beam_3d, slab_size_m, dx_target_m,
        chatter=chatter, **kw)
