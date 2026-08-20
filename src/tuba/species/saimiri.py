"""Squirrel monkey (Saimiri sciureus museum microCT + VALiDATe29 atlas)
species binding -- NIH R01 EB037345, Aim 3.

Geometry-only pillar. Where the mouse/rat/macaque/human pillars each
register a subject skull to a species brain atlas and then map intensity
to acoustic properties, this pillar is bootstrapped on an **uncalibrated
museum microCT** (squirrel monkey *Saimiri sp.*, NMNH USNM 194346,
MorphoSource media 000116521; see :mod:`tuba.data.fetch_saimiri`). It
therefore does the geometry --
skull segmentation, endocranial cavity, VALiDATe29 atlas registration,
somatosensory/motor target export -- but the quantitative CT->acoustics
step (:class:`tuba.core.hu_acoustics.AubryHUMapping`) **refuses to run**
on the uncalibrated scan, and the slab loader falls back to an
explicitly-flagged placeholder ramp. Swap in an HU-calibrated colony CT
(set :data:`CT_CALIBRATION`) to unlock quantitative acoustics.

Scope: deliverables D1 (CT->acoustic model + PPW + one grid-convergence
run) and D2 (VALiDATe29 registration + S1/M1 target export + QC). The
downstream wave physics -- time-reversal aberration correction,
multifocal synthesis, the two-foci feasibility sweep, and MI/bioheat
safety -- lives in the fullwave2 solver siblings and consumes the
(c, rho, dz) slab this module exports.

Study targets (from the funded proposal)
----------------------------------------
Hand representations in S1 areas 3b and 1 and in M1 (area 4). S1 and M1
are only ~2-3 mm apart in the squirrel monkey; whether two independently
controllable foci can exist at that separation is the feasibility
question the solver pillar must answer. Neuromod operating point:
2 MHz (design range 2-4 MHz), 400 kPa peak-negative pressure at target.

Coordinate frame
----------------
Subject aligned-native RAS, bone-centroid origin -- the same canonical
world frame the other pillars use, so a solver/placement driver wired
against one species runs on this one unchanged.

Environment variables
---------------------
* ``TUBA_SAIMIRI_REG_DIR``   -- registration/cache output (default
  ``~/.cache/tuba/saimiri/registration``).
* ``SAIMIRI_SOURCE_DIR``     -- raw museum microCT TIFF stack.
* ``VALIDATE29_DEST``        -- VALiDATe29 atlas root (mirrors
  :mod:`tuba.data.fetch_validate29`).

Calibration state
-----------------
The orientation, intensity thresholds, and cavity hull geometry below
are CALIBRATED against the staged scan (USNM 194346): the rat/macaque
museum-CT orientation convention transfers correctly (confirmed clean
RAS by an orthoslice probe), the bone/cavity thresholds are set from the
histogram, and the hull envelope + watertight-shell threshold give a
solid ~15.9 mL endocranial cavity. The cavity_binary SyN to the
VALiDATe29 brain mask fits to Dice 0.96, and the warped S1/M1 targets
are anatomically correct (M1 rostral+dorsal to S1). Only the acoustics
stay guarded (no HU-calibrated colony CT). See docs/saimiri/manuscript.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from tuba.atlases.validate29 import VALiDATe29
from tuba.core import (align, cavity as cav, downsample, frame, hu_acoustics,
                       placement as plc, slab, surface, warp)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REG_DIR = os.environ.get(
    'TUBA_SAIMIRI_REG_DIR',
    os.path.expanduser('~/.cache/tuba/saimiri/registration'))

SOURCE_DIR = os.environ.get(
    'SAIMIRI_SOURCE_DIR',
    os.path.expanduser('~/.cache/tuba/saimiri/source'))

# MorphoSource/UTCT media 000116521 (USNM 194346) unpacks to a 16-bit
# TIFF stack under 16bit/; glob is permissive because the museum archive
# layout is not fixed until it is staged.
TIFF_GLOB = os.path.join(SOURCE_DIR, '**', '*.tif*')

# Cached intermediates (subject aligned-native RAS frame, isotropised to
# WORKING_VOXEL_MM). The staged scan is anisotropic; see downsample_to_working.
SAIMIRI_RAS_NIFTI         = os.path.join(REG_DIR, 'saimiri_skull_98um.nii.gz')
SAIMIRI_RAS_ALIGNED       = os.path.join(REG_DIR, 'saimiri_skull_aligned_98um.nii.gz')
SAIMIRI_RAS_WORKING       = SAIMIRI_RAS_ALIGNED  # slab/surface source
CAVITY_NII                = os.path.join(REG_DIR, 'saimiri_cranial_cavity.nii.gz')
CAVITY_QC                 = os.path.join(REG_DIR, 'saimiri_cavity_qc.png')
TEMPLATE_IN_SUBJECT       = os.path.join(REG_DIR, 'validate29_template_in_saimiri.nii.gz')
ANNOT_NII                 = os.path.join(REG_DIR, 'validate29_labels_in_saimiri.nii.gz')
BRAIN_IN_SUBJECT          = os.path.join(REG_DIR, 'validate29_brainmask_in_saimiri.nii.gz')
TARGETS_JSON              = os.path.join(REG_DIR, 'saimiri_targets.json')
ATLAS_QC_PNG              = os.path.join(REG_DIR, 'saimiri_atlas_qc.png')

# VALiDATe29 atlas binding (resolves template/labels/LUT by keyword glob).
ATLAS = VALiDATe29()

SYN_PREFIX = 'saimiri_cavity_to_validate29'

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
# Staged scan is ANISOTROPIC: 0.0977 mm in-plane (x, y) x 0.1189 mm slice
# (z), per the MorphoSource media 000116521 manifest (UTCT ACTIS recon of
# USNM 194346). The shared cavity/align/slab tools assume isotropic voxels
# (rat/mouse museum scans are), so downsample_to_working isotropises the
# stack to WORKING_VOXEL_MM by upsampling only the coarse slice axis
# (order-1, thin-bone-preserving) -- keeping the fine in-plane resolution.
NATIVE_INPLANE_MM = 0.0977    # media 000116521 x/y pixel spacing
NATIVE_SLICE_MM = 0.1189      # media 000116521 z spacing (slice thickness)
NATIVE_VOXEL_MM = NATIVE_INPLANE_MM   # reformat pitch (in-plane, factor 1)
WORKING_VOXEL_MM = NATIVE_INPLANE_MM  # isotropic working pitch (~98 um)

# Storage-axis convention (CONFIRMED on USNM 194346, media 000116521).
# The DigiMorph/UTCT stack follows the same (slice=AP, row=DV, col=LR)
# convention as the rat pillar; an orientation probe (mid-stack
# orthoslices, saimiri_orient_probe) confirms the resulting frame is
# clean RAS: anterior (snout/orbits) at +y, dorsal (braincase dome) at
# +z, bilaterally symmetric about x, right-handed. True L/R handedness is
# not recoverable from a bare museum skull (no fiducial); +x is taken as
# Right per RAS and the L/R target labels carry that caveat.
SLICE_FLIP = False
ROW_FLIP = False
COL_FLIP = True
SWAP_ROW_COL = True
AXIS_FLIP_KWARGS = {'slice_flip': SLICE_FLIP, 'row_flip': ROW_FLIP,
                    'col_flip': COL_FLIP, 'swap_row_col': SWAP_ROW_COL}
VOXEL_SIGNS = (-1, -1, +1)    # storage (i=R, j=A, k=V) -> RAS world

# Intensity thresholds (CONFIRMED from the staged-scan histogram, 16-bit
# uncalibrated counts). Air/mount+soft-tissue mode is <=~5000 (p90=5267);
# the bone knee is at ~6000 (voxel fraction 13.0% ->7.5% across 5-6k),
# cortical bone runs to ~40k (p99.9=43k). See saimiri_orient_probe.
BONE_LOW = 6000.0            # shell-seed floor + placeholder-ramp i_low
BONE_HIGH = 18000.0          # cavity-exclusion ceiling
BONE_RAMP_HIGH = 35000.0     # placeholder-ramp cortical anchor (i_high)

# Pre-alignment rotation (PROVISIONAL). Museum stacks are near axis-aligned
# after the swap+transpose; a PCA probe on the staged bone cloud fixes any
# residual tilt (cf. rat pillar, which lands at 0 deg).
ALIGN_RX_DEG = 0.0
ALIGN_RY_DEG = 0.0
ALIGN_RZ_DEG = 0.0
NATIVE_PAD_VOXELS = 200

# Cavity extraction, using the shared macaque/rat bone-shell recipe
# (tuba.core.cavity.bone_shell_cavity). Same pattern as every pillar: a
# solid, correctly-shaped endocranial "seed" is the moving image for the
# atlas-brain-mask registration; it need not be a perfect segmentation,
# only solid and anatomically bounded.
#
# CAVITY_BONE_LOW (5000) is a WATERTIGHT-SHELL threshold, set *below* the
# BONE_LOW air/bone knee (6000). Like the rat (whose 8-bit threshold-50
# base is naturally watertight), the squirrel needs the thin midline
# basicranium captured so the per-column floor seals it: at 6000 the
# midline base drops below threshold and "outside" floods up into the
# braincase centre (a hollow-centred cavity); at 5000 the floor is
# watertight and the cavity is solid (verified: central-fill 0.97 vs
# 0.26). The slightly thicker shell makes the volume a conservative,
# inner-table-inclusive estimate. Confirmed on saimiri_cavity_solid.
CAVITY_BONE_LOW = 5000.0
SEED_CLOSE_MM = 1.5
PLUG_SMOOTH_MM = 1.5
# Generous braincase envelope (squirrel scale); confirmed non-clipping on
# the staged scan (cavity extent x +-18, y [-27,+16], z [-10,+20] mm all
# sit inside these planes). Rostral cutoff is tighter at the basicranium
# (low z) than at the vault (high z), as in the rat/macaque pillars.
HULL_X_HALF_MM = 24.0
HULL_Y_MAX_LOWZ_MM = +8.0
HULL_Y_MAX_HIGHZ_MM = +20.0
HULL_Z_MIN_MM = -16.0
HULL_Z_YCUTOFF_BREAK_MM = +1.0
# Endocranial-cavity QC bracket. The solid extraction lands ~16 mL
# (conservative, inner-table-inclusive); Saimiri cranial capacity in the
# comparative literature spans ~16-26 mL across individuals/species. The
# cavity is the registration moving image, so the affine to the 33 mL
# VALiDATe29 brain mask absorbs the individual scale difference.
CAVITY_TARGET_ML = (13.0, 22.0)

# ---------------------------------------------------------------------------
# Acoustic model (deliverable 1) -- calibration state + mappings.
# ---------------------------------------------------------------------------
# The staged scan is an uncalibrated museum microCT. Replace with
#   CT_CALIBRATION = hu_acoustics.CTCalibration(kind='hu', hu_bone_max=<scan>)
# when binding an HU-calibrated colony CT.
CT_CALIBRATION = hu_acoustics.UNCALIBRATED_MICROCT

# Quantitative HU->(rho, c, alpha) mapping (Aubry 2003). GUARDED: calling
# this on the uncalibrated museum scan raises UncalibratedInputError.
HU_MAPPING = hu_acoustics.AubryHUMapping(calibration=CT_CALIBRATION)

# Sanctioned geometry-only fallback: an explicitly-flagged placeholder
# ramp (carries .placeholder=True) mapping uncalibrated intensity to
# nominal cortical-bone endpoints so the skull shell + beam path can be
# sampled/visualized without claiming calibrated acoustics.
SLAB_RAMP = hu_acoustics.placeholder_ramp(
    i_low=BONE_LOW, i_high=BONE_RAMP_HIGH,
    reason='Saimiri museum microCT (uncalibrated; geometry only)')
SLAB_CAVITY_INFILL = slab.CavityInfill(c_parenchyma=1540.0, rho_parenchyma=1040.0)

# Attenuation (literature; applied by the solver). Cortical bone /
# soft-tissue power-law coefficients, dB/cm/MHz.
ALPHA_BONE_DB_PER_CM_PER_MHZ = 8.0
ALPHA_TISSUE_DB_PER_CM_PER_MHZ = 0.5

# Simulation grid + operating band. The neuromod band is 2-4 MHz; the
# FDTD grid pitch must resolve the shortest (water) wavelength at 4 MHz.
SIM_FREQS_HZ = (2.0e6, 3.0e6, 4.0e6)
SIM_DX_M = 60.0e-6            # 6.25 PPW @ 4 MHz in water; reported below
PPW_TARGET = 6.0

# Beam-aligned slab (SI). Small NHP head: 20x20 mm footprint, 40 mm along
# the beam. Demo/QC pitch is coarser than SIM_DX_M for speed; the PPW +
# grid-convergence report use SIM_DX_M.
SLAB_SIZE_M = (0.02, 0.02, 0.04)
SLAB_DX_M = 150.0e-6


# ---------------------------------------------------------------------------
# Staging guards
# ---------------------------------------------------------------------------
def _tiff_paths():
    paths = sorted(glob.glob(TIFF_GLOB, recursive=True))
    if not paths:
        raise FileNotFoundError(
            f'No TIFFs match {TIFF_GLOB}. '
            f'Run `python -m tuba.data.fetch_saimiri` and stage the scan first.')
    return paths


def _assert_atlas_staged():
    # Touch a resolved path so a missing atlas raises the fetcher hint early.
    _ = ATLAS.annotation_path
    return ATLAS


# ---------------------------------------------------------------------------
# Stage 1-2: ingest + align (dry-skull museum-CT template, cf. rat pillar)
# ---------------------------------------------------------------------------
def downsample_to_working(force=False, verbose=True):
    """Reformat the native TIFF stack into an aligned-axes RAS NIfTI at
    the working (~98 um) *isotropic* pitch.

    The staged scan is anisotropic (0.0977 mm in-plane, 0.1189 mm slice).
    We reuse the shared TIFF streamer at the in-plane pitch (factor 1,
    no in-plane decimation), which -- after the fixed
    ``apply_axis_flips`` transpose -- puts the coarse slice axis at
    output axis 1 (world AP). A single order-1 zoom along that axis
    upsamples it from 0.1189 to 0.0977 mm, yielding a truly isotropic
    volume so the downstream isotropic-voxel tools (cavity, align, slab)
    are exact. Upsampling (never downsampling) the slice axis cannot
    erode the thin cortical-bone shell.
    """
    import nibabel as nib
    import scipy.ndimage as sn

    os.makedirs(REG_DIR, exist_ok=True)
    if os.path.exists(SAIMIRI_RAS_NIFTI) and not force:
        if verbose:
            print(f'  using cached {SAIMIRI_RAS_NIFTI}')
        return SAIMIRI_RAS_NIFTI

    raw_path = os.path.join(REG_DIR, 'saimiri_skull_inplane_raw.nii.gz')
    downsample.downsample_tiff_stack_to_nifti(
        _tiff_paths(),
        target_voxel_mm=NATIVE_INPLANE_MM,
        native_voxel_mm=NATIVE_INPLANE_MM,
        out_path=raw_path,
        axis_flip_kwargs=AXIS_FLIP_KWARGS,
        voxel_signs=VOXEL_SIGNS,
        force=force, verbose=verbose,
    )

    # Isotropise: the slice axis (output axis 1) is true 0.1189 mm but the
    # reformat labelled it 0.0977; zoom it up so the physical spacing
    # matches the label on every axis.
    img = nib.load(raw_path)
    vol = np.asarray(img.dataobj).astype(np.float32)
    zoom_j = NATIVE_SLICE_MM / NATIVE_INPLANE_MM     # 1.217x along AP
    iso = sn.zoom(vol, (1.0, zoom_j, 1.0), order=1)
    affine = frame.affine_for_shape(iso.shape, WORKING_VOXEL_MM, signs=VOXEL_SIGNS)
    nib.save(nib.Nifti1Image(iso.astype(np.float32), affine), SAIMIRI_RAS_NIFTI)
    if verbose:
        print(f'  isotropised slice axis {vol.shape} -> {iso.shape} '
              f'@ {WORKING_VOXEL_MM*1e3:.1f} um iso -> {SAIMIRI_RAS_NIFTI}')
    return SAIMIRI_RAS_NIFTI


def align_skull_to_axes(force=False, verbose=True):
    """Rotate the working cache about the bone centroid (PROVISIONAL
    angles; 0 deg by default -- confirm with a PCA probe on the scan)."""
    return align.align_nifti_to_axes(
        in_path=SAIMIRI_RAS_NIFTI, out_path=SAIMIRI_RAS_ALIGNED,
        ax_deg=ALIGN_RX_DEG, ay_deg=ALIGN_RY_DEG, az_deg=ALIGN_RZ_DEG,
        bone_thresh=BONE_LOW, pad_voxels=12,
        voxel_signs=VOXEL_SIGNS, force=force, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Stage 3: endocranial cavity (bone-shell recipe shared in tuba.core.cavity)
# ---------------------------------------------------------------------------
def extract_cavity(force=False, verbose=True):
    """Extract the squirrel-monkey endocranial cavity as the (solid)
    moving image for the cavity_binary SyN -- the same shared bone-shell
    recipe every pillar uses (per-column basicranium floor + per-row
    rostral/caudal plugs + per-axial convex hull) -> fill-holes ->
    largest-CC. Uses CAVITY_BONE_LOW (watertight-shell threshold) so the
    midline basicranium seals and the cavity is solid. Writes
    ``saimiri_cranial_cavity.nii.gz`` and returns its path."""
    import nibabel as nib
    import scipy.ndimage as sn

    if os.path.exists(CAVITY_NII) and not force:
        if verbose:
            print(f'  using cached {CAVITY_NII}')
        return CAVITY_NII

    if not os.path.exists(SAIMIRI_RAS_ALIGNED):
        align_skull_to_axes(verbose=verbose)
    img = nib.load(SAIMIRI_RAS_ALIGNED)
    arr = img.get_fdata().astype(np.float32)
    aff = img.affine
    voxel_mm = float(abs(aff[0, 0]))
    if verbose:
        print(f'\nExtract Saimiri cavity at {voxel_mm*1000:.1f} um, '
              f'shape={arr.shape}')

    # Bone-shell hull constants are expressed relative to a bone-centroid
    # origin; pass a centred affine (cf. rat pillar).
    bone_mask = arr > BONE_LOW
    if not bone_mask.any():
        raise RuntimeError(
            f'No voxels above BONE_LOW={BONE_LOW} in {SAIMIRI_RAS_ALIGNED} '
            f'(intensity range [{arr.min():.0f}, {arr.max():.0f}]); the scan '
            f'may be mis-staged or differently scaled than USNM 194346.')
    bone_centroid_world = aff[:3, :3] @ np.argwhere(bone_mask).mean(0) + aff[:3, 3]
    aff_centred = aff.copy()
    aff_centred[:3, 3] -= bone_centroid_world

    seed, info = cav.bone_shell_cavity(
        arr, aff_centred, bone_low=CAVITY_BONE_LOW,
        close_mm=SEED_CLOSE_MM,
        x_hull_half_mm=HULL_X_HALF_MM,
        y_hull_max_lowz_mm=HULL_Y_MAX_LOWZ_MM,
        y_hull_max_highz_mm=HULL_Y_MAX_HIGHZ_MM,
        z_hull_min_mm=HULL_Z_MIN_MM,
        z_ycutoff_break_mm=HULL_Z_YCUTOFF_BREAK_MM,
        use_hull=True, use_geodesic_propagation=False,
        smooth_plug_mm=PLUG_SMOOTH_MM, verbose=verbose)

    cavm = sn.binary_fill_holes(seed)
    lab, n = sn.label(cavm)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        cavm = (lab == int(np.argmax(sizes)))
    vol_ml = cavm.sum() * voxel_mm ** 3 / 1000.0
    nib.save(nib.Nifti1Image(cavm.astype(np.uint8), aff), CAVITY_NII)
    if verbose:
        lo, hi = CAVITY_TARGET_ML
        flag = '' if lo <= vol_ml <= hi else '  <-- outside target; recheck hull/threshold'
        print(f'  seed cavity {info["cavity_mm3"]/1000:.2f} mL -> '
              f'production {vol_ml:.2f} mL (target {lo}-{hi} mL){flag}')
        print(f'  wrote {CAVITY_NII}')
    return CAVITY_NII


# ---------------------------------------------------------------------------
# Stage 4: VALiDATe29 registration (affine + deformable = cavity_binary SyN)
# ---------------------------------------------------------------------------
def register_to_atlas(force=False, verbose=True):
    """Register the subject cavity (moving) to the VALiDATe29 brain mask
    (fixed) with ANTs SyN (affine + deformable), then warp the atlas
    template + cortical labels + brain mask back into subject space.

    Writes the transforms (prefix :data:`SYN_PREFIX`) and
    ``validate29_{template,labels,brainmask}_in_saimiri.nii.gz``.
    """
    _assert_atlas_staged()
    if not os.path.exists(CAVITY_NII):
        extract_cavity(force=force, verbose=verbose)
    mask_path = ATLAS.derive_brain_mask(verbose=verbose)

    inv_txt = os.path.join(REG_DIR, f'{SYN_PREFIX}_syn_inv.txt')
    if force or not os.path.exists(inv_txt):
        if verbose:
            print('\n[register] cavity <-> VALiDATe29 brain mask '
                  '(SyN, affine + deformable)')
        warp.register_subject_to_atlas(
            CAVITY_NII, mask_path, REG_DIR, SYN_PREFIX,
            mode='cavity_binary', verbose=verbose)
    elif verbose:
        print(f'\n[register] using cached transforms ({inv_txt})')

    ref = SAIMIRI_RAS_ALIGNED
    if verbose:
        print('\n[warp] atlas brain mask + template + labels -> subject grid')
    warp.warp_atlas_into_subject(mask_path, ref, REG_DIR, SYN_PREFIX,
                                 BRAIN_IN_SUBJECT,
                                 interpolator='nearestNeighbor', verbose=verbose)
    warp.warp_atlas_into_subject(ATLAS.template_path, ref, REG_DIR, SYN_PREFIX,
                                 TEMPLATE_IN_SUBJECT,
                                 interpolator='linear', verbose=verbose)
    warp.warp_atlas_into_subject(ATLAS.annotation_path, ref, REG_DIR, SYN_PREFIX,
                                 ANNOT_NII,
                                 interpolator='nearestNeighbor', verbose=verbose)
    return {'template': TEMPLATE_IN_SUBJECT, 'labels': ANNOT_NII,
            'brain_mask': BRAIN_IN_SUBJECT, 'reg_dir': REG_DIR}


# ---------------------------------------------------------------------------
# Targets: S1 + M1 (deliverable 2)
# ---------------------------------------------------------------------------
# VALiDATe29's histological parcellation is region-level (confirmed
# against the staged LUT, 2026-08-18): the study's S1 hand areas 3b/1
# map to anterior_parietal_cortex (APC, which bundles areas 3a/3b/1/2)
# and M1 to primary_motor_cortex. Left/right are separate label ids.
TARGET_KEYS = ('S1', 'M1')


def _label_centroid_world_mm(label_id):
    """Centroid (subject RAS mm) + voxel count of a warped VALiDATe29 label
    id. Left/right are already separate ids in VALiDATe29, so the caller
    picks the hemisphere via :meth:`VALiDATe29.resolve_label`; no x-split."""
    import nibabel as nib
    img = nib.load(ANNOT_NII)
    arr = np.asarray(img.dataobj)
    aff = img.affine
    ijk = np.argwhere(arr == label_id)
    if len(ijk) == 0:
        raise RuntimeError(f'No voxels with label id {label_id} in {ANNOT_NII}')
    ras = (aff[:3, :3] @ ijk.T).T + aff[:3, 3]
    return ras.mean(0), int(len(ijk))


def target_world_mm(target_key, hemisphere='left'):
    """Subject-space RAS-mm centroid of a study target (``'S1'`` / ``'M1'``),
    resolved through the VALiDATe29 cortical labels warped into subject
    space. ``hemisphere`` selects the ``l_``/``r_`` label id.

    NOTE: this returns the whole-area centroid --- for S1 that is anterior
    parietal cortex (areas 3a/3b/1/2 bundled). Isolating the *hand*
    sub-representation (or splitting 3b from area 1) needs a stereotaxic
    prior on the homunculus, not derivable from the distributed label
    volume; wire it into ``export_targets(hand_prior=...)`` when the
    subject stereotaxic frame is fixed.
    """
    if not os.path.exists(ANNOT_NII):
        raise FileNotFoundError(
            f'{ANNOT_NII} missing; run register_to_atlas() first.')
    name, label_id = ATLAS.resolve_label(target_key, hemisphere=hemisphere)
    world, n = _label_centroid_world_mm(label_id)
    return {'target': target_key, 'atlas_label': name, 'label_id': label_id,
            'hemisphere': hemisphere, 'world_mm': tuple(float(v) for v in world),
            'n_vox': n}


def export_targets(hemisphere='left', verbose=True):
    """Export S1 (anterior parietal cortex) and M1 (primary motor cortex)
    target coordinates (subject RAS mm) to :data:`TARGETS_JSON`, alongside
    the transform prefix. Deliverable 2.
    """
    import json
    out = {'frame': 'saimiri_aligned_native_RAS',
           'transform_prefix': SYN_PREFIX,
           'reg_dir': REG_DIR, 'targets': {}}
    for key in TARGET_KEYS:
        t = target_world_mm(key, hemisphere=hemisphere)
        out['targets'][key] = t
        if verbose:
            print(f'    {key:9s} <- {t["atlas_label"]!r} (id {t["label_id"]}): '
                  f'{tuple(round(v,2) for v in t["world_mm"])} mm '
                  f'({t["n_vox"]} vox)')
    with open(TARGETS_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    if verbose:
        print(f'  wrote {TARGETS_JSON}')
    return out


# ---------------------------------------------------------------------------
# Placement + slab (geometry; feeds the solver pillar)
# ---------------------------------------------------------------------------
def place_on_skull(target_key='M1', hemisphere='left',
                   apex_to_target_mm=None, scalp_clearance_mm=None,
                   beam_tilt_deg_yz=0.0, lateral_search_mm=4.0, verbose=True):
    """Place a focused bowl on the squirrel-monkey skull aimed at a study
    target. Same return shape as the other pillars' ``place_on_skull``."""
    tgt = target_world_mm(target_key, hemisphere=hemisphere)
    surf = surface.outer_bone_surface_world_mm(
        skull_path=SAIMIRI_RAS_WORKING, bone_thresh=BONE_LOW,
        close_mm=1.5, head_thresh=BONE_HIGH, method='close+fill',
        verbose=verbose)
    cavity_centroid = cav.cavity_centroid_world_mm(CAVITY_NII)
    return plc.place_on_skull(
        target_world_mm=np.asarray(tgt['world_mm']),
        surface_world_mm=surf, cavity_centroid_world_mm=cavity_centroid,
        scalp_clearance_mm=scalp_clearance_mm, apex_to_target_mm=apex_to_target_mm,
        beam_tilt_deg_yz=beam_tilt_deg_yz, lateral_search_mm=lateral_search_mm,
        target_name=target_key, frame_name='saimiri_aligned_native_RAS',
        verbose=verbose)


def load_slab(apex_world_mm, beam_3d, slab_size_m=SLAB_SIZE_M,
              dx_target_m=SLAB_DX_M, cache_path=None, cavity_mask_path=None,
              chatter=True):
    """Extract a beam-aligned ``(c, rho)`` slab from the working cache.

    Uses the FLAGGED placeholder ramp (``SLAB_RAMP.placeholder is True``)
    because the museum scan is uncalibrated: geometry is faithful, the
    absolute (c, rho) values are nominal. Returns
    ``(c_map, rho_map, dz_slab, frame_dict)``."""
    return slab.load_slab(
        cache_path=cache_path or SAIMIRI_RAS_WORKING,
        apex_world_mm=apex_world_mm, beam_3d=beam_3d,
        slab_size_m=slab_size_m, dx_target_m=dx_target_m,
        ramp=SLAB_RAMP, cavity_mask_path=cavity_mask_path or CAVITY_NII,
        cavity_infill=SLAB_CAVITY_INFILL, verbose=chatter)


# ---------------------------------------------------------------------------
# Deliverable 1: acoustic-model report (PPW + one grid-convergence run)
# ---------------------------------------------------------------------------
def _central_ray_c_profile(apex_world_mm, beam_3d, dx_m, depth_m=None):
    """Sample sound speed along the central beam ray (a 1x1xN column) at
    pitch ``dx_m`` from the working cache via the slab sampler. Returns
    (c_profile[m/s], dx_m). Needs the staged scan."""
    depth_m = depth_m if depth_m is not None else SLAB_SIZE_M[2]
    c_map, _rho, dz, _fr = slab.load_slab(
        cache_path=SAIMIRI_RAS_WORKING, apex_world_mm=apex_world_mm,
        beam_3d=beam_3d, slab_size_m=(dx_m, dx_m, depth_m), dx_target_m=dx_m,
        ramp=SLAB_RAMP, cavity_mask_path=CAVITY_NII,
        cavity_infill=SLAB_CAVITY_INFILL, verbose=False)
    return c_map[0, 0, :].astype(np.float64), dz


def acoustic_model_report(placement=None, verbose=True):
    """Deliverable 1 report: calibration state, the PPW table at 2/3/4 MHz
    on the simulation grid, and -- if a placement + staged scan are
    available -- one grid-convergence run of the transcranial time-of-flight
    aberration (SIM_DX_M vs SIM_DX_M/2).

    Returns a dict; safe to call without data (PPW is pure math, the
    convergence run is skipped with a note if the scan is not staged).
    """
    if verbose:
        print('=== Saimiri acoustic-model report (deliverable 1) ===')
        cal = 'HU-calibrated' if CT_CALIBRATION.is_calibrated else 'UNCALIBRATED'
        print(f'  CT calibration: {cal} ({CT_CALIBRATION.source or CT_CALIBRATION.kind})')
        print(f'  quantitative HU->acoustics: '
              f'{"ENABLED" if CT_CALIBRATION.is_calibrated else "REFUSED (guarded)"}; '
              f'slab ramp: '
              f'{"placeholder (flagged)" if getattr(SLAB_RAMP, "placeholder", False) else "calibrated"}')
    ppw = hu_acoustics.report_ppw(SIM_DX_M, SIM_FREQS_HZ, c_min_m_s=1500.0,
                                  ppw_target=PPW_TARGET, verbose=verbose)
    report = {'calibration': CT_CALIBRATION.kind, 'sim_dx_m': SIM_DX_M,
              'ppw': [{'freq_hz': f, 'lambda_m': lam, 'ppw': p}
                      for f, lam, p in ppw]}

    if placement is not None and os.path.exists(SAIMIRI_RAS_WORKING):
        apex = np.asarray(placement['xdc_center_lps'])
        beam = np.asarray(placement['beam_dir_3d'])
        c_coarse, dxc = _central_ray_c_profile(apex, beam, SIM_DX_M)
        c_fine, dxf = _central_ray_c_profile(apex, beam, SIM_DX_M / 2.0)
        report['grid_convergence'] = hu_acoustics.grid_convergence_tof(
            c_coarse, dxc, c_fine, dxf, verbose=verbose)
    elif verbose:
        print('  grid-convergence run skipped (needs a placement + staged scan)')
    return report


# ---------------------------------------------------------------------------
# Deliverable 2: QC overlay figure
# ---------------------------------------------------------------------------
def qc_figure(out_png=ATLAS_QC_PNG, targets=None, verbose=True):
    """Three-panel orthoslice QC: bone shell (gray) + warped VALiDATe29
    brain mask (cyan) + S1/M1 target markers (yellow). Deliverable 2."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nibabel as nib

    bone = np.asarray(nib.load(SAIMIRI_RAS_WORKING).dataobj)
    brain_img = nib.load(BRAIN_IN_SUBJECT)
    brain = np.asarray(brain_img.dataobj) > 0
    aff = brain_img.affine
    inv = np.linalg.inv(aff)
    bone_b = bone > BONE_LOW
    c = np.argwhere(brain).mean(0).astype(int)

    if targets is None and os.path.exists(TARGETS_JSON):
        import json
        with open(TARGETS_JSON) as f:
            targets = json.load(f)['targets']
    targets = targets or {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    planes = [('sagittal', 0), ('coronal', 1), ('axial', 2)]
    for ax, (title, axis) in zip(axes, planes):
        sel = [slice(None)] * 3
        sel[axis] = int(c[axis])
        sl = tuple(sel)
        ax.imshow(bone_b[sl].T, origin='lower', cmap='gray')
        ax.imshow(np.ma.masked_where(~brain[sl].T, brain[sl].T),
                  origin='lower', cmap='cool', alpha=0.45)
        for key, t in targets.items():
            vx = inv[:3, :3] @ np.array(t['world_mm']) + inv[:3, 3]
            xy = {0: (vx[1], vx[2]), 1: (vx[0], vx[2]),
                  2: (vx[0], vx[1])}[axis]
            ax.plot(*xy, 'o', ms=7, mfc='none', mec='yellow', mew=2)
            ax.annotate(key, xy, color='yellow', fontsize=7,
                        xytext=(4, 4), textcoords='offset points')
        ax.set_title(f'{title}  (bone gray, VALiDATe29 brain cyan)')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('tuba.saimiri — VALiDATe29 atlas + S1/M1 targets in the '
                 'Saimiri sciureus skull', fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=100)
    if verbose:
        print(f'  wrote {out_png}')
    return out_png


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------
def build(force=False, verbose=True):
    """D1+D2 end-to-end: ingest -> align -> cavity -> VALiDATe29 SyN ->
    warp atlas -> export S1/M1 targets -> QC overlay -> acoustic-model
    report. Needs the staged scan + atlas (+ antspyx)."""
    os.makedirs(REG_DIR, exist_ok=True)
    downsample_to_working(force=force, verbose=verbose)
    align_skull_to_axes(force=force, verbose=verbose)
    extract_cavity(force=force, verbose=verbose)
    register_to_atlas(force=force, verbose=verbose)
    targets = export_targets(verbose=verbose)
    qc_figure(targets=targets['targets'], verbose=verbose)
    pl = place_on_skull(target_key='M1', apex_to_target_mm=20.0, verbose=verbose)
    acoustic_model_report(placement=pl, verbose=verbose)
    return {'targets': TARGETS_JSON, 'qc': ATLAS_QC_PNG,
            'labels': ANNOT_NII, 'placement': pl}


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    if cmd == 'build':
        build()
    elif cmd == 'cavity':
        extract_cavity()
    elif cmd == 'register':
        register_to_atlas()
    elif cmd == 'targets':
        export_targets()
    elif cmd == 'report':
        acoustic_model_report()      # PPW only (no data needed)
    else:
        print('Usage: python -m tuba.species.saimiri '
              '{build|cavity|register|targets|report}')
