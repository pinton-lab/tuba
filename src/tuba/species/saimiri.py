"""Squirrel monkey (Saimiri sciureus museum microCT + VALiDATe29 atlas)
species binding -- NIH R01 EB037345, Aim 3.

Geometry-only pillar. Where the mouse/rat/macaque/human pillars each
register a subject skull to a species brain atlas and then map intensity
to acoustic properties, this pillar is bootstrapped on an **uncalibrated
museum microCT** (Saimiri sciureus USNM 338948; see
:mod:`tuba.data.fetch_saimiri`). It therefore does the geometry --
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

PROVISIONAL constants
--------------------
Orientation flips, intensity thresholds, and the cavity hull geometry
below are marked ``PROVISIONAL``: like every other pillar they are fixed
by an orientation/histogram probe on the *staged* scan (cf. the rat
pillar's determination trail). They carry rat/macaque-scaled defaults so
the pipeline is runnable the moment the scan lands, but must be
re-confirmed against it before any result is trusted.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from tuba.atlases.validate29 import VALiDATe29
from tuba.core import (align, cavity as cav, downsample, hu_acoustics,
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

# DigiMorph/UTCT USNM 338948 unpacks to a TIFF stack; glob is permissive
# because the museum archive layout is not fixed until it is staged.
TIFF_GLOB = os.path.join(SOURCE_DIR, '**', '*.tif*')

# Cached intermediates (subject aligned-native RAS frame).
SAIMIRI_RAS_NIFTI         = os.path.join(REG_DIR, 'saimiri_skull_119um.nii.gz')
SAIMIRI_RAS_ALIGNED       = os.path.join(REG_DIR, 'saimiri_skull_aligned_119um.nii.gz')
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
NATIVE_VOXEL_MM = 0.1189      # USNM 338948 reconstructed pitch (DigiMorph)
WORKING_VOXEL_MM = NATIVE_VOXEL_MM  # already ~119 um; no downsample needed

# Storage-axis convention (PROVISIONAL). Defaulted to the macaque/rat
# museum-CT convention (slice=AP, row=DV, col=LR); confirm with an
# orientation probe (mid-stack orthoslices) on the staged scan.
SLICE_FLIP = False
ROW_FLIP = False
COL_FLIP = True
SWAP_ROW_COL = True
AXIS_FLIP_KWARGS = {'slice_flip': SLICE_FLIP, 'row_flip': ROW_FLIP,
                    'col_flip': COL_FLIP, 'swap_row_col': SWAP_ROW_COL}
VOXEL_SIGNS = (-1, -1, +1)    # storage (i=R, j=A, k=V) -> RAS world

# Intensity thresholds (PROVISIONAL, uncalibrated counts). Set from the
# staged scan's histogram (air / soft-tissue / cortical-bone modes) as in
# the rat pillar. Defaults assume a 16-bit reconstruction.
BONE_LOW = 8000.0             # shell-seed floor + placeholder-ramp i_low
BONE_HIGH = 20000.0           # cavity-exclusion ceiling
BONE_RAMP_HIGH = 32000.0      # placeholder-ramp cortical anchor (i_high)

# Pre-alignment rotation (PROVISIONAL). Museum stacks are near axis-aligned
# after the swap+transpose; a PCA probe on the staged bone cloud fixes any
# residual tilt (cf. rat pillar, which lands at 0 deg).
ALIGN_RX_DEG = 0.0
ALIGN_RY_DEG = 0.0
ALIGN_RZ_DEG = 0.0
NATIVE_PAD_VOXELS = 200

# Cavity hull geometry (PROVISIONAL, squirrel-monkey scale). Squirrel
# monkey whole-brain volume is ~22 mL; the endocranial cavity target is
# ~20-28 mL. Hull half-extents interpolate between the rat (~2 mL) and
# macaque (~90 mL) pillars.
SEED_CLOSE_MM = 1.5
PLUG_SMOOTH_MM = 1.5
HULL_X_HALF_MM = 18.0
HULL_Y_MAX_LOWZ_MM = -3.0
HULL_Y_MAX_HIGHZ_MM = +10.0
HULL_Z_MIN_MM = -12.0
HULL_Z_YCUTOFF_BREAK_MM = +1.0
CAVITY_TARGET_ML = (20.0, 28.0)

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
    the working (~119 um) pitch. Factor-1 (the scan is already ~119 um):
    applies axis flips + RAS affine only."""
    return downsample.downsample_tiff_stack_to_nifti(
        _tiff_paths(),
        target_voxel_mm=WORKING_VOXEL_MM,
        native_voxel_mm=NATIVE_VOXEL_MM,
        out_path=SAIMIRI_RAS_NIFTI,
        axis_flip_kwargs=AXIS_FLIP_KWARGS,
        voxel_signs=VOXEL_SIGNS,
        force=force, verbose=verbose,
    )


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
    """Extract the squirrel-monkey endocranial cavity as the moving image
    for the cavity_binary SyN. Bone-shell seed (per-column floor + per-row
    plugs + per-axial hull) -> close/fill -> largest-CC. Writes
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
            f'No voxels above BONE_LOW={BONE_LOW} in {SAIMIRI_RAS_ALIGNED}. '
            f'The PROVISIONAL threshold needs calibrating to the staged scan '
            f'(intensity range [{arr.min():.0f}, {arr.max():.0f}]).')
    bone_centroid_world = aff[:3, :3] @ np.argwhere(bone_mask).mean(0) + aff[:3, 3]
    aff_centred = aff.copy()
    aff_centred[:3, 3] -= bone_centroid_world

    seed, info = cav.bone_shell_cavity(
        arr, aff_centred, bone_low=BONE_LOW,
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
# Targets: S1 (areas 3b, 1) + M1 hand representations (deliverable 2)
# ---------------------------------------------------------------------------
TARGET_KEYS = ('S1_3b', 'S1_area1', 'M1')


def _label_centroid_world_mm(label_id, hemisphere=None):
    """Centroid (subject RAS mm) of a warped VALiDATe29 label id, optionally
    restricted to one hemisphere by sign of x."""
    import nibabel as nib
    img = nib.load(ANNOT_NII)
    arr = np.asarray(img.dataobj)
    aff = img.affine
    ijk = np.argwhere(arr == label_id)
    if len(ijk) == 0:
        raise RuntimeError(f'No voxels with label id {label_id} in {ANNOT_NII}')
    ras = (aff[:3, :3] @ ijk.T).T + aff[:3, 3]
    if hemisphere == 'left':
        ras = ras[ras[:, 0] < 0]
    elif hemisphere == 'right':
        ras = ras[ras[:, 0] > 0]
    if len(ras) == 0:
        raise RuntimeError(f'No {hemisphere!r} voxels for label {label_id}')
    return ras.mean(0), int(len(ras))


def target_world_mm(target_key, hemisphere='left'):
    """Subject-space RAS-mm centroid of a study target
    (``'S1_3b'`` / ``'S1_area1'`` / ``'M1'``), resolved through the
    VALiDATe29 cortical labels warped into subject space.

    NOTE: this returns the whole-area centroid. Localizing the *hand*
    sub-representation within the area needs VALiDATe29's stereotaxic
    frame (a mediolateral/AP prior on the homunculus); that prior is
    applied by ``export_targets(hand_prior=...)`` once the atlas
    coordinate frame is confirmed against the staged LUT.
    """
    if not os.path.exists(ANNOT_NII):
        raise FileNotFoundError(
            f'{ANNOT_NII} missing; run register_to_atlas() first.')
    name, label_id = ATLAS.resolve_label(target_key)
    world, n = _label_centroid_world_mm(label_id, hemisphere=hemisphere)
    return {'target': target_key, 'atlas_label': name, 'label_id': label_id,
            'hemisphere': hemisphere, 'world_mm': tuple(float(v) for v in world),
            'n_vox': n}


def export_targets(hemisphere='left', verbose=True):
    """Export S1-3b, S1-area-1, and M1 target coordinates (subject RAS mm)
    to :data:`TARGETS_JSON`, alongside the transform prefix. Deliverable 2.
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
