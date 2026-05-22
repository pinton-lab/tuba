"""Rat (DigiMorph TMM M-2272 dry-skull microCT + WHS atlas) species binding.

Production pillar -- structurally a near-clone of the macaque pillar
(both are dry-skull / cavity-binary registrations using the bone-shell
cavity builder). The constants below are calibrated against the
staged DigiMorph TMM M-2272 stack; see ``docs/rat/manuscript.pdf``
for the calibration trail and quantitative figures.

Scan (DigiMorph TMM M-2272)
---------------------------
* Source: Texas Memorial Museum specimen M-2272 (Rattus norvegicus),
  Digital Morphology Library, UT Austin. 0.6 GB ZIP from
  utexas.box.com (see :mod:`tuba.data.fetch_rat`).
* 1571 axial TIFFs of 1024x1024 at 29.61 um isotropic;
  ~28 x 28 x 46 mm extent. Dry skull only (no soft tissue, no
  in-situ brain). Strain unspecified — museum specimen.

Wavelength criterion
--------------------
At a typical rat-TUS centre frequency of 500 kHz, cortical-bone
wavelength is ~5.8 mm (c_bone ~ 2900 m/s). The native 29.6 um cache
samples ~196 vox/lambda, which keeps ~24 vox/lambda even after
factor-8 block-max to a 237 um working cache — same headroom as the
macaque 242 um pillar.

Atlas
-----
Waxholm Space SD rat atlas v4.01 (Kleven et al. 2023, Nature Methods):
39 um ex vivo T2*-weighted MRI template from an 80-day-old male
Sprague-Dawley rat (Duke CIVM) with 222 labelled structures. See
:class:`tuba.atlases.whs.WaxholmSpace`. The strain mismatch between
the TMM museum specimen and the lab SD atlas is on the same order as
the AMNH macaque vs NMT v2 cohort gap.

Pipeline stages
---------------
Same staging template as the mouse module:
* §3 Orientation flips from per-slice bone-mass profile + orthoslices
* §4 Streaming block-max downsample to a working ANTs cache
* §5 PCA-based rigid pre-alignment on the cortical-bone cloud
* §6 Endocranial-cavity extraction (hysteresis threshold + close +
     caudal seal)
* §7 ANTs SyN: cavity (moving) -> WHS brain mask (fixed)
* §8 Piecewise-linear intensity -> (c, rho) ramp

Environment variables
---------------------
* ``TUBA_RAT_REG_DIR``     -- registration/cache output (default
  ``~/.cache/tuba/rat/registration``).
* ``TUBA_RAT_SOURCE_DIR``  -- raw DigiMorph TIFF stack (default
  ``~/.cache/tuba/rat/source``; see :mod:`tuba.data.fetch_rat`).
* ``WHS_DEST``             -- WHS atlas root (mirrors
  :mod:`tuba.data.fetch_whs`).
"""
import glob
import os

import nibabel as nib
import numpy as np

from tuba.atlases.whs import WaxholmSpace
from tuba.core import (align, cavity as cav, downsample, frame, placement as plc,
                       qc, slab, surface, warp)
from tuba.io.tiff_stack import load_stack_to_array

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REG_DIR = os.environ.get(
    'TUBA_RAT_REG_DIR',
    os.path.expanduser('~/.cache/tuba/rat/registration'))

SOURCE_DIR = os.environ.get(
    'TUBA_RAT_SOURCE_DIR',
    os.path.expanduser('~/.cache/tuba/rat/source'))
# DigiMorph distributes TMM M-2272 as 1571 uint8 TIFFs at
# `TMM_M-2272/Rattus_norvegicus/8bitTIFF/rat0001.tif` through rat1571.tif.
TIFF_GLOB = os.path.join(
    SOURCE_DIR, 'TMM_M-2272', 'Rattus_norvegicus', '8bitTIFF', 'rat*.tif')

# Cached intermediates (rat aligned-native RAS frame).
RAT_RAS_NIFTI            = os.path.join(REG_DIR, 'rat_skull_240um.nii.gz')
RAT_RAS_NIFTI_60UM       = os.path.join(REG_DIR, 'rat_skull_60um.nii.gz')
RAT_RAS_NIFTI_ALIGNED    = os.path.join(REG_DIR, 'rat_skull_aligned_240um.nii.gz')
RAT_RAS_NATIVE_ALIGNED   = os.path.join(REG_DIR, 'rat_skull_aligned_native.nii.gz')
RAT_RAS_ALIGNED_60UM     = os.path.join(REG_DIR, 'rat_skull_aligned_60um.nii.gz')
RAT_RAS_ALIGNED_240UM    = os.path.join(REG_DIR, 'rat_skull_aligned_native_240um.nii.gz')

CAVITY_NII = os.path.join(REG_DIR, 'rat_cranial_cavity.nii.gz')
CAVITY_QC  = os.path.join(REG_DIR, 'rat_cranial_cavity_qc.png')
ANNOT_NII  = os.path.join(REG_DIR, 'whs_annotation_in_rat.nii.gz')

# WHS atlas binding. Defaults to the fetch_whs.py cache location; can be
# overridden by setting WHS_DEST.
ATLAS = WaxholmSpace(
    atlas_dir=os.environ.get(
        'WHS_DEST',
        os.path.expanduser('~/.cache/tuba/rat/atlas/whs_sd_v4')))

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
NATIVE_VOXEL_MM     = 0.029606  # TMM M-2272 reconstructed pitch (29.606 um)
DOWNSAMPLE_VOXEL_MM = 0.240     # ANTs cache (factor-8 -> 236.8 um)

# Storage-axis convention. DigiMorph TMM M-2272 storage is
# (slice=AP, row=DV, col=LR) -- same as macaque AMNH, opposite of mouse
# Maga (slice=AP, row=LR, col=DV). SWAP_ROW_COL is set True so the
# rest of the pipeline (SLICE_FLIP / ROW_FLIP / COL_FLIP semantics and
# the closing transpose(1,0,2)) match mouse exactly. Orientation probe
# (orthoslices in docs/rat/figures/native_*.png): storage slice 0 is
# rostral (nasal cavity cross-section), slice 1556 is caudal (occipital
# / foramen magnum); storage row 0 is ventral (mandible/maxilla side),
# row 1023 is dorsal (cranial vault); storage col 0 is one side of the
# skull, col 1023 the other (L/R sign is symmetric and resolved by SyN).
SLICE_FLIP = False              # storage slice 0 = rostral; after the
                                # closing transpose(1,0,2) this lands at
                                # j=0 = anterior (+y) directly
ROW_FLIP   = False              # symmetric L-R; let Affine catch any mirror
COL_FLIP   = True               # storage row 0 = dorsal (cranial vault
                                # side); the inverted-U at the bottom of
                                # anchor slice 568 looked like a dental
                                # arcade but is actually a vault-side view
                                # of the frontal bones. Confirmed by the
                                # location of dental enamel (the brightest
                                # voxels): without the flip they sit at
                                # +z (supposedly dorsal) -- impossible for
                                # ventrally-located teeth. COL_FLIP=True
                                # puts ventral (dental) at -z and dorsal
                                # (vault) at +z, matching RAS convention.
SWAP_ROW_COL = True
AXIS_FLIP_KWARGS = {'slice_flip': SLICE_FLIP, 'row_flip': ROW_FLIP,
                    'col_flip': COL_FLIP, 'swap_row_col': SWAP_ROW_COL}
VOXEL_SIGNS = (-1, -1, +1)       # storage (i=R, j=A, k=V) -> RAS world

# Intensity thresholds. DigiMorph TMM M-2272 is distributed as 8-bit
# TIFFs (uint8, range 0-255). The histogram from a mid-stack slice
# resolves three populations: air <=31 (~92% of voxels), soft-tissue /
# mounting / trabecular bone 46-153 (~7%), cortical bone >=168 (~1%).
# A single threshold cleanly separates bone from background; the
# split-low/split-high convention from mouse is retained for the
# cavity extractor's hysteresis step.
BONE_THRESH      = 50       # orientation-probe threshold (above air)
ALIGN_PCA_THRESH = 120      # above mount AND vault; concentrates the
                            # PCA cloud on cortical bone
BONE_HIGH        = 140      # cavity-exclusion ceiling
BONE_LOW         =  50      # shell-seed floor + slab-ramp i_low

# Cavity extraction. Rat cranial cavity is ~2 mL (vs ~0.45 mL mouse);
# scale the morphological closing radius up accordingly to bridge the
# larger sutural gaps.
CLOSE_MM        = 2.0       # TODO refine: 1.0-2.5 mm range, optimize on volume
CAUDAL_SEAL_MM  = 1.5       # foramen magnum is larger in rat (~3 mm dia)

# Pre-alignment rotation. The DigiMorph TMM M-2272 skull is already
# axis-aligned in world frame after the swap_row_col + transpose
# (orthoslice QC: docs/rat/figures/world_orthoslices_240um.png), so no
# meaningful rotation is needed; the residual eigenvector tilts are <5
# deg per axis. PCA pre-alignment via tuba.core.align.find_pca_rotation
# is skipped because the dry rat skull's bone-voxel variance ordering
# is AP > DV > LR (the missing mandible removes most ventral mass), not
# the AP > LR > DV assumed by find_pca_rotation -- the function's
# automatic eigenvector-to-axis assignment would mis-label DV and LR
# and produce a ~85 deg spurious Ry rotation. See manuscript Sec 4.
ALIGN_RX_DEG = 0.0
ALIGN_RY_DEG = 0.0
ALIGN_RZ_DEG = 0.0

# Output padding for native rotation: 200 voxels at 29.6 um = 5.9 mm
# cushion (scaled from mouse's 396 vox * 6.127 um = 2.4 mm).
NATIVE_PAD_VOXELS = 200

# Slab loader: piecewise-linear intensity -> (c, rho) ramp anchored on
# the 8-bit DigiMorph histogram. i_low at the air/soft-tissue separator
# (50), i_high at the cortical-bone mode (~170; tail extends to ~245).
SLAB_RAMP = slab.IntensityToAcousticRamp(
    i_low=50.0, i_high=170.0,
    c_water=1540.0, rho_water=1000.0,
    c_bone_max=2900.0, rho_bone_max=1900.0,
)
SLAB_CAVITY_INFILL = slab.CavityInfill(c_parenchyma=1540.0, rho_parenchyma=1040.0)

# Attenuation (literature values; not pillar-specific).
ALPHA_BONE_DB_PER_CM_PER_MHZ   = 8.0
ALPHA_TISSUE_DB_PER_CM_PER_MHZ = 0.5

# Registration prefix.
SYN_PREFIX = 'rat_cavity_to_whs'


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def _tiff_paths():
    paths = sorted(glob.glob(TIFF_GLOB))
    if not paths:
        raise FileNotFoundError(
            f'No TIFFs match {TIFF_GLOB}. '
            f'Run python -m tuba.data.fetch_rat first.')
    return paths


def downsample_to_240um(force=False, verbose=True):
    """Stream native TIFFs into a 240 um RAS NIfTI cache (factor-8)."""
    return downsample.downsample_tiff_stack_to_nifti(
        _tiff_paths(),
        target_voxel_mm=DOWNSAMPLE_VOXEL_MM,
        native_voxel_mm=NATIVE_VOXEL_MM,
        out_path=RAT_RAS_NIFTI,
        axis_flip_kwargs=AXIS_FLIP_KWARGS,
        voxel_signs=VOXEL_SIGNS,
        force=force, verbose=verbose,
    )


def downsample_to_118um(force=False, verbose=True):
    """Build the 118 um working cache directly from native TIFFs
    (factor-4 block-max). Skips the align-native step because the
    DigiMorph storage is already axis-aligned in world frame after the
    swap+transpose -- see manuscript Sec 3 + 4. Output path reuses
    :data:`RAT_RAS_ALIGNED_60UM`, the species-API canonical working
    cache (the constant is mis-named for legacy reasons; the actual
    voxel size is 118.4 um, not 60 um)."""
    return downsample.downsample_tiff_stack_to_nifti(
        _tiff_paths(),
        target_voxel_mm=4 * NATIVE_VOXEL_MM,    # exact factor-4 -> 118.4 um
        native_voxel_mm=NATIVE_VOXEL_MM,
        out_path=RAT_RAS_ALIGNED_60UM,
        axis_flip_kwargs=AXIS_FLIP_KWARGS,
        voxel_signs=VOXEL_SIGNS,
        force=force, verbose=verbose,
    )


# Back-compat alias matching the species-API naming used by other modules.
downsample_to_60um = downsample_to_118um


def find_alignment_rotation(in_path=None, bone_thresh=ALIGN_PCA_THRESH,
                            verbose=True):
    """PCA-based rotation angles for the 240 um cache."""
    if in_path is None:
        in_path = RAT_RAS_NIFTI
    img = nib.load(in_path)
    vol = img.get_fdata()
    return align.find_pca_rotation(vol, bone_thresh=bone_thresh, verbose=verbose)


def align_skull_to_axes(in_path=None, out_path=None,
                        ax_deg=ALIGN_RX_DEG, ay_deg=ALIGN_RY_DEG,
                        az_deg=ALIGN_RZ_DEG, force=False, order=1,
                        bone_thresh=ALIGN_PCA_THRESH, pad_voxels=12,
                        verbose=True):
    """Rotate the 240 um cache about the bone centroid."""
    return align.align_nifti_to_axes(
        in_path=in_path or RAT_RAS_NIFTI,
        out_path=out_path or RAT_RAS_NIFTI_ALIGNED,
        ax_deg=ax_deg, ay_deg=ay_deg, az_deg=az_deg,
        bone_thresh=bone_thresh, pad_voxels=pad_voxels,
        voxel_signs=VOXEL_SIGNS, order=order, force=force, verbose=verbose,
    )


def align_native_to_axes(out_path=None,
                         ax_deg=ALIGN_RX_DEG, ay_deg=ALIGN_RY_DEG,
                         az_deg=ALIGN_RZ_DEG, force=False,
                         bone_thresh=ALIGN_PCA_THRESH,
                         pad_voxels=NATIVE_PAD_VOXELS, verbose=True):
    """Stream native TIFFs, apply axis flips, rotate about bone centroid,
    write a uint16 29.6 um NIfTI (~3 GB raw / ~1 GB gzipped).
    """
    import time
    out_path = out_path or RAT_RAS_NATIVE_ALIGNED
    if os.path.exists(out_path) and not force:
        return out_path
    t0 = time.time()

    paths = _tiff_paths()
    if verbose:
        print(f'[1/4] Loading {len(paths)} native TIFFs into a uint16 array...')
    raw = load_stack_to_array(paths, dtype=np.uint16, verbose=verbose)

    if verbose:
        print('[2/4] Applying axis flips and transpose to RAS index space...')
    flipped = frame.apply_axis_flips(raw, **AXIS_FLIP_KWARGS)
    del raw
    flipped = np.ascontiguousarray(flipped)
    if verbose:
        print(f'  flipped/transposed: shape={flipped.shape}, '
              f'size={flipped.nbytes/1e9:.2f} GB, t={(time.time()-t0):.0f} s')

    if verbose:
        print(f'[3/4] Rotating native volume about bone centroid '
              f'(thresh > {bone_thresh}, pad={pad_voxels})...')
    rotated = align.rotate_volume_about_centroid(
        flipped, ax_deg, ay_deg, az_deg,
        bone_thresh=bone_thresh, pad_voxels=pad_voxels,
        order=1, verbose=verbose)
    del flipped

    if verbose:
        print('[4/4] Writing NIfTI...')
    affine = frame.affine_for_shape(rotated.shape, NATIVE_VOXEL_MM,
                                    signs=VOXEL_SIGNS)
    nib.save(nib.Nifti1Image(rotated, affine), out_path)
    if verbose:
        print(f'  wrote {out_path}  ({os.path.getsize(out_path)/1e9:.2f} GB)')
        print(f'  TOTAL wall time: {(time.time()-t0)/60:.1f} min')
    return out_path


def downsample_aligned_to_60um(in_path=None, out_path=None,
                                factor=2, force=False, verbose=True):
    """Block-max factor-2 from the aligned native -> 60 um cache."""
    align_native_to_axes()
    return downsample.downsample_nifti_by_factor(
        in_path=in_path or RAT_RAS_NATIVE_ALIGNED,
        out_path=out_path or RAT_RAS_ALIGNED_60UM,
        factor=factor, force=force, verbose=verbose,
    )


def downsample_aligned_to_240um(in_path=None, out_path=None,
                                 factor=8, force=False, verbose=True):
    """Block-max factor-8 from the aligned native -> 240 um cache."""
    align_native_to_axes()
    return downsample.downsample_nifti_by_factor(
        in_path=in_path or RAT_RAS_NATIVE_ALIGNED,
        out_path=out_path or RAT_RAS_ALIGNED_240UM,
        factor=factor, force=force, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Cavity extraction + atlas warp (Affine recipe, ported from macaque)
# ---------------------------------------------------------------------------
# The pure-hysteresis cavity extractor (tuba.core.cavity.extract_cavity)
# fails on rat: the orbits (~6 mm) and foramen magnum (~3 mm) leak
# through any practical closing radius. We adopt the macaque recipe
# (tuba._macaque.fast_iter_250um, refactored into
# tuba.core.cavity.bone_shell_cavity in this work):
#
#   1. Bone-shell seed: per-column curved basicranium floor + per-row
#      curved rostral/caudal plugs + per-axial-slice convex hull with
#      z-dependent rostral cutoff + region-grow outside + largest CC +
#      geodesic propagation into bone-bounded pockets.
#   2. ANTs Affine: bone-shell cavity (moving) -> WHS brain mask
#      (fixed); mean-squares metric over the 12-DOF affine.
#   3. Inverse-warp the WHS brain mask back into rat space + 2 mm
#      morphological closing to clean linear-interp jaggedness.
#   4. Bone-clip against a soft threshold (arr > BONE_CLIP_SOFT) to
#      guarantee no cavity voxel sits inside cortical bone.
#
# The same Affine matrix is reused (no re-fit) to warp the WHS T2*
# template + v4.01 annotation into rat space.
SEED_CLOSE_MM           = 2.0       # macaque-style closing radius for the
                                    # bone shell (good shape, anatomically
                                    # tight); the larger POST_CLOSE_MM
                                    # below corrects the 2 mm caudal gap
                                    # that closing introduces.
POST_DILATE_MM          = 1.0       # isotropic post-warp dilation of the
                                    # bone-shell seed. Extends the cavity
                                    # 1 mm in all directions: fills the
                                    # caudal gap (the seed under-shoots
                                    # the most-caudal bone by ~2 mm)
                                    # and adds margin around the lateral
                                    # + dorsal boundaries.
VENTRAL_DILATE_MM       = 2.0       # asymmetric -z (ventral) dilation of
                                    # the bone-shell seed. The per-column
                                    # basicranium floor of bone_shell_cavity
                                    # truncates the brainstem / cerebellar
                                    # peduncles that sit just dorsal of
                                    # the basicranium plate but extend
                                    # slightly below where bone exists in
                                    # some (i,j) columns. Extending the
                                    # cavity 2 mm in -z recaptures them.
                                    # The subsequent bone-clip prevents
                                    # overshoot into the basicranium plate
                                    # or dental palate.
                                    # caudal occipital bone at y=-23.15
                                    # mm without flat-line cutoffs in
                                    # other directions.
POST_CLOSE_MM           = 2.0       # post-warp jaggedness cleanup
CAUDAL_PAD_MM           = 4.0       # asymmetric caudal padding of the
                                    # cavity after bone-clip + largest-CC.
                                    # The Affine + bone-clip undershoots
                                    # the most-caudal occipital bone by
                                    # ~2-4 mm because the WHS atlas brain
                                    # is bounded by the WHS atlas's own
                                    # cerebellar contour, not the rat
                                    # specimen's actual back-of-skull.
                                    # Extending the cavity caudally
                                    # (and re-clipping against bone)
                                    # closes the gap.
BONE_CLIP_SOFT          = 28        # legacy; not used in production pipeline.
BONE_CLIP_HARD          = 100       # only the densest cortical bone walls
                                    # are excluded from the cavity. Soft
                                    # tissue + trabecular fragments + dried
                                    # meninges (intensity 25-100) stay as
                                    # cavity because they are physically
                                    # inside the cranium and don't impede
                                    # the acoustic-property mapping (the
                                    # cavity infill at \S\ref{sec:acoustic-mapping}
                                    # overrides their intensity with brain-
                                    # parenchymal properties regardless).
                                    # At threshold 100 the bone-clip only
                                    # cuts dense cortical walls, so the
                                    # cavity stays smooth + atlas-shaped.
# Rat-scale per-axial hull constants (macaque equivalents in parens; rat
# is ~3x smaller in linear dimension). These are expressed relative to
# a BONE-CENTROID-AT-ORIGIN affine (the macaque convention; the macaque
# pipeline gets there via `align_native_to_axes` which both rotates and
# translates). For rat we skip the rotation but still need the
# bone-centroid translation, applied inline in extract_cavity() before
# calling bone_shell_cavity.
HULL_X_HALF_MM          = 12.0      # (35.0) lateral half-width
HULL_Y_MAX_LOWZ_MM      = -3.0      # (-3.0) rostral cutoff at low z
                                    # (basicranium range, exclude snout
                                    # rostral of olfactory bulb start)
HULL_Y_MAX_HIGHZ_MM     = +5.0      # (+20.0) rostral cutoff at high z
                                    # (vault range; rat olfactory bulb +
                                    # frontal vault extend rostrally to
                                    # ~+5 mm of bone centroid)
HULL_Z_MIN_MM           = -8.0      # (-25.0) ventral cutoff (below
                                    # basicranium = dental / mounting)
HULL_Z_YCUTOFF_BREAK_MM = +1.0      # (0.0) z transition basicranium -> vault
PLUG_SMOOTH_MM          = 2.5       # 2D median-filter radius for the per-
                                    # column floor + per-row rostral / caudal
                                    # plugs. Smooths the bone-shell boundary
                                    # surface so the cavity contour doesn't
                                    # show the per-row stepped pattern at
                                    # rat 118 um voxel pitch.

# Concave-bowl-shaped "bone dam" added to the bone mask at the foramen
# magnum. The dam is the difference of two overlapping ellipsoids: an
# OUTER ellipsoid (the bone shell) and a smaller INNER ellipsoid
# offset rostrally (the cerebellar fossa cavity carved into the
# shell). The result is a bowl whose CONCAVE side faces rostrally
# (toward the brain) -- the cavity is bounded by the bowl's inner
# surface, which curves around the cerebellum like the actual rat
# occipital bone does.
DAM_OUTER_X_MM          = 0.0       # LR centre
DAM_OUTER_Y_MM          = -22.1     # outer centre at AP midpoint of foramen
DAM_OUTER_Z_MM          = +0.45     # DV centre at foramen midpoint
DAM_OUTER_X_RADIUS_MM   = 4.5       # outer LR radius (+1 mm vs. tight foramen)
DAM_OUTER_Y_RADIUS_MM   = 2.4       # outer AP radius (+1 mm)
DAM_OUTER_Z_RADIUS_MM   = 3.05      # outer DV radius (+1 mm)
DAM_INNER_X_MM          = 0.0       # LR centre
DAM_INNER_Y_MM          = -20.7     # inner centre rostral of outer -> bowl opens rostrally
DAM_INNER_Z_MM          = +0.45     # DV centre same as outer
DAM_INNER_X_RADIUS_MM   = 3.5       # inner LR radius (+1 mm)
DAM_INNER_Y_RADIUS_MM   = 3.0       # inner AP radius (+1 mm)
DAM_INNER_Z_RADIUS_MM   = 2.5       # inner DV radius (+1 mm; bowl rim 0.55 mm top/bot)
DAM_ROT_X_DEG           = +15.0     # rotation in DV/AP plane about LR axis


def extract_cavity(verbose=True):
    """Extract the rat endocranial cavity via Affine-warp of the WHS
    brain mask (manuscript Sec 5). Side-effect: writes
    ``rat_cavity_seed.nii.gz`` (bone-shell seed),
    ``rat_cavity_to_whs_affine.mat`` (saved Affine), and
    ``rat_cranial_cavity.nii.gz`` (production cavity). Returns the
    production-cavity NIfTI path."""
    import ants
    import scipy.ndimage as sn
    import shutil

    # Stage the 118 um working cache.
    cache = downsample_to_118um(verbose=verbose)
    img = nib.load(cache)
    arr = img.get_fdata().astype(np.float32)
    aff = img.affine
    voxel_mm = float(abs(aff[0, 0]))
    if verbose:
        print(f'\nExtract rat cavity at {voxel_mm*1000:.2f} um, shape={arr.shape}')

    # Compute bone-centroid offset and pass a centered affine to
    # bone_shell_cavity. The macaque hull constants are expressed
    # relative to the bone centroid (the macaque PCA-alignment step
    # puts the bone centroid at world origin); for rat we skip the
    # rotation but still need to translate to that convention.
    bone_mask = arr > BONE_LOW
    bone_centroid_world = (
        aff[:3, :3] @ np.argwhere(bone_mask).mean(0) + aff[:3, 3])
    aff_centred = aff.copy()
    aff_centred[:3, 3] -= bone_centroid_world
    if verbose:
        print(f'  bone centroid world: '
              f'({bone_centroid_world[0]:+.2f}, '
              f'{bone_centroid_world[1]:+.2f}, '
              f'{bone_centroid_world[2]:+.2f}) mm; using centred affine')

    # Step 1: bone-shell cavity (macaque recipe, shared in tuba.core.cavity).
    if verbose:
        print('  building bone-shell cavity (per-column floor + per-row '
              'plugs + per-axial hull + geodesic propagation)...')
    seed, info = cav.bone_shell_cavity(
        arr, aff_centred,
        bone_low=BONE_LOW,
        close_mm=SEED_CLOSE_MM,
        x_hull_half_mm=HULL_X_HALF_MM,
        y_hull_max_lowz_mm=HULL_Y_MAX_LOWZ_MM,
        y_hull_max_highz_mm=HULL_Y_MAX_HIGHZ_MM,
        z_hull_min_mm=HULL_Z_MIN_MM,
        z_ycutoff_break_mm=HULL_Z_YCUTOFF_BREAK_MM,
        use_hull=True,
        # Skip geodesic propagation for rat: at 118 um the bone walls
        # are 2-3 voxels thick and the propagation regularly leaks
        # through thin spots into orbital / cerebellar fossa pockets.
        use_geodesic_propagation=False,
        # Smooth the per-column / per-row plug boundaries so the bone-
        # shell surface follows a curved contour instead of the jagged
        # per-row stepped boundary that's visible in cavity QC plots
        # at 118 um voxel pitch. The macaque pillar's 250 um voxel size
        # makes the stepping invisible, so its default is 0.
        smooth_plug_mm=PLUG_SMOOTH_MM,
        verbose=verbose,
    )
    if verbose:
        print(f'  bone-shell cavity: {info["cavity_mm3"]/1000:.3f} mL '
              f'(target 1.8-2.4 mL adult rat)')

    seed_path = os.path.join(REG_DIR, 'rat_cavity_seed.nii.gz')
    nib.save(nib.Nifti1Image(seed.astype(np.uint8), aff), seed_path)

    # Step 2: WHS brain mask + ANTs Affine.
    mask_path = ATLAS.derive_brain_mask(verbose=verbose)
    if verbose:
        print('  Affine rat bone-shell -> WHS brain mask (12 DOF, meansquares)...')
    moving = ants.image_read(seed_path)
    fixed = ants.image_read(mask_path)
    reg = ants.registration(
        fixed=fixed, moving=moving,
        type_of_transform='Affine',
        aff_metric='meansquares',
        verbose=False)
    saved_aff = os.path.join(REG_DIR, 'rat_cavity_to_whs_affine.mat')
    shutil.copy(reg['fwdtransforms'][0], saved_aff)

    # Step 3: macaque recipe + rounded bone dam at foramen magnum.
    warped = ants.apply_transforms(
        fixed=moving, moving=fixed,
        transformlist=[saved_aff],
        whichtoinvert=[True],
        interpolator='linear')
    cav_arr = (warped.numpy() > 0.5)
    cav_arr = sn.binary_closing(cav_arr,
                                 iterations=int(round(POST_CLOSE_MM/voxel_mm)))

    # Concave-bowl bone dam at the foramen magnum: outer ellipsoid
    # minus inner ellipsoid offset rostrally. Bowl opens toward brain.
    # Both ellipsoids rotated by DAM_ROT_X_DEG about the LR axis
    # (rotation in DV/AP plane).
    ii_g, jj_g, kk_g = np.indices(arr.shape)
    cos_t = np.cos(np.deg2rad(DAM_ROT_X_DEG))
    sin_t = np.sin(np.deg2rad(DAM_ROT_X_DEG))

    def _ellipsoid_voxel(cx_mm, cy_mm, cz_mm, rx_mm, ry_mm, rz_mm):
        ci = int(round((cx_mm - aff[0, 3]) / aff[0, 0]))
        cj = int(round((cy_mm - aff[1, 3]) / aff[1, 1]))
        ck = int(round((cz_mm - aff[2, 3]) / aff[2, 2]))
        ri = max(1, int(round(rx_mm / voxel_mm)))
        rj = max(1, int(round(ry_mm / voxel_mm)))
        rk = max(1, int(round(rz_mm / voxel_mm)))
        dj = (jj_g - cj)
        dk = (kk_g - ck)
        # Rotate (dj, dk) by DAM_ROT_X_DEG in the DV/AP plane.
        dj_rot = dj * cos_t - dk * sin_t
        dk_rot = dj * sin_t + dk * cos_t
        return (((ii_g - ci) / ri) ** 2 +
                (dj_rot / rj) ** 2 +
                (dk_rot / rk) ** 2) <= 1.0

    outer = _ellipsoid_voxel(DAM_OUTER_X_MM, DAM_OUTER_Y_MM, DAM_OUTER_Z_MM,
                              DAM_OUTER_X_RADIUS_MM, DAM_OUTER_Y_RADIUS_MM,
                              DAM_OUTER_Z_RADIUS_MM)
    inner = _ellipsoid_voxel(DAM_INNER_X_MM, DAM_INNER_Y_MM, DAM_INNER_Z_MM,
                              DAM_INNER_X_RADIUS_MM, DAM_INNER_Y_RADIUS_MM,
                              DAM_INNER_Z_RADIUS_MM)
    dam = outer & ~inner
    if verbose:
        print(f'  concave-bowl bone dam: outer center ({DAM_OUTER_X_MM:+.1f},'
              f'{DAM_OUTER_Y_MM:+.1f},{DAM_OUTER_Z_MM:+.1f}) mm '
              f'r=({DAM_OUTER_X_RADIUS_MM},{DAM_OUTER_Y_RADIUS_MM},'
              f'{DAM_OUTER_Z_RADIUS_MM}); inner center '
              f'({DAM_INNER_X_MM:+.1f},{DAM_INNER_Y_MM:+.1f},'
              f'{DAM_INNER_Z_MM:+.1f}) r=({DAM_INNER_X_RADIUS_MM},'
              f'{DAM_INNER_Y_RADIUS_MM},{DAM_INNER_Z_RADIUS_MM}); '
              f'{dam.sum()*voxel_mm**3/1000:.3f} mL bowl shell')

    # Bone-clip against bone + dam.
    bone_with_dam = (arr > BONE_LOW) | dam
    cav_arr = cav_arr & ~bone_with_dam

    # Largest-CC: drops bone-clip islands.
    lab, n_cc = sn.label(cav_arr)
    if n_cc > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        main = int(np.argmax(sizes))
        cav_arr = (lab == main)
        if verbose:
            print(f'  largest-CC: {n_cc} -> 1 components '
                  f'({(1-sizes[main]/sizes.sum())*100:.1f}% of fragments dropped)')

    # 1.5 mm opening: erodes then dilates. Erosion disconnects thin
    # fingers (narrower than 2x the radius) that extend the cavity
    # into basicranium / sphenoid sub-cavities through small foramina;
    # subsequent dilation restores the main brain body to its
    # original extent.
    erode_iters = int(round(1.5 / voxel_mm))
    cav_arr = sn.binary_erosion(cav_arr, iterations=erode_iters)
    # Take largest CC of the eroded mask = the main brain body
    # (disconnected thin extensions are now separate components).
    lab, _ = sn.label(cav_arr)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    if sizes.size > 1:
        cav_arr = (lab == int(np.argmax(sizes)))
    cav_arr = sn.binary_dilation(cav_arr, iterations=erode_iters)
    cav_arr = sn.binary_fill_holes(cav_arr)
    cav_arr = cav_arr & ~bone_with_dam
    if verbose:
        print(f'  1.5 mm opening (erode + largest-CC + dilate): '
              f'{cav_arr.sum() * voxel_mm**3 / 1000:.3f} mL')

    # Posterior padding LAST: shift the cleaned-up cavity caudally
    # (+j storage = -y world). Padding after closing/largest-CC ensures
    # we extend the smoothed main body, not the noisy raw bone-clip
    # output. Re-clip against bone + dam so the pad cannot push
    # through the foramen magnum.
    n_caudal = max(0, int(round(CAUDAL_PAD_MM / voxel_mm)))
    if n_caudal > 0:
        padded = cav_arr.copy()
        for _ in range(n_caudal):
            shifted = np.zeros_like(padded)
            shifted[:, 1:, :] = padded[:, :-1, :]
            padded = padded | shifted
        cav_arr = padded & ~bone_with_dam
        if verbose:
            print(f'  posterior pad +{CAUDAL_PAD_MM:.1f} mm '
                  f'({n_caudal} voxels along +j storage) + bone-clip')

    # Final largest-CC after pad.
    lab, n_cc = sn.label(cav_arr)
    if n_cc > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        main = int(np.argmax(sizes))
        cav_arr = (lab == main)

    vol_mL = cav_arr.sum() * voxel_mm**3 / 1000

    cav_path = CAVITY_NII
    nib.save(nib.Nifti1Image(cav_arr.astype(np.uint8), aff), cav_path)
    if verbose:
        print(f'\nproduction cavity: {vol_mL:.3f} mL  (target 1.8-2.4 mL)')
        print(f'  wrote {cav_path}')
        print(f'  saved Affine: {saved_aff}')
    return cav_path


def register_to_atlas(verbose=True):
    """Affine cavity (\\S 5) + warp WHS T2* + v4.01 annotation into rat.

    Reuses the Affine transform produced inside :func:`extract_cavity`
    (no separate SyN step) -- the macaque pattern. After this call,
    ``whs_template_in_rat.nii.gz`` and ``whs_annotation_in_rat.nii.gz``
    are written to :data:`REG_DIR`."""
    import ants

    if not os.path.exists(CAVITY_NII):
        extract_cavity(verbose=verbose)
    saved_aff = os.path.join(REG_DIR, 'rat_cavity_to_whs_affine.mat')
    if not os.path.exists(saved_aff):
        raise RuntimeError(
            f'No saved Affine at {saved_aff}; rerun extract_cavity().')

    moving = ants.image_read(RAT_RAS_ALIGNED_60UM)
    t2 = ants.image_read(ATLAS.template_path)
    annot = ants.image_read(ATLAS.annotation_path)

    out_t2 = os.path.join(REG_DIR, 'whs_template_in_rat.nii.gz')
    out_annot = ANNOT_NII

    if verbose:
        print(f'Warping WHS T2* + v4.01 annotation into rat space via {saved_aff}')
    w_t2 = ants.apply_transforms(fixed=moving, moving=t2,
                                  transformlist=[saved_aff],
                                  whichtoinvert=[True],
                                  interpolator='linear')
    ants.image_write(w_t2, out_t2)
    if verbose:
        print(f'  wrote {out_t2}')
    w_annot = ants.apply_transforms(fixed=moving, moving=annot,
                                     transformlist=[saved_aff],
                                     whichtoinvert=[True],
                                     interpolator='nearestNeighbor')
    ants.image_write(w_annot, out_annot)
    if verbose:
        print(f'  wrote {out_annot}')


def warp_atlas_into_rat(verbose=True):
    """Alias kept for the species-API parity with mouse and macaque.
    Same effect as :func:`register_to_atlas` since the Affine matrix
    is computed exactly once inside :func:`extract_cavity`."""
    register_to_atlas(verbose=verbose)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------
def _structure_centroid_world_mm(structure_name, hemisphere=None):
    """Centroid of a warped WHS structure in rat aligned-native world-mm.

    Parameters
    ----------
    structure_name : str
        Canonical WHS v4.01 .label name (e.g.
        ``'Corpus callosum and associated subcortical white matter'``).
    hemisphere : {'left', 'right', None}
        Restrict to one hemisphere by sign of x (RAS); None returns
        the bilateral centroid.
    """
    img = nib.load(ANNOT_NII)
    arr = np.asarray(img.dataobj)
    aff = img.affine
    label_id = ATLAS.structure_id(structure_name)
    ijk = np.argwhere(arr == label_id)
    if len(ijk) == 0:
        raise RuntimeError(
            f'No voxels labelled {structure_name!r} (id {label_id}) in {ANNOT_NII}')
    ras = (aff[:3, :3] @ ijk.T).T + aff[:3, 3]
    if hemisphere == 'left':
        ras = ras[ras[:, 0] < 0]
    elif hemisphere == 'right':
        ras = ras[ras[:, 0] > 0]
    if len(ras) == 0:
        raise RuntimeError(
            f'No {hemisphere!r}-hemisphere voxels for {structure_name!r}')
    return ras.mean(axis=0), int(len(ras))


def place_on_skull(target_name='cavity_centre',
                   scalp_clearance_mm=None,
                   apex_to_target_mm=None,
                   beam_tilt_deg_yz=0.0,
                   lateral_search_mm=3.0,
                   verbose=True):
    """Place a focused bowl on the rat dry skull aimed at a brain target.

    Parameters
    ----------
    target_name : str
        - ``'cavity_centre'`` (default): cavity-mask centroid in the
          aligned-native frame.
        - any WHS v4.01 structure name (see
          :meth:`tuba.atlases.whs.WaxholmSpace.all_structures`), e.g.
          ``'Corpus callosum and associated subcortical white matter'``,
          ``'Caudate putamen'``, ``'Hippocampal formation'``.
    scalp_clearance_mm : float, optional
        Water column above outer bone. Ignored if ``apex_to_target_mm``
        is set.
    apex_to_target_mm : float, optional
        Euclidean apex-to-target distance (geometric focal length).
    beam_tilt_deg_yz : float
        Sagittal-plane tilt about the cavity centroid.
    lateral_search_mm : float
        Apex search cylinder radius.

    Returns
    -------
    placement : dict (same shape as
        :func:`tuba.species.mouse.place_on_skull`).
    """
    if verbose:
        print(f'Placing bowl on rat skull for target {target_name!r}')

    if target_name == 'cavity_centre':
        target_world = cav.cavity_centroid_world_mm(CAVITY_NII)
    else:
        target_world, n_vox = _structure_centroid_world_mm(target_name)
        if verbose:
            print(f'  {target_name!r} voxels (warped annotation): {n_vox}')

    if verbose:
        print('  building outer bone surface from 60 um aligned cache...')
    surf = surface.outer_bone_surface_world_mm(
        skull_path=RAT_RAS_ALIGNED_60UM,
        bone_thresh=BONE_LOW,
        close_mm=2.0,
        head_thresh=500.0,
        method='close+fill',
        verbose=verbose)

    cavity_centroid = cav.cavity_centroid_world_mm(CAVITY_NII)
    return plc.place_on_skull(
        target_world_mm=target_world,
        surface_world_mm=surf,
        cavity_centroid_world_mm=cavity_centroid,
        scalp_clearance_mm=scalp_clearance_mm,
        apex_to_target_mm=apex_to_target_mm,
        beam_tilt_deg_yz=beam_tilt_deg_yz,
        lateral_search_mm=lateral_search_mm,
        target_name=target_name,
        frame_name='rat_aligned_native_RAS',
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Slab loader
# ---------------------------------------------------------------------------
def load_slab(apex_world_mm, beam_3d, slab_size_m, dx_target_m,
              cache_path=None, cavity_mask_path=None,
              chatter=True):
    """Extract a beam-aligned slab from the rat skull at the 60 um
    aligned-native cache (default).

    Returns ``(c_map, rho_map, dz_slab, frame_dict)``.
    """
    return slab.load_slab(
        cache_path=cache_path or RAT_RAS_ALIGNED_60UM,
        apex_world_mm=apex_world_mm,
        beam_3d=beam_3d,
        slab_size_m=slab_size_m,
        dx_target_m=dx_target_m,
        ramp=SLAB_RAMP,
        cavity_mask_path=cavity_mask_path or CAVITY_NII,
        cavity_infill=SLAB_CAVITY_INFILL,
        verbose=chatter,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import json
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'place':
        target = os.environ.get('RAT_TARGET', 'cavity_centre')
        p = place_on_skull(target_name=target, apex_to_target_mm=20.0)
        out = os.path.join(REG_DIR, 'rat_placement.json')
        with open(out, 'w') as f:
            json.dump(p, f, indent=2)
        print(json.dumps(p, indent=2))
        print(f'wrote {out}')
    elif len(sys.argv) > 1 and sys.argv[1] == 'cavity':
        extract_cavity()
    elif len(sys.argv) > 1 and sys.argv[1] == 'register':
        register_to_atlas()
    else:
        print('Usage: python -m tuba.species.rat {place|cavity|register}')
