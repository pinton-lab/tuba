"""Mouse (Maga, UW 4K dry-skull microCT) species binding for TUBA.

Reference: Pinton, *Migrating the 15 MHz transcranial mouse pipeline
from DigiMouse to the Maga (UW) 4K microCT*, in
``mouse_therapy/validation_report_maga/manuscript.tex`` (sections cited
below as §N).

Scan (manuscript §1, Table 1)
-----------------------------
* Source: Maga lab, University of Washington (Bruker Skyscan 1272,
  4K detector, 70 kV / 142 uA, Al 0.5 mm filter, 0.1 deg rotation
  steps over 187.8 deg, NRecon on GPU at 6.127 um pitch).
* 3882 axial TIFFs of 2596 x 1908 uint16; ~38 GB uint16 stack.
* Dry skull only: no soft tissue, no in-situ brain. Endocranial
  cavity contains air, not saturated tissue (cf. DigiMouse, which
  needed a 1.2 mm skull-shell distance constraint).
* No hydroxyapatite phantom calibration; intensities are raw NRecon
  uint16 counts ("kcts" = 10^3 counts in manuscript notation), not
  Hounsfield units (manuscript §8).

Wavelength criterion (manuscript Table 2)
-----------------------------------------
At 15 MHz, cortical-bone wavelength is ~176 um (c_bone = 2640 m/s).
The 25 um working cache samples ~7 voxels per wavelength vs ~1.76 in
the 100 um DigiMouse cache -- a ~4x sampling improvement.

Atlas (manuscript §2)
---------------------
Allen Common Coordinate Framework v3 (CCFv3), averaged in-vivo
mouse template from ~1,675 specimens, ~670 hierarchical structures.
Ships at 10, 25, 50 um isotropic; this module binds to the 10 um
companion (warped products) and 25 um companion (SyN moving image).
See :class:`tuba.atlases.allen_ccfv3.AllenCCFv3`.

Pipeline stages (manuscript §3-9)
---------------------------------
* §3  Orientation determination from per-slice bone-mass profile +
      native orthoslice QC: ``SLICE_FLIP``, ``ROW_FLIP``, ``COL_FLIP``.
* §4  Streaming block-max downsampling to a 200 um ANTs cache
      (factor 33) and a 50 um intermediate cache (factor 8).
* §5  PCA-based rigid pre-alignment on ``ALIGN_PCA_THRESH = 30000``
      cortical-bone cloud, yielding ``ALIGN_R{X,Y,Z}_DEG``. Cortical-
      bone centroid translated to world origin; brain cavity sits at
      (0, 0, 0). Output padded by 396 voxels at native, 12 voxels at
      200 um so foramen-magnum end is not clipped (§6 caudal seal
      becomes unreliable if it is).
* §6  Endocranial-cavity extraction at the 24.5 um cache (factor-4
      block-max from aligned native, matching Allen's 25 um grid):
      hysteresis-threshold shell with closing (not dilation), caudal
      seal across most-caudal cortical-bone slice. Recovered cavity
      ~450 mm^3 lands in the middle of the published 420-550 mm^3
      adult C57BL/6 cranial-cavity range.
* §7  ANTs SyN (cavity moving, Allen 25 um brain mask fixed; warped
      Allen products sourced from 10 um for output detail).
* §8  Piecewise-linear intensity -> (c, rho) ramp with endpoints
      anchored on histogram-derived intensities: ``I_low = BONE_LOW
      = 5000`` (water/soft-tissue), ``I_high = 25000`` (cortical-bone
      histogram mode). Cavity infill with parenchymal (c, rho) =
      (1540 m/s, 1040 kg/m^3).
* §9  Focal-prediction comparison vs DigiMouse: at f0 = 15 MHz, f/1
      bowl on cavity centroid, calibrated to MI=1.9, the Maga skull
      shows -0.62 dB relative transmission (~7% more electrical drive
      required); focal-spot FWHM and z-offset both at or below
      sub-grid level.

CC-target naming (open in manuscript)
-------------------------------------
The runtime default in the driver is ``'cavity_centre'``: centroid of
the extracted cavity mask. Manuscript §9 reports the focal peak lands
at Allen world (+0.06, -0.76, +0.24) mm -- essentially at Bregma in
the midline, within the corpus-callosum / thalamus boundary region
targeted by the f/1 bowl placement. The cavity centroid co-locates
with the Allen CC body to within ~1 mm; ``'CC_left_body'`` is also
accepted and resolves to the warped Allen CC-body annotation, left
hemisphere. See :func:`place_on_skull` for the resolution mechanics.
"""
import os
import glob
import json
import numpy as np
import nibabel as nib

from tuba.core import (align, cavity as cav, downsample, frame,
                       placement as plc, qc, slab, surface, warp)
from tuba.io.tiff_stack import load_stack_to_array
from tuba.atlases.allen_ccfv3 import AllenCCFv3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Output dir for cached caches/transforms/QC. Configurable via the
# ``TUBA_MOUSE_REG_DIR`` environment variable; falls back to
# ``~/.cache/tuba/mouse/registration``. The legacy ``mouse_therapy/
# registration/`` layout is preserved by setting the env var to that
# location, which lets the regression tests reuse existing on-disk
# products.
REG_DIR = os.environ.get(
    'TUBA_MOUSE_REG_DIR',
    os.path.expanduser('~/.cache/tuba/mouse/registration'))

# Raw Maga 4K TIFF stack location. Configurable via
# ``TUBA_MOUSE_SOURCE_DIR``; falls back to a placeholder under
# ``~/.cache/tuba/mouse/source/`` that doesn't exist until staged
# (intentional — see the ``mouse.skull`` entry in
# :file:`tuba/data/sources.toml` for the upstream download URL).
SOURCE_DIR = os.environ.get(
    'TUBA_MOUSE_SOURCE_DIR',
    os.path.expanduser('~/.cache/tuba/mouse/source'))
TIFF_GLOB = os.path.join(SOURCE_DIR, 'dry_skull_4K_12.5micron__rec0*.tif')

# Cached intermediate volumes (Maga aligned-native world frame).
MAGA_RAS_NIFTI            = os.path.join(REG_DIR, 'maga_skull_200um.nii.gz')
MAGA_RAS_NIFTI_50UM       = os.path.join(REG_DIR, 'maga_skull_50um.nii.gz')
MAGA_RAS_NIFTI_ALIGNED    = os.path.join(REG_DIR, 'maga_skull_aligned_200um.nii.gz')
MAGA_RAS_NATIVE_ALIGNED   = os.path.join(REG_DIR, 'maga_skull_aligned_native.nii.gz')
MAGA_RAS_ALIGNED_25UM     = os.path.join(REG_DIR, 'maga_skull_aligned_25um.nii.gz')
MAGA_RAS_ALIGNED_200UM    = os.path.join(REG_DIR, 'maga_skull_aligned_native_200um.nii.gz')

CAVITY_NII   = os.path.join(REG_DIR, 'maga_cranial_cavity.nii.gz')
CAVITY_QC    = os.path.join(REG_DIR, 'maga_cranial_cavity_qc.png')
ANNOT_NII    = os.path.join(REG_DIR, 'allen_annotation_in_maga.nii.gz')

# Allen CCFv3 binding (atlas files live alongside the registration
# outputs in mouse_therapy/registration/).
ATLAS = AllenCCFv3(atlas_dir=REG_DIR)

# ---------------------------------------------------------------------------
# Physical parameters (manuscript Table 1 + §3 + §5 + §6 + §8)
# ---------------------------------------------------------------------------
NATIVE_VOXEL_MM     = 0.006127   # rec.log "Pixel Size (um)=6.12669"; §1
DOWNSAMPLE_VOXEL_MM = 0.200      # ANTs cache (33x decimation -> 202.2 um); §4

# Storage-axis flips (manuscript §3, disambiguated by orthoslice QC since
# the bone-mass profile alone is symmetric to slice-direction flip).
SLICE_FLIP = True     # storage slice 0 = caudal -> flip so +y = anterior
ROW_FLIP   = False    # L-R: skull is approximately symmetric; default
                      # left; any residual mirror is caught by Allen SyN
COL_FLIP   = True     # storage k=0 has zero bone; high-k k=52..56 has
                      # 272..41 cortical voxels = basicranium (ventral).
                      # Flip so +z = dorsal in RAS.
AXIS_FLIP_KWARGS = {'slice_flip': SLICE_FLIP, 'row_flip': ROW_FLIP,
                    'col_flip': COL_FLIP}
VOXEL_SIGNS = (-1, -1, +1)   # storage (i=R, j=A, k=V) -> RAS world

# Intensity thresholds (uint16 NRecon counts; manuscript §6, Fig. 4).
# The histogram resolves three populations: air (<100, 19% of voxels),
# mounting material at 4-8 kcts (50%), cortical bone at 20-40 kcts (9%).
BONE_THRESH      = 8000     # orientation-probe threshold (§3)
ALIGN_PCA_THRESH = 30000    # §5: above mount AND diffuse vault, so PCA
                            # covariance is dominated by the bilateral
                            # dental arches + basicranial plates
BONE_HIGH        = 15000    # §6: above cortical bone exclusion. Voxels
                            # above this are excluded from the cavity.
BONE_LOW         = 5000     # §6: shell seed (and §8 ramp lower endpoint).
                            # Lowered from legacy 6000 because the
                            # rotation-interpolated cache softens cortical
                            # intensities by ~25% (order-1 interpolation);
                            # 14.3% above 5000 in new cache matches 10.4%
                            # above 6000 in raw-TIFF block-max cache.

# Cavity extraction (manuscript §6).
CLOSE_MM        = 1.0       # closing seals foramina up to ~2 mm diameter
                            # (cribriform plate, optic canals, ear bullae,
                            # sutural gaps). 0.6 mm gave 567 mm^3 (cavity
                            # leaks rostrally into nasal cavity); 1.0 mm
                            # gives 450 mm^3, well within the published
                            # 420-550 mm^3 adult C57BL/6 range.
CAUDAL_SEAL_MM  = 0.6       # foramen-magnum plug over most-caudal
                            # cortical-bone (i, k) bounding box

# Pre-alignment rotation (manuscript §5, intrinsic ZYX, PCA-derived).
# Eigenvalue ordering by anatomical extent: largest -> A-P (23.8 mm),
# middle -> R-L (15.9 mm), smallest -> D-V (11.7 mm). Dorsal vault apex
# lands within +/- 0.5 voxel of the midsagittal plane after rotation.
ALIGN_RX_DEG = 8.07
ALIGN_RY_DEG = 8.40
ALIGN_RZ_DEG = -19.51

# Output padding for native rotation (manuscript §5): 396 voxels at
# 6.127 um keeps the caudal extreme of the skull inside the bounding
# box so the §6 caudal seal stays reliable.
NATIVE_PAD_VOXELS = 396

# Slab loader: piecewise-linear intensity -> (c, rho) ramp (manuscript §8).
# Endpoints anchored on histogram-derived intensities (BONE_LOW = §6 seed,
# I_high = cortical-bone histogram mode), not on a numerically arbitrary
# saturation. Sacrifices absolute calibration -- no HU phantom in the
# release -- for a deterministic, reproducible mapping with no free
# parameters once the histogram is fixed.
SLAB_RAMP = slab.IntensityToAcousticRamp(
    i_low=5000.0, i_high=25000.0,
    c_water=1540.0, rho_water=1000.0,
    c_bone_max=2900.0, rho_bone_max=1900.0,
)
# Cavity infill: the §6 cavity mask contains air in the source scan;
# §8 populates it with brain parenchymal properties. No skull-distance
# constraint needed (cf. DigiMouse), because dry-skull air and bone are
# already well-separated by intensity.
SLAB_CAVITY_INFILL = slab.CavityInfill(c_parenchyma=1540.0, rho_parenchyma=1040.0)

# Attenuation (manuscript §8, adopted unchanged from DigiMouse):
# alpha_bone = 8 dB cm^-1 MHz^-1, alpha_tissue = 0.5 dB cm^-1 MHz^-1,
# linear frequency dependence. Validation at 15 MHz would require a
# transmission measurement not in the source release; results are
# therefore the *relative* effect of substituting the Maga skull for
# the DigiMouse skull at fixed acoustic-property choices.
ALPHA_BONE_DB_PER_CM_PER_MHZ   = 8.0
ALPHA_TISSUE_DB_PER_CM_PER_MHZ = 0.5

# Registration prefix (used by tuba.core.warp to name transform files).
SYN_PREFIX = 'maga_cavity_to_allen'


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def _tiff_paths():
    paths = sorted(glob.glob(TIFF_GLOB))
    if not paths:
        raise FileNotFoundError(f'No TIFFs match {TIFF_GLOB}')
    return paths


def downsample_to_200um(force=False, verbose=True):
    """Stream native TIFFs into a 200 um RAS NIfTI cache (33x decimation)."""
    return downsample.downsample_tiff_stack_to_nifti(
        _tiff_paths(),
        target_voxel_mm=DOWNSAMPLE_VOXEL_MM,
        native_voxel_mm=NATIVE_VOXEL_MM,
        out_path=MAGA_RAS_NIFTI,
        axis_flip_kwargs=AXIS_FLIP_KWARGS,
        voxel_signs=VOXEL_SIGNS,
        force=force, verbose=verbose,
    )


def downsample_to_50um(force=False, verbose=True):
    """50 um cache: ~8x decimation from native, ~6 GB float32."""
    return downsample.downsample_tiff_stack_to_nifti(
        _tiff_paths(),
        target_voxel_mm=0.050,
        native_voxel_mm=NATIVE_VOXEL_MM,
        out_path=MAGA_RAS_NIFTI_50UM,
        axis_flip_kwargs=AXIS_FLIP_KWARGS,
        voxel_signs=VOXEL_SIGNS,
        force=force, verbose=verbose,
    )


def find_alignment_rotation(in_path=None, bone_thresh=ALIGN_PCA_THRESH,
                            verbose=True):
    """PCA-based rotation angles for the 200 um cache."""
    if in_path is None:
        in_path = MAGA_RAS_NIFTI
    img = nib.load(in_path)
    vol = img.get_fdata()
    return align.find_pca_rotation(vol, bone_thresh=bone_thresh, verbose=verbose)


def align_skull_to_axes(in_path=None, out_path=None,
                        ax_deg=ALIGN_RX_DEG, ay_deg=ALIGN_RY_DEG,
                        az_deg=ALIGN_RZ_DEG, force=False, order=1,
                        bone_thresh=ALIGN_PCA_THRESH, pad_voxels=12,
                        verbose=True):
    """Rotate the 200 um cache about the bone centroid."""
    return align.align_nifti_to_axes(
        in_path=in_path or MAGA_RAS_NIFTI,
        out_path=out_path or MAGA_RAS_NIFTI_ALIGNED,
        ax_deg=ax_deg, ay_deg=ay_deg, az_deg=az_deg,
        bone_thresh=bone_thresh, pad_voxels=pad_voxels,
        voxel_signs=VOXEL_SIGNS, order=order, force=force, verbose=verbose,
    )


def align_native_to_axes(out_path=None,
                         ax_deg=ALIGN_RX_DEG, ay_deg=ALIGN_RY_DEG,
                         az_deg=ALIGN_RZ_DEG, force=False,
                         bone_thresh=ALIGN_PCA_THRESH,
                         pad_voxels=NATIVE_PAD_VOXELS, verbose=True):
    """Stream all 3882 native TIFFs, apply axis flips, rotate about
    bone centroid, write a ``uint16`` 6.127 um NIfTI (~38 GB).

    Memory: ~3 x 38 GB peak. Wall time dominated by the single-threaded
    scipy affine_transform pass.
    """
    out_path = out_path or MAGA_RAS_NATIVE_ALIGNED
    if os.path.exists(out_path) and not force:
        return out_path
    import time
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
              f'size={flipped.nbytes/1e9:.1f} GB, t={(time.time()-t0):.0f} s')

    if verbose:
        print(f'[3/4] Rotating native volume about bone centroid '
              f'(thresh > {bone_thresh}, pad={pad_voxels})...')
    rotated = align.rotate_volume_about_centroid(
        flipped, ax_deg, ay_deg, az_deg,
        bone_thresh=bone_thresh, pad_voxels=pad_voxels,
        order=1, verbose=verbose)
    del flipped

    if verbose:
        print('[4/4] Writing NIfTI (gzip can take ~30 min)...')
    affine = frame.affine_for_shape(rotated.shape, NATIVE_VOXEL_MM,
                                    signs=VOXEL_SIGNS)
    nib.save(nib.Nifti1Image(rotated, affine), out_path)
    if verbose:
        print(f'  wrote {out_path}  ({os.path.getsize(out_path)/1e9:.2f} GB)')
        print(f'  TOTAL wall time: {(time.time()-t0)/60:.1f} min')
    return out_path


def downsample_aligned_to_25um(in_path=None, out_path=None,
                                factor=4, force=False, verbose=True):
    """Block-max factor-4 from the aligned native -> 25 um cache
    (24.508 um exactly). Used for cavity extraction and SyN."""
    align_native_to_axes()
    return downsample.downsample_nifti_by_factor(
        in_path=in_path or MAGA_RAS_NATIVE_ALIGNED,
        out_path=out_path or MAGA_RAS_ALIGNED_25UM,
        factor=factor, force=force, verbose=verbose,
    )


def downsample_aligned_to_200um(in_path=None, out_path=None,
                                 factor=33, force=False, verbose=True):
    """Block-max factor-33 from the aligned native -> 200 um cache
    (202.19 um). Shares the native's world coord frame."""
    align_native_to_axes()
    return downsample.downsample_nifti_by_factor(
        in_path=in_path or MAGA_RAS_NATIVE_ALIGNED,
        out_path=out_path or MAGA_RAS_ALIGNED_200UM,
        factor=factor, force=force, verbose=verbose,
    )


def extract_cavity(verbose=True):
    """Extract the endocranial cavity at 25 um. Writes
    ``maga_cranial_cavity.nii.gz`` and a QC PNG; returns the NIfTI path."""
    align_native_to_axes()
    aligned = downsample_aligned_to_25um(verbose=verbose)
    if verbose:
        print(f'Loading {aligned} ...')
    img = nib.load(aligned)
    arr = img.get_fdata().astype(np.float32)
    aff = img.affine
    voxel_mm = float(abs(aff[0, 0]))
    if verbose:
        print(f'  shape {arr.shape}, voxel {voxel_mm*1000:.2f} um, '
              f'intensity range [{arr.min():.0f}, {arr.max():.0f}]')
    cavity_mask, info = cav.extract_cavity(
        vol=arr, voxel_mm=voxel_mm,
        bone_high=BONE_HIGH, bone_low=BONE_LOW,
        close_mm=CLOSE_MM, caudal_seal_mm=CAUDAL_SEAL_MM,
        verbose=verbose)
    nib.save(nib.Nifti1Image(cavity_mask.astype(np.uint8), aff), CAVITY_NII)
    if verbose:
        print(f'wrote {CAVITY_NII}')
    qc.cavity_orthoslice_qc(
        arr=arr, cavity=cavity_mask, affine=aff, voxel_mm=voxel_mm,
        out_path=CAVITY_QC,
        title=(f'Maga cranial cavity extraction at {voxel_mm*1000:.1f} um '
               f'(volume {info["cavity_mm3"]:.0f} mm$^3$)'))
    return CAVITY_NII


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def register_to_atlas(verbose=True):
    """ANTs SyN: Maga cavity (moving) -> Allen brain template (fixed).
    Saves transforms to REG_DIR with prefix ``maga_cavity_to_allen``.

    After registration, the inverse warps are applied to bring the
    Allen template + annotation back into Maga 25 um skull space.
    """
    extract_cavity(verbose=verbose)
    if verbose:
        print('Registering Maga cavity -> Allen brain template (SyN)...')
    ATLAS.derive_brain_mask(verbose=verbose)
    fwd_stable, inv_stable, paths = warp.register_subject_to_atlas(
        subject_path=CAVITY_NII,
        atlas_path=ATLAS.brain_mask_path,
        reg_dir=REG_DIR, prefix=SYN_PREFIX,
        mode='cavity_binary',
        syn_metric='meansquares', aff_metric='meansquares',
        verbose=verbose)
    if verbose:
        print(f'  saved fwd transforms: {fwd_stable}')
        print(f'  saved inv transforms: {inv_stable}')
    # Warp 10 um Allen products back into Maga 25 um space
    warp.warp_atlas_into_subject(
        atlas_volume_path=ATLAS.template_10um,
        subject_reference_path=MAGA_RAS_ALIGNED_25UM,
        reg_dir=REG_DIR, prefix=SYN_PREFIX,
        out_path=os.path.join(REG_DIR, 'allen_template_in_maga.nii.gz'),
        interpolator='linear', verbose=verbose)
    warp.warp_atlas_into_subject(
        atlas_volume_path=ATLAS.annotation_10um,
        subject_reference_path=MAGA_RAS_ALIGNED_25UM,
        reg_dir=REG_DIR, prefix=SYN_PREFIX,
        out_path=ANNOT_NII,
        interpolator='nearestNeighbor', verbose=verbose)


def warp_atlas_into_maga(verbose=True):
    """Apply the saved inverse warp to bring Allen products into Maga
    25 um space. Standalone-callable after ``register_to_atlas`` has
    been run."""
    warp.warp_atlas_into_subject(
        atlas_volume_path=ATLAS.template_10um,
        subject_reference_path=MAGA_RAS_ALIGNED_25UM,
        reg_dir=REG_DIR, prefix=SYN_PREFIX,
        out_path=os.path.join(REG_DIR, 'allen_template_in_maga.nii.gz'),
        interpolator='linear', verbose=verbose)
    warp.warp_atlas_into_subject(
        atlas_volume_path=ATLAS.annotation_10um,
        subject_reference_path=MAGA_RAS_ALIGNED_25UM,
        reg_dir=REG_DIR, prefix=SYN_PREFIX,
        out_path=ANNOT_NII,
        interpolator='nearestNeighbor', verbose=verbose)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------
def _cc_left_centroid_world_mm():
    """Centroid of the warped Allen CC-body in the mouse's left
    hemisphere (Maga RAS world-mm, x < 0)."""
    img = nib.load(ANNOT_NII)
    arr = np.asarray(img.dataobj)
    aff = img.affine
    cc_body = ATLAS.structure_id('CC_body')
    cc_top  = ATLAS.structure_id('CC_top')
    cc = (arr == cc_body) | (arr == cc_top)
    ijk = np.argwhere(cc)
    if len(ijk) == 0:
        raise RuntimeError(f'No CC voxels in {ANNOT_NII}')
    ras = (aff[:3, :3] @ ijk.T).T + aff[:3, 3]
    left = ras[ras[:, 0] < 0]
    if len(left) == 0:
        raise RuntimeError('No left-hemisphere CC voxels found')
    return left.mean(axis=0), int(len(left))


def place_on_skull(target_name='cavity_centre',
                   scalp_clearance_mm=None,
                   apex_to_target_mm=None,
                   beam_tilt_deg_yz=0.0,
                   lateral_search_mm=3.0,
                   verbose=True):
    """Place a focused bowl on the Maga dry skull aimed at a brain target.

    Parameters
    ----------
    target_name : {'cavity_centre', 'CC_left_body'}
        - ``'cavity_centre'`` (default): cavity-mask centroid in the
          aligned-native frame. This is what the validation manuscript
          (``mouse_therapy/validation_report_maga/manuscript.tex``) and
          the runtime driver in ``run_mouse_cc.py`` actually use; it
          co-locates with the Allen CC body to within ~1 mm.
        - ``'CC_left_body'``: warped Allen CC-body-left annotation
          centroid (atlas-true target). Useful for cross-species target
          comparisons; the legacy default before the manuscript audit.
    scalp_clearance_mm : float, optional
        Water column above outer bone. Ignored if ``apex_to_target_mm``
        is set.
    apex_to_target_mm : float, optional
        Euclidean apex-to-target distance constraint (geometric focus).
    beam_tilt_deg_yz : float
        Sagittal-plane tilt about the cavity centroid.
    lateral_search_mm : float
        Apex search cylinder radius.

    Returns
    -------
    placement : dict (compatible with the legacy
        ``place_on_maga_skull`` return value).
    """
    if target_name not in ('CC_left_body', 'cavity_centre'):
        raise ValueError(f'Unsupported target {target_name!r}')

    if verbose:
        print(f'Placing bowl on Maga skull for target {target_name!r}')

    if target_name == 'CC_left_body':
        target_world, n_vox = _cc_left_centroid_world_mm()
        if verbose:
            print(f'  CC-left voxels (warped annotation): {n_vox}')
    else:
        target_world = cav.cavity_centroid_world_mm(CAVITY_NII)

    if verbose:
        print('  building outer bone surface from 25 um aligned cache...')
    surf = surface.outer_bone_surface_world_mm(
        skull_path=MAGA_RAS_ALIGNED_25UM,
        bone_thresh=BONE_LOW,
        close_mm=1.0,
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
        frame_name='maga_aligned_native_RAS',
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Slab loader
# ---------------------------------------------------------------------------
def load_slab(apex_world_mm, beam_3d, slab_size_m, dx_target_m,
              cache_path=None, cavity_mask_path=None,
              chatter=True):
    """Extract a beam-aligned slab from the Maga 4K skull at the
    25 um aligned-native cache (default).

    Returns ``(c_map, rho_map, dz_slab, frame_dict)``.
    """
    return slab.load_slab(
        cache_path=cache_path or MAGA_RAS_ALIGNED_25UM,
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
# Back-compat aliases (legacy function names)
# ---------------------------------------------------------------------------
place_on_maga_skull = place_on_skull
load_maga_skull_slab_3d = load_slab


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'place':
        tilt = float(os.environ.get('MOUSE_CC_TILT_DEG', 20.0))
        target = os.environ.get('MOUSE_CC_TARGET', 'cavity_centre')
        p = place_on_skull(target_name=target,
                            apex_to_target_mm=20.0,
                            beam_tilt_deg_yz=tilt)
        out = os.path.join(REG_DIR, 'maga_placement.json')
        with open(out, 'w') as f:
            json.dump(p, f, indent=2)
        print(json.dumps(p, indent=2))
        print(f'wrote {out}')
    elif len(sys.argv) > 1 and sys.argv[1] == 'cavity':
        extract_cavity()
    elif len(sys.argv) > 1 and sys.argv[1] == 'register':
        register_to_atlas()
    else:
        print('Usage: python -m tuba.species.mouse {place|cavity|register}')
