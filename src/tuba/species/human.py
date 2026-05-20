"""Human (Halle dry-skull microCT) species binding for TUBA.

Halle Zenodo NRRD + MNI152 ICBM 2009a binding. Differs from the
mouse Maga and macaque pipelines in three ways:

1. **Intensity-mode atlas registration** (not cavity-binary). The MNI
   T1 template is an intensity image, not a clean brain mask, so the
   pipeline uses ANTs SyN with cross-modality mutual information
   (CT vs T1) rather than mask-to-mask SyN. The fixed image is the
   *subject* Halle CT (so the forward warp brings MNI -> Halle and
   the warped MNI products land directly on the Halle grid).
2. **Raycast outer-bone surface**. The morphological close+fill
   envelope sits inside the bone or on the inner table; ray-casting
   from the bone-mass centroid records the OUTERMOST bone hit along
   each ray, giving the proper outer table where a transducer would
   physically contact.
3. **Perpendicular-to-skull placement**. The bowl apex is placed on
   the outer skull with the beam axis along the local inward
   skull normal. The vertex picked is the one whose inward-normal
   line passes closest to the target.

Coordinate-frame migration
--------------------------
The legacy human pipeline returns placement coords in
**NRRD-voxel-mm** (origin at NRRD voxel (0, 0, 0), axes
(+i, +j, +k) = (+L, +A, +S) since the Halle NRRD is stored in LAS
voxel order despite the LPS header tag -- see
:mod:`tuba.species.human` constants below).

TUBA migrates this to **Halle RAS world** = NIfTI RAS derived from
``halle_affine_to_ras = diag(-vox, +vox, +vox)``:

* TUBA Halle RAS x = -NRRD x   (since NRRD axis 0 is patient-L, TUBA
                                 RAS +x is patient-R)
* TUBA Halle RAS y = +NRRD y   (both anterior)
* TUBA Halle RAS z = +NRRD z   (both superior)

The species' "world-mm" origin is therefore the NRRD corner (NOT the
bone centroid). This relaxes TUBA's strict "bone-centroid origin"
spec for the human case because the Halle pipeline does not go
through a PCA pre-alignment (the legacy never did one; the NRRD is
already approximately RAS-aligned in storage). Adding a PCA pre-
alignment for human is a future option; for now we keep the migration
minimal and rely on the species module to convert coords as needed.

The downstream slab loader and placement helper both accept and
return Halle RAS world-mm. A :func:`nrrd_voxel_mm_to_halle_ras`
helper is provided for callers carrying legacy coords.
"""
import os
import numpy as np

from tuba.core import (placement as plc, slab, surface as surf, warp,
                       frame as fr)
from tuba.core.slab import IntensityToAcousticRamp
from tuba.io.nrrd import read_nrrd
from tuba.atlases.mni152 import MNI152

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Output dir for cached caches/transforms/QC. Configurable via the
# ``TUBA_HUMAN_REG_DIR`` environment variable; falls back to
# ``~/.cache/tuba/human/registration``. The legacy
# ``neuromod_parameters/registration/`` layout is preserved by setting
# the env var to that location.
REG_DIR = os.environ.get(
    'TUBA_HUMAN_REG_DIR',
    os.path.expanduser('~/.cache/tuba/human/registration'))

# Halle source NRRD (Zenodo release). Configurable via
# ``TUBA_HUMAN_NRRD_PATH``; falls back to a placeholder under
# ``~/.cache/tuba/human/source/`` that doesn't exist until the user
# fetches the Zenodo blob (see the ``human.skull`` entry in
# :file:`tuba/data/sources.toml`).
NRRD_PATH = os.environ.get(
    'TUBA_HUMAN_NRRD_PATH',
    os.path.expanduser('~/.cache/tuba/human/source/halle_skull.nrrd'))

# 1 mm Halle NIfTI used as the ANTs fixed image.
HALLE_CT_1MM         = os.path.join(REG_DIR, 'halle_ct_1mm.nii.gz')
# Outer-bone-surface caches (raycast preferred; close+fill is fallback).
OUTER_BONE_SURFACE   = os.path.join(REG_DIR, 'halle_outer_bone_surface.npz')
OUTER_ENVELOPE_FALLBACK = os.path.join(REG_DIR, 'halle_outer_surface.npz')
# Warped MNI products.
MNI_BRAIN_IN_HALLE   = os.path.join(REG_DIR, 'mni_brain_in_halle.nii.gz')
MNI_T1_IN_HALLE      = os.path.join(REG_DIR, 'mni_t1_in_halle.nii.gz')

# MNI152 atlas binding. Atlas files cached under
# ``$TUBA_HUMAN_REG_DIR/../templates/mni152_icbm_2009a/`` by default
# (i.e. ``~/.cache/tuba/human/templates/mni152_icbm_2009a/``); pointing
# TUBA_HUMAN_REG_DIR at the legacy ``neuromod_parameters/registration/``
# directory preserves the legacy templates layout one level up.
ATLAS = MNI152(
    atlas_dir=os.path.join(REG_DIR, '..', 'templates', 'mni152_icbm_2009a'))

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
NATIVE_VOXEL_MM   = 0.125    # Halle NRRD pitch
DOWNSAMPLE_FACTOR = 8        # 1 mm working voxel for ANTs SyN
BONE_HU           = 700      # clinical CT bone threshold (Hounsfield units)

# Outer-bone surface (raycast).
N_RAYCAST_AZIM = 240
N_RAYCAST_ELEV = 120

# Close+fill alternative (used by extract_halle_surface.py legacy script).
CLOSE_RADIUS_MM = 22.0       # bridges foramen magnum (~30 mm) + orbits

# Slab loader: piecewise-linear intensity -> (c, rho) ramp.
# Halle CT is in Hounsfield units (clinical scale); endpoints HU=700
# (cancellous-bone lower bound) and HU=1973 (cortical-bone upper bound,
# empirically calibrated for the Halle scan).
SLAB_RAMP = IntensityToAcousticRamp(
    i_low=700.0, i_high=1973.0,
    c_water=1540.0, rho_water=1000.0,
    c_bone_max=2900.0, rho_bone_max=2200.0,
)

# Default bowl geometry (placeholder; override per-experiment).
DEFAULT_FOCAL_LENGTH_MM = 30.0
DEFAULT_BOWL_RADIUS_MM  = 15.0
DEFAULT_EEG_SEARCH_RADIUS_MM = 10.0

SYN_PREFIX = 'ants_syn'   # legacy prefix: ants_syn_forward.txt / ants_syn_inverse.txt
# Legacy filenames don't match the default {prefix}_syn_{fwd,inv}.txt
# template; pass these explicit paths to tuba.core.warp helpers.
FWD_TXT_PATH = os.path.join(REG_DIR, 'ants_syn_forward.txt')
INV_TXT_PATH = os.path.join(REG_DIR, 'ants_syn_inverse.txt')


# ---------------------------------------------------------------------------
# Coordinate-frame helpers
# ---------------------------------------------------------------------------
def halle_affine_to_ras(voxel_mm):
    """4x4 affine: Halle storage voxel index -> RAS world mm.

    The Halle NRRD is stored in LAS voxel order (+i = L, +j = A,
    +k = S). The NIfTI affine that maps this to anatomically-correct
    RAS world coords is ``diag(-vox, +vox, +vox)`` with zero origin.
    """
    return np.diag([-voxel_mm, voxel_mm, voxel_mm, 1.0])


def nrrd_voxel_mm_to_halle_ras(nrrd_xyz):
    """Legacy NRRD-voxel-mm -> TUBA Halle RAS world mm.

    NRRD-voxel-mm = (+i*vox, +j*vox, +k*vox); TUBA Halle RAS =
    (-i*vox, +j*vox, +k*vox). Migration is a sign flip on x.
    """
    arr = np.atleast_2d(np.asarray(nrrd_xyz, dtype=np.float64))
    out = np.column_stack([-arr[:, 0], arr[:, 1], arr[:, 2]])
    return out[0] if arr.shape[0] == 1 else out


def halle_ras_to_nrrd_voxel_mm(halle_ras_xyz):
    """TUBA Halle RAS world mm -> legacy NRRD-voxel-mm. Inverse of
    :func:`nrrd_voxel_mm_to_halle_ras`."""
    arr = np.atleast_2d(np.asarray(halle_ras_xyz, dtype=np.float64))
    out = np.column_stack([-arr[:, 0], arr[:, 1], arr[:, 2]])
    return out[0] if arr.shape[0] == 1 else out


def _halle_lps_to_halle_ras(halle_lps):
    """ANTs LPS world for the Halle NIfTI -> TUBA Halle RAS.

    From affine ``diag(-vox, +vox, +vox)``: NIfTI RAS = (-i*vox,
    +j*vox, +k*vox); LPS = (+i*vox, -j*vox, +k*vox). So
    RAS = (-LPS_x, -LPS_y, LPS_z).
    """
    arr = np.atleast_2d(np.asarray(halle_lps, dtype=np.float64))
    out = np.column_stack([-arr[:, 0], -arr[:, 1], arr[:, 2]])
    return out[0] if arr.shape[0] == 1 else out


def mni_ras_to_halle_ras(mni_ras_xyz, reg_dir=None, prefix=SYN_PREFIX):
    """Map a point from MNI RAS world to TUBA Halle RAS world.

    Chain: MNI RAS -> MNI LPS (negate x,y) -> ANTs inverse transform
    (intensity mode, Halle was fixed) -> Halle LPS -> Halle RAS
    (negate x,y).
    """
    if reg_dir is None:
        reg_dir = REG_DIR
    arr_in = np.asarray(mni_ras_xyz, dtype=np.float64)
    pts = np.atleast_2d(arr_in)
    mni_lps = np.column_stack([-pts[:, 0], -pts[:, 1], pts[:, 2]])
    halle_lps = warp.apply_transforms_to_points_atlas_to_subject_intensity(
        mni_lps, reg_dir=reg_dir, prefix=prefix,
        inv_txt_path=INV_TXT_PATH)
    # halle_lps is always (N, 3); convert to RAS while keeping 2D.
    halle_lps_2d = np.atleast_2d(halle_lps)
    halle_ras_2d = np.column_stack(
        [-halle_lps_2d[:, 0], -halle_lps_2d[:, 1], halle_lps_2d[:, 2]])
    if arr_in.ndim == 1:
        return halle_ras_2d[0]
    return halle_ras_2d


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def downsample_halle_to_1mm(force=False, verbose=True):
    """Block-max downsample the Halle NRRD (0.125 mm) to 1 mm,
    save as NIfTI with the LAS->RAS affine. Used as the ANTs fixed
    image."""
    import nibabel as nib
    if os.path.exists(HALLE_CT_1MM) and not force:
        if verbose:
            print(f'  using cached {HALLE_CT_1MM}')
        return HALLE_CT_1MM
    data, voxel_mm, _ = read_nrrd(NRRD_PATH, verbose=verbose)
    DS = DOWNSAMPLE_FACTOR
    sh = data.shape
    n0, n1, n2 = (s // DS for s in sh)
    trimmed = data[:n0*DS, :n1*DS, :n2*DS]
    blocks = trimmed.reshape(n0, DS, n1, DS, n2, DS)
    hu_ds = blocks.max(axis=(1, 3, 5)).astype(np.float32)
    new_voxel_mm = voxel_mm * DS
    if verbose:
        print(f'  Halle 1mm: shape={hu_ds.shape}, voxel={new_voxel_mm} mm')
    affine = halle_affine_to_ras(new_voxel_mm)
    nib.save(nib.Nifti1Image(hu_ds, affine), HALLE_CT_1MM)
    if verbose:
        print(f'  saved {HALLE_CT_1MM}')
    return HALLE_CT_1MM


def extract_outer_bone_surface(force=False, verbose=True):
    """Build the proper outer-bone-table surface via ray-casting from
    the bone-mass centroid. Saves verts + normals in Halle RAS
    world-mm to ``halle_outer_bone_surface.npz``.
    """
    if os.path.exists(OUTER_BONE_SURFACE) and not force:
        if verbose:
            print(f'  using cached {OUTER_BONE_SURFACE}')
        return OUTER_BONE_SURFACE
    data, voxel_mm, _ = read_nrrd(NRRD_PATH, verbose=verbose)
    DS = DOWNSAMPLE_FACTOR
    sh = data.shape
    n0, n1, n2 = (s // DS for s in sh)
    trimmed = data[:n0*DS, :n1*DS, :n2*DS]
    blocks = trimmed.reshape(n0, DS, n1, DS, n2, DS)
    hu_ds = blocks.max(axis=(1, 3, 5))
    new_voxel_mm = voxel_mm * DS
    bone = hu_ds >= BONE_HU
    if verbose:
        print(f'  bone fraction: {bone.mean():.3f}')
    aff_storage_to_ras = halle_affine_to_ras(new_voxel_mm)[:3, :3]
    verts_world_mm, normals_world_mm = surf.raycast_outer_bone_surface_world_mm(
        bone_mask=bone, voxel_mm=new_voxel_mm,
        affine_storage_to_world=aff_storage_to_ras,
        n_azim=N_RAYCAST_AZIM, n_elev=N_RAYCAST_ELEV,
        return_normals=True, normal_k=12,
        verbose=verbose)
    np.savez(OUTER_BONE_SURFACE,
             verts=verts_world_mm.astype(np.float32),
             vertex_normals=normals_world_mm.astype(np.float32),
             voxel_mm_downsampled=np.float32(new_voxel_mm),
             frame=np.array('halle_ras_world_mm', dtype='<U32'))
    if verbose:
        print(f'  wrote {OUTER_BONE_SURFACE}')
    return OUTER_BONE_SURFACE


def _load_outer_bone_surface_world_mm():
    """Load (verts, normals) in Halle RAS world-mm, building the cache
    if it's missing. Falls back to the legacy close+fill envelope if
    the raycast cache is unavailable."""
    if os.path.exists(OUTER_BONE_SURFACE):
        npz = np.load(OUTER_BONE_SURFACE)
        verts = np.asarray(npz['verts'], dtype=np.float64)
        normals = np.asarray(npz['vertex_normals'], dtype=np.float64)
        # Legacy raycast NPZ stores verts in NRRD-voxel-mm; new TUBA
        # cache labels its frame. Detect legacy by absence of 'frame'.
        if 'frame' not in npz.files or str(npz['frame']) != 'halle_ras_world_mm':
            verts = nrrd_voxel_mm_to_halle_ras(verts)
            normals = nrrd_voxel_mm_to_halle_ras(normals)
        return verts, normals
    if os.path.exists(OUTER_ENVELOPE_FALLBACK):
        npz = np.load(OUTER_ENVELOPE_FALLBACK)
        verts = nrrd_voxel_mm_to_halle_ras(
            np.asarray(npz['verts'], dtype=np.float64))
        normals = nrrd_voxel_mm_to_halle_ras(
            np.asarray(npz['vertex_normals'], dtype=np.float64))
        return verts, normals
    raise FileNotFoundError(
        f'No outer-bone surface cache found; run extract_outer_bone_surface()')


def register_to_atlas(verbose=True):
    """ANTs SyN registering MNI T1 (moving) into Halle CT (fixed),
    intensity mode. Saves transforms to ``REG_DIR`` with prefix
    ``ants_syn`` (matches legacy filenames). Also warps the MNI brain
    mask and T1 template into Halle space for QC.
    """
    halle_path = downsample_halle_to_1mm(verbose=verbose)
    fwd_stable, inv_stable, paths = warp.register_atlas_to_subject_intensity(
        atlas_intensity_path=ATLAS.t1_path,
        subject_intensity_path=halle_path,
        reg_dir=REG_DIR, prefix=SYN_PREFIX,
        verbose=verbose,
    )
    if verbose:
        print(f'  saved fwd transforms: {fwd_stable}')
        print(f'  saved inv transforms: {inv_stable}')
    warp.warp_atlas_image_into_subject_via_fwd(
        atlas_volume_path=ATLAS.brain_mask_path,
        subject_reference_path=halle_path,
        reg_dir=REG_DIR, prefix=SYN_PREFIX,
        out_path=MNI_BRAIN_IN_HALLE,
        interpolator='nearestNeighbor', verbose=verbose)
    warp.warp_atlas_image_into_subject_via_fwd(
        atlas_volume_path=ATLAS.t1_path,
        subject_reference_path=halle_path,
        reg_dir=REG_DIR, prefix=SYN_PREFIX,
        out_path=MNI_T1_IN_HALLE,
        interpolator='linear', verbose=verbose)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------
def place_on_skull(target_name=None, target_mni_ras=None,
                   scalp_site=None, scalp_mni_ras=None,
                   eeg_search_radius_mm=DEFAULT_EEG_SEARCH_RADIUS_MM,
                   focal_length_mm=DEFAULT_FOCAL_LENGTH_MM,
                   bowl_radius_mm=DEFAULT_BOWL_RADIUS_MM,
                   verbose=True):
    """Perpendicular-to-skull placement aimed at an MNI target.

    Parameters
    ----------
    target_name : str, optional
        Atlas target name from
        :data:`tuba.atlases.mni152.ANATOMICAL_TARGETS_MNI` (e.g.
        ``'S1_left'``). Mutually exclusive with ``target_mni_ras``.
    target_mni_ras : (3,) array-like, optional
        Explicit MNI RAS coord.
    scalp_site : str, optional
        EEG site (e.g. ``'CP3'``) from
        :data:`tuba.atlases.mni152.EEG_SITES_MNI`. Restricts the apex
        search to outer-skull vertices within ``eeg_search_radius_mm``
        of the warped site.
    scalp_mni_ras : (3,) array-like, optional
        Explicit MNI RAS coord for an EEG-like scalp seed.
    focal_length_mm, bowl_radius_mm : float
        Bowl geometry; defaults match the human-skull sweep's
        legacy halle_placement defaults.

    Returns
    -------
    placement : dict (in Halle RAS world-mm; keys mirror
        :func:`tuba.core.placement.place_perpendicular_to_surface`).
        For legacy back-compat the NRRD-voxel-mm versions of the
        positional keys are also included with a ``_nrrd_voxel_mm``
        suffix.
    """
    if target_name is not None and target_mni_ras is None:
        target_mni_ras = ATLAS.target_ras(target_name)
    if target_mni_ras is None:
        raise ValueError('Need target_name or target_mni_ras')
    if scalp_site is not None and scalp_mni_ras is None:
        scalp_mni_ras = ATLAS.eeg_ras(scalp_site)

    target_halle_ras = mni_ras_to_halle_ras(target_mni_ras)
    eeg_seed_halle_ras = (mni_ras_to_halle_ras(scalp_mni_ras)
                          if scalp_mni_ras is not None else None)

    if verbose:
        print(f'Placing bowl on Halle skull for target {target_name!r}')
        print(f'  target MNI RAS = {tuple(round(v,2) for v in target_mni_ras)}')
        print(f'  target Halle RAS = {target_halle_ras.round(3)}')
        if eeg_seed_halle_ras is not None:
            print(f'  EEG seed MNI RAS = {tuple(round(v,2) for v in scalp_mni_ras)}')
            print(f'  EEG seed Halle RAS = {eeg_seed_halle_ras.round(3)}')

    verts, normals = _load_outer_bone_surface_world_mm()

    placement = plc.place_perpendicular_to_surface(
        target_world_mm=target_halle_ras,
        surface_world_mm=verts,
        surface_normals_world_mm=normals,
        eeg_seed_world_mm=eeg_seed_halle_ras,
        eeg_search_radius_mm=eeg_search_radius_mm,
        focal_length_mm=focal_length_mm,
        bowl_radius_mm=bowl_radius_mm,
        target_name=target_name,
        frame_name='halle_ras_world_mm',
        verbose=verbose,
    )
    # Legacy back-compat: include NRRD-voxel-mm versions of positional
    # keys for callers that still operate in the old frame.
    for key in ('target_lps', 'xdc_center_lps', 'focus_lps',
                'scalp_contact_lps'):
        if placement.get(key) is not None:
            placement[f'{key}_nrrd_voxel_mm'] = tuple(
                float(v) for v in halle_ras_to_nrrd_voxel_mm(placement[key]))
    return placement


# ---------------------------------------------------------------------------
# Slab loader
# ---------------------------------------------------------------------------
def load_slab(apex_world_mm, beam_3d, slab_size_m, dx_target_m,
              nrrd_path=None, chatter=True):
    """Extract a beam-aligned slab from the Halle NRRD at native
    0.125 mm resolution.

    ``apex_world_mm`` and ``beam_3d`` are in Halle RAS world-mm
    (TUBA convention; see module docstring). Returns
    ``(c_map, rho_map, dz_slab, frame)`` matching the legacy
    ``skull_slab_3d.load_skull_slab_3d`` shape.
    """
    if nrrd_path is None:
        nrrd_path = NRRD_PATH
    data, voxel_mm, _ = read_nrrd(nrrd_path, verbose=chatter)
    # World axes in TUBA Halle RAS: axis 0 (NRRD +L) maps to RAS -x,
    # so the world coordinate along axis 0 *decreases* as the storage
    # index increases. RegularGridInterpolator accepts strictly
    # monotonic (ascending OR descending) axes.
    axes_world_mm = (
        np.arange(data.shape[0]) * (-voxel_mm),
        np.arange(data.shape[1]) * (+voxel_mm),
        np.arange(data.shape[2]) * (+voxel_mm),
    )
    return slab.sample_slab_world_mm(
        skull_array=data.astype(np.float32),
        skull_world_axes=axes_world_mm,
        apex_world_mm=apex_world_mm, beam_3d=beam_3d,
        slab_size_m=slab_size_m, dx_target_m=dx_target_m,
        ramp=SLAB_RAMP,
        cavity_array=None, cavity_world_axes=None,
        cavity_infill=None,
        verbose=chatter,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'surface':
        extract_outer_bone_surface()
    elif len(sys.argv) > 1 and sys.argv[1] == 'register':
        register_to_atlas()
    elif len(sys.argv) > 1 and sys.argv[1] == 'place':
        target = sys.argv[2] if len(sys.argv) > 2 else 'S1_left'
        scalp = sys.argv[3] if len(sys.argv) > 3 else 'CP3'
        p = place_on_skull(target_name=target, scalp_site=scalp)
        import json
        print(json.dumps(p, indent=2))
    else:
        print('Usage: python -m tuba.species.human '
              '{surface|register|place [target] [scalp_site]}')
