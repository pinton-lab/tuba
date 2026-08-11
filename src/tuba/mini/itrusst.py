"""``tuba.mini`` binding: the ITRUSST benchmark skull with a TUBA atlas.

A self-contained, lightweight counterpart to the four full TUBA species
pipelines. Where those register a multi-GB subject skull microCT to a
species atlas, ``tuba.mini`` takes the small, openly-hosted **ITRUSST
benchmark skull** (two ~6 MB STL surfaces) and gives it brain atlasing by
registering the **MNI152** template into its endocranial cavity -- the
same ``cavity_binary`` SyN path the mouse/macaque bindings use.

Pipeline (all cached under ``$TUBA_MINI_DIR`` = ``~/.cache/tuba/mini``):

    fetch STLs + MNI152          (tuba.mini.fetch)
      -> rasterize bone+cavity   (tuba.mini.skull)   1 mm RAS NIfTI
      -> SyN cavity <-> MNI mask  (tuba.mini.register)
      -> warp MNI brain mask + a parcellation onto the skull grid
      -> named MNI targets (S1, dentate, ...) via ANATOMICAL_TARGETS_MNI

The heavy ``antspyx`` dependency is needed only to *build* the transforms
(:func:`build`); consuming the cached warped products is pure
numpy/nibabel.
"""
import os
import numpy as np

from tuba.atlases.mni152 import (ANATOMICAL_TARGETS_MNI,
                                  EEG_SITES_MNI, get_parcellation)
from tuba.core import placement as plc, slab
from tuba.core.slab import IntensityToAcousticRamp
from tuba.mini import fetch, skull, register


# ---------------------------------------------------------------------------
# Paths (all under $TUBA_MINI_DIR)
# ---------------------------------------------------------------------------
def reg_dir():
    d = os.path.join(fetch.mini_dir(), 'registration')
    os.makedirs(d, exist_ok=True)
    return d


BONE_NII = 'itrusst_bone_1mm.nii.gz'
CAVITY_NII = 'itrusst_cavity_1mm.nii.gz'
BRAIN_IN_SKULL = 'mni_brainmask_in_itrusst.nii.gz'
PARC_IN_SKULL = 'parc_in_itrusst.nii.gz'

# Physical parameters (500 kHz benchmark; homogeneous cortical bone as in
# the ITRUSST intercomparison whole-skull case).
PITCH_MM = 1.0
C_BONE = 2800.0    # m/s
C_WATER = 1500.0
RHO_BONE = 1900.0  # kg/m^3
RHO_WATER = 1000.0

# Transducer bowl geometry -- the ITRUSST 500 kHz benchmark focused bowl
# (64 mm radius of curvature, 64 mm aperture diameter -> R = 32 mm).
DEFAULT_FOCAL_LENGTH_MM = 64.0
DEFAULT_BOWL_RADIUS_MM = 32.0
DEFAULT_EEG_SEARCH_RADIUS_MM = 12.0

# Beam-aligned acoustic slab (SI units): 30x30 mm aperture footprint,
# 80 mm along the beam (water -> skull -> focus), ~0.5 mm (6 PPW @ 500 kHz).
SLAB_SIZE_M = (0.03, 0.03, 0.08)
SLAB_DX_M = 0.5e-3


def paths():
    d = reg_dir()
    return {
        'bone': os.path.join(d, BONE_NII),
        'cavity': os.path.join(d, CAVITY_NII),
        'brain_in_skull': os.path.join(d, BRAIN_IN_SKULL),
        'parc_in_skull': os.path.join(d, PARC_IN_SKULL),
        'reg_dir': d,
    }


def mni_paths(verbose=False):
    """Resolve the MNI152 ICBM 2009a T1 + brain-mask paths (nilearn),
    fetching on first use. Returns ``{'dir', 't1', 'mask'}``."""
    return fetch.fetch_mni152(verbose=verbose)


# ---------------------------------------------------------------------------
# Build (needs antspyx)
# ---------------------------------------------------------------------------
def build_skull(pitch_mm=PITCH_MM, force=False, verbose=True):
    """Fetch the ITRUSST STLs and rasterize bone + cavity NIfTIs.
    Returns the paths dict. ANTs not required."""
    p = paths()
    if (not force and os.path.exists(p['bone']) and os.path.exists(p['cavity'])):
        if verbose:
            print(f'  using cached {p["bone"]}, {p["cavity"]}')
        return p
    inner, outer = fetch.fetch_itrusst_skull(verbose=verbose)
    masks = skull.rasterize(inner, outer, pitch_mm=pitch_mm, verbose=verbose)
    skull.save_masks(masks, p['bone'], p['cavity'], verbose=verbose)
    return p


def build(parcellation='harvard_oxford_117', force=False, verbose=True):
    """Full build: skull masks -> MNI152 -> SyN -> warp brain mask +
    parcellation onto the skull grid. Needs ``antspyx``.

    Returns the paths dict (with the warped products populated).
    """
    p = build_skull(force=force, verbose=verbose)
    mp = mni_paths(verbose=verbose)

    inv_txt = os.path.join(p['reg_dir'], f'{register.PREFIX}_syn_inv.txt')
    if force or not os.path.exists(inv_txt):
        if verbose:
            print('\n[register] cavity <-> MNI brain mask (SyN, cavity_binary)')
        register.register(p['cavity'], mp['mask'], p['reg_dir'],
                          verbose=verbose)
    elif verbose:
        print(f'\n[register] using cached transforms ({inv_txt})')

    if verbose:
        print('\n[warp] MNI brain mask -> skull grid')
    register.warp_into_skull(mp['mask'], p['bone'], p['reg_dir'],
                             p['brain_in_skull'], interpolator='nearestNeighbor',
                             verbose=verbose)

    parc = get_parcellation(parcellation)
    if os.path.exists(parc.path):
        if verbose:
            print(f'\n[warp] parcellation {parc.name!r} -> skull grid')
        register.warp_into_skull(parc.path, p['bone'], p['reg_dir'],
                                 p['parc_in_skull'],
                                 interpolator='nearestNeighbor',
                                 verbose=verbose)
    elif verbose:
        print(f'\n[warp] parcellation {parcellation!r} not staged at '
              f'{parc.path}; skipping (brain mask still warped).')
    return p


def target_ras(name):
    """MNI RAS-mm coordinate of a named anatomical target. Because the
    ITRUSST skull is authored in MNI-aligned RAS, this coordinate is also
    (to within the SyN refinement) the skull-space location; for exact
    skull-space placement warp a marker with
    :func:`tuba.mini.register.mni_point_marker`."""
    return ANATOMICAL_TARGETS_MNI[name]


def eeg_ras(name):
    """MNI RAS-mm coordinate of a named EEG 10-20 site -- e.g. to
    constrain a perpendicular-to-skull placement to the region around an
    experimental scalp location (see the human pipeline's
    ``place_on_skull(scalp_site=...)``)."""
    return EEG_SITES_MNI[name]


# ---------------------------------------------------------------------------
# Placement + acoustic slab (tuba.core.placement / tuba.core.slab)
# ---------------------------------------------------------------------------
def outer_surface(verbose=True):
    """Outer skull surface as ``(verts, outward_normals)`` in RAS world mm.

    Uses the ITRUSST outer STL directly (watertight, winding-consistent),
    so the normals are the true outer-table normals rather than a raycast
    approximation. This is the surface :func:`place` searches."""
    trimesh = skull._require_trimesh()
    _, outer_stl = fetch.fetch_itrusst_skull(verbose=False)
    mesh = trimesh.load(outer_stl)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    # Enforce outward orientation (normals pointing away from the centroid).
    if np.einsum('ij,ij->i', normals, verts - verts.mean(0)).mean() < 0:
        normals = -normals
    if verbose:
        print(f'  outer surface: {len(verts):,} verts')
    return verts, normals


def place(target_name=None, target_ras_mm=None, scalp_site=None,
          surface=None, focal_length_mm=DEFAULT_FOCAL_LENGTH_MM,
          bowl_radius_mm=DEFAULT_BOWL_RADIUS_MM,
          eeg_search_radius_mm=DEFAULT_EEG_SEARCH_RADIUS_MM, verbose=True):
    """Place a focused bowl perpendicular to the outer skull, aimed at an
    MNI target (also the skull-space coordinate, the skull being
    MNI-aligned) via :func:`tuba.core.placement.place_perpendicular_to_surface`.

    Parameters
    ----------
    target_name : str, optional
        Name in :data:`ANATOMICAL_TARGETS_MNI` (e.g. ``'M1_left'``).
    target_ras_mm : (3,) array-like, optional
        Explicit RAS-mm target (overrides ``target_name``).
    scalp_site : str, optional
        EEG 10-20 site (e.g. ``'C3'``) constraining the apex search to a
        sphere around that scalp location.
    surface : (verts, normals), optional
        Skip the STL fetch and use a supplied outer surface (for tests).

    Returns the tuba.core.placement dict (apex ``xdc_center_lps``, inward
    ``beam_dir_3d``, ``focus_lps``, ``perp_residual_mm``,
    ``focus_to_target_mm``, ...); coordinates are RAS world mm.
    """
    if target_ras_mm is None:
        if target_name is None:
            raise ValueError('need target_name or target_ras_mm')
        target_ras_mm = target_ras(target_name)
    verts, normals = (surface if surface is not None
                      else outer_surface(verbose=verbose))
    seed = eeg_ras(scalp_site) if scalp_site else None
    return plc.place_perpendicular_to_surface(
        target_world_mm=target_ras_mm,
        surface_world_mm=verts, surface_normals_world_mm=normals,
        eeg_seed_world_mm=seed, eeg_search_radius_mm=eeg_search_radius_mm,
        focal_length_mm=focal_length_mm, bowl_radius_mm=bowl_radius_mm,
        target_name=target_name, frame_name='itrusst_ras_world_mm',
        verbose=verbose)


def load_slab(placement, slab_size_m=SLAB_SIZE_M, dx_target_m=SLAB_DX_M,
              verbose=True):
    """Extract a beam-aligned ``(c, rho)`` acoustic slab from the 1 mm bone
    mask along the placement's beam, via :func:`tuba.core.slab.load_slab`.

    The ITRUSST benchmark medium is homogeneous (cortical bone in water),
    so a one-sided ramp maps the mask straight to ``(c_bone, c_water)`` --
    no cavity/parenchyma infill (unlike the dry-skull species pipelines).
    Returns ``(c_map, rho_map, dz_slab, frame)``.
    """
    p = paths()
    ramp = IntensityToAcousticRamp(
        i_low=0.5, i_high=1.0,
        c_water=C_WATER, rho_water=RHO_WATER,
        c_bone_max=C_BONE, rho_bone_max=RHO_BONE)
    return slab.load_slab(
        cache_path=p['bone'],
        apex_world_mm=placement['xdc_center_lps'],
        beam_3d=placement['beam_dir_3d'],
        slab_size_m=slab_size_m, dx_target_m=dx_target_m,
        ramp=ramp, verbose=verbose)
