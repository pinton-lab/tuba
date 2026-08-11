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

from tuba.atlases.mni152 import (ANATOMICAL_TARGETS_MNI,
                                  EEG_SITES_MNI, get_parcellation)
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
