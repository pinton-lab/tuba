"""Species-agnostic primitives for TUBA.

This package contains the four-stage pipeline machinery shared by every
species binding:

    orientation/align  →  downsample  →  cavity-or-surface  →  ANTs SyN

plus the slab loader and placement geometry used downstream by acoustic
solvers. All public APIs operate in **world-mm** (RAS, origin at the
species' aligned-native bone-centroid). No module here imports any
species-specific constants — that responsibility lives in
:mod:`tuba.species`.

Submodules
----------
align
    PCA + axis-flip rigid pre-alignment of raw skull volumes. Threshold is
    a parameter; sign-ambiguity is resolved by an optional axis-sign
    triple or by an "auto" mode that uses bone-mass profile asymmetry.

downsample
    Block-max factor-N caching with NIfTI-affine propagation. Caches are
    always derived from the *aligned* native volume to avoid the
    block-max smearing bias that biases the bone centroid ventrally when
    PCA is run on an independently-coarsened cache.

cavity
    Morphological-closing extractor that seals foramina without shifting
    the inner bone surface, stops at ``bone_low`` (extending to
    ``bone_high`` leaks through sutures), and runs at
    *atlas-matching resolution* (mouse 25 µm, macaque 250 µm). Optional
    seal regions (caudal, basicranium-floor, rostral/orbital plug)
    parameterise the macaque's larger foramina without subclassing.

surface
    Outer-bone envelope extractor. Two methods: ``"close+fill"`` (large
    morphological closing, used by the human Halle pipeline at 22 mm) and
    ``"raycast"`` (ray-cast from skull centroid; gives a different
    envelope near orbits and is used by mouse/macaque placement helpers
    for apex search).

slab
    Beam-aligned slab sampler. Geometry is
    ``apex + L·t1 + E·t2 + d·beam`` where ``(t1, t2)`` is built by
    Gram-Schmidt from ``beam`` and a seed axis ([0,0,1] unless the beam
    is near-vertical, then [1,0,0]). All inputs in world-mm.

placement
    Apex search on the outer-bone surface within a cylinder around the
    target's lateral position; then lifts the apex via ``scalp_clearance_mm``
    or solves Pythagorean geometry for ``apex_to_target_mm``; optionally
    rotates apex and target about the cavity centroid by
    ``beam_tilt_deg_yz``.

warp
    ANTs SyN wrapper. The convention is frozen as
    ``fixed=atlas, moving=subject``; users interact only with
    :func:`warp_atlas_into_subject` and :func:`warp_subject_into_atlas`,
    never with raw forward/inverse transform lists. Two registration
    modes are supported, switched by the atlas binding:

    - ``"cavity_binary"`` — subject cavity mask vs. atlas brain mask
      (mouse Allen CCFv3, macaque NMT v2).
    - ``"intensity"`` — subject CT vs. atlas MRI template
      (human Halle vs. MNI152).

frame
    Coordinate-frame conventions and voxel↔world helpers. Documents the
    canonical TUBA world frame (RAS, bone-centroid origin) and provides
    :class:`WorldFrame` for tagging arrays with their frame of origin so
    that mixing voxel-mm and world-mm coordinates is a type error.

qc
    Shared plot templates used by the validation reports: orthoslice
    cavity overlay, atlas-on-skull intensity composite, focal-pressure
    comparison.

Notes
-----
The post-hoc drive-calibration convention (``p0 = 1 Pa`` unit drive,
rescale to MI target post-simulation; relative skull transmission as the
unit-drive focal-pressure ratio in dB) is a downstream convention and
does not live in this package — it belongs to the acoustic solver
wrapper.
"""

# Public submodules are imported lazily by callers; we intentionally do
# not import them here so that ``import tuba.core`` succeeds even when an
# optional dependency (e.g. antspyx) is missing.

__all__ = [
    "align",
    "downsample",
    "cavity",
    "surface",
    "slab",
    "placement",
    "warp",
    "frame",
    "qc",
]
