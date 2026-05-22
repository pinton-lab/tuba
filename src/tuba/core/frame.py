"""Coordinate-frame conventions and helpers.

TUBA's canonical world frame is **RAS with origin at the bone centroid
of the aligned-native volume** (:data:`ALIGNED_RAS_BONE_CENTROID`):

  +x = right, +y = anterior, +z = superior (dorsal)

Mouse and macaque ports follow this convention end-to-end. The human
port (Halle, :mod:`tuba.species.human`) does **not**: there is no PCA
pre-alignment step, the volume is loaded directly from the source
NRRD's LAS-tagged storage, and the world origin sits at the NRRD
corner rather than at the bone centroid. This is acknowledged with the
:data:`HALLE_NATIVE` frame tag so consumers can detect the
relaxation at the type level rather than discovering it from
documentation alone.

All public APIs in :mod:`tuba.core` accept and return world-mm. The
frame tag (when present) describes which convention the coordinates
live in. Storage-axis flip flags and voxel-sign tuples that map a
given species' raw acquisition into a world frame are species-specific
and live in :mod:`tuba.species`.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WorldFrame:
    """Tag for a coordinate system. Used to label arrays/structs with the
    frame they live in so that mixing voxel-mm and world-mm is a type
    error at the boundary."""
    name: str
    description: str = ""


# -----------------------------------------------------------------------------
# Canonical TUBA frames. Species modules import these and tag their
# returned coordinates/placements with one of these instances rather
# than emitting raw arrays. Downstream code can dispatch on
# ``placement.frame`` instead of relying on docstring comments.
# -----------------------------------------------------------------------------

ALIGNED_RAS_BONE_CENTROID = WorldFrame(
    name='aligned_ras_bone_centroid',
    description=(
        'RAS world-mm; +x right, +y anterior, +z dorsal. Origin at the '
        'bone centroid of the aligned-native volume, after PCA rigid '
        'pre-alignment. Canonical TUBA frame for mouse and macaque.'
    ),
)

HALLE_NATIVE = WorldFrame(
    name='halle_native_ras',
    description=(
        'RAS world-mm derived from the Halle NRRD by overriding the '
        'header (which tags the storage as LAS) with '
        'diag(-vox, +vox, +vox). Origin sits at the NRRD corner, not '
        'at the bone centroid -- no PCA pre-alignment is run for the '
        'human pipeline. Consumers that expect '
        '`ALIGNED_RAS_BONE_CENTROID` must apply an offset or refuse to '
        'mix the two.'
    ),
)


def affine_for_shape(shape, voxel_mm, signs=(-1, -1, +1)):
    """RAS world affine for a volume, with origin at volume centre.

    Parameters
    ----------
    shape : (3,) tuple/array of voxel counts.
    voxel_mm : isotropic voxel size in mm.
    signs : (sx, sy, sz) sign of each storage axis relative to RAS.
        Default (-1, -1, +1) matches a storage convention where i=0
        is the right side, j=0 is anterior, k=0 is ventral; with the
        origin at volume centre, world coords are RAS.
    """
    A = np.eye(4)
    for d in range(3):
        A[d, d] = signs[d] * voxel_mm
        A[d, 3] = -A[d, d] * (shape[d] - 1) * 0.5
    return A


def affine_world_axes(arr_shape, affine):
    """Per-axis world-coordinate arrays implied by a diagonal NIfTI affine.

    Returns a 3-tuple of 1D arrays usable as the grid for
    :class:`scipy.interpolate.RegularGridInterpolator`.
    """
    return tuple(affine[d, 3] + np.arange(arr_shape[d]) * affine[d, d]
                 for d in range(3))


def apply_axis_flips(arr, slice_flip=False, row_flip=False, col_flip=False,
                     swap_row_col=False, transpose_to_ras=True):
    """Apply storage-axis flips and optional (slice, row, col) -> (i, j, k)
    transpose to a 3D array.

    The base transpose convention is the mouse one: raw storage is
    (slice=AP, row=LR, col=DV); after the closing transpose(1,0,2) the
    array is in RAS index order (i=LR, j=AP, k=DV).

    Some scanners (macaque AMNH, DigiMorph rat) store
    (slice=AP, row=DV, col=LR) instead -- row and col are swapped
    relative to mouse. Set ``swap_row_col=True`` to do a transpose(0,2,1)
    before the per-axis flips, restoring the mouse convention so the
    SLICE_FLIP / ROW_FLIP / COL_FLIP semantics and the closing
    transpose remain unchanged.
    """
    if swap_row_col:
        arr = arr.transpose(0, 2, 1)
    if slice_flip:
        arr = arr[::-1, :, :]
    if row_flip:
        arr = arr[:, ::-1, :]
    if col_flip:
        arr = arr[:, :, ::-1]
    if transpose_to_ras:
        arr = arr.transpose(1, 0, 2)
    return arr
