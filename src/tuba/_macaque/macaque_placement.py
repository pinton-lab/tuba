"""Place a focused bowl on the AMNH macaque skull aimed at a CHARM-
defined dorsal anterior cingulate cortex (dACC) target.

Direct port of mouse_therapy/registration/maga_placement.py with:
  * Atlas: NMT v2 + CHARM (4D 1-x-6 cortical hierarchy) instead of
    Allen CCFv3.
  * Default target: 'dACC' (CHARM L6 area_24a/24b, plus optional
    24a'/24b'/area_32) -- bilateral, midline approach.
  * Outer-bone surface threshold matches the macaque BONE_LOW (10000)
    rather than Maga's 5000.
  * Pre-set scalp clearance and apex_to_target distance defaults
    appropriate for a 500 kHz f/1 bowl with ROC=45 mm aimed at a
    ~30-35 mm-deep cortical target.

Returned ``placement`` keys mirror ``maga_placement`` so the run
script (``run_macaque.py``) can be the same shape as ``run_mouse_cc.py``.
"""
import os
from .macaque_skull_constants import REG_DIR
import sys
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import (binary_fill_holes, binary_closing,
                            binary_erosion)


CHARM_NII = os.path.join(REG_DIR, 'charm_in_macaque.nii.gz')
SARM_NII = os.path.join(REG_DIR, 'sarm_in_macaque.nii.gz')
SKULL_250UM = os.path.join(REG_DIR,
                            'macaque_skull_aligned_native_250um.nii.gz')
CAVITY_NII = os.path.join(REG_DIR, 'macaque_cranial_cavity.nii.gz')

# CHARM L6 (finest cortical hierarchy) integer IDs for area 24
# (a / b / a' / b' = dorsal cingulate sub-areas) and area 32
# (perigenual). From tables_CHARM/CHARM_key_6.txt:
#    8  area_24a              (level 6 native)
#    9  area_24b              (level 6 native)
#   13  area_24a_prime        (level 6 native)
#   14  area_24b_prime        (level 6 native)
#    4  area_32               (first appears at level 4)
DACC_IDS = {8, 9, 13, 14}                  # bilateral area 24
DACC_BROAD_IDS = {4, 8, 9, 13, 14}         # add area 32 (perigenual)

BONE_LOW = 10000.0


def _charm_target_centroid(target='dACC'):
    """Centroid of the warped CHARM target structure(s) in macaque
    aligned-native world-mm. CHARM is 5D shape (x, y, z, 1, 6); we
    extract level 6 (finest) as ``arr[..., 0, 5]`` and union the
    requested integer IDs."""
    img = nib.load(CHARM_NII)
    arr = np.asarray(img.dataobj)
    aff = img.affine
    if arr.ndim == 5:
        l6 = arr[..., 0, 5]
    elif arr.ndim == 4:
        l6 = arr[..., 5]
    else:
        l6 = arr
    if target == 'dACC':
        ids = DACC_IDS
    elif target == 'dACC_broad':
        ids = DACC_BROAD_IDS
    else:
        raise ValueError(f'Unknown target {target!r}')
    mask = np.isin(l6, list(ids))
    ijk = np.argwhere(mask)
    if len(ijk) == 0:
        raise RuntimeError(f'No voxels for target {target} (ids {ids}) '
                            f'in {CHARM_NII}')
    ras = (aff[:3, :3] @ ijk.T).T + aff[:3, 3]
    return ras.mean(axis=0), int(len(ijk))


def _outer_bone_surface_world_mm(skull_path=SKULL_250UM,
                                   bone_thresh=BONE_LOW):
    """Outer-bone surface points in macaque RAS world-mm. The closing
    radius is scaled to voxel size; 2 mm at 250 um voxel = 8 iter."""
    img = nib.load(skull_path)
    arr = np.asarray(img.dataobj).astype(np.float32)
    aff = img.affine
    bone = arr > bone_thresh
    head = bone | (arr > 1000)            # below mounting plateau
    vox_mm = float(abs(aff[0, 0]))
    close_iters = max(1, int(round(2.0 / vox_mm)))
    head = binary_closing(head, iterations=close_iters)
    head = binary_fill_holes(head)
    surf = head & ~binary_erosion(head, iterations=1)
    ijk = np.argwhere(surf).astype(np.float64)
    if len(ijk) == 0:
        raise RuntimeError(f'No outer-bone surface voxels found in {skull_path}')
    ras = (aff[:3, :3] @ ijk.T).T + aff[:3, 3]
    return ras


def _rotate_about_x_at_pivot(p, pivot, theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0,   c,  -s],
                  [0.0,   s,   c]])
    return pivot + R @ (np.asarray(p, dtype=np.float64) - pivot)


def _brain_centre_world_mm():
    """Cavity-centroid pivot for the sagittal-plane tilt, in world-mm."""
    if not os.path.exists(CAVITY_NII):
        return np.array([0.0, 0.0, 0.0])
    img = nib.load(CAVITY_NII)
    arr = np.asarray(img.dataobj)
    aff = img.affine
    ijk = np.argwhere(arr > 0.5)
    if len(ijk) == 0:
        return np.array([0.0, 0.0, 0.0])
    return (aff[:3, :3] @ ijk.mean(axis=0)) + aff[:3, 3]


def place_on_macaque_skull(target_name='dACC',
                              scalp_clearance_mm=None,
                              apex_to_target_mm=45.0,
                              beam_tilt_deg_yz=0.0,
                              lateral_search_mm=10.0,
                              debug=False):
    """Compute placement for a 500 kHz single-element bowl aimed at the
    named macaque-brain target.

    Defaults match the validation report bowl: f0=500 kHz, D=45 mm,
    ROC=45 mm (f/1). apex_to_target_mm=45 puts the geometric focus
    on the target centroid for a 45 mm-ROC bowl.

    Parameters
    ----------
    target_name : {'dACC', 'dACC_broad', 'cavity_centre'}
    scalp_clearance_mm : float, optional
        Water-column distance above the outer bone surface. Ignored if
        apex_to_target_mm is set.
    apex_to_target_mm : float, optional (default 45)
        If set, lift the apex along +z so its Euclidean distance to the
        target equals this value (matches a geometric-focus bowl with
        the focal point at the target).
    beam_tilt_deg_yz : float
        Sagittal-plane tilt about the cavity centroid.
    lateral_search_mm : float (default 10)
        Radius around (target_x, target_y) within which the highest-z
        outer-bone surface point is taken as the apex candidate. The
        macaque skull is much larger than the mouse so the search
        radius is correspondingly larger.

    Returns
    -------
    placement : dict
    """
    print(f'Placing bowl on macaque skull for target {target_name!r}')

    if target_name == 'cavity_centre':
        target_world = _brain_centre_world_mm()
        n_vox = -1
    else:
        target_world, n_vox = _charm_target_centroid(target_name)
        print(f'  CHARM target voxels: {n_vox}')
    print(f'  target world-mm = {target_world.round(3)}')

    print('  building outer bone surface from 250 um aligned cache...')
    surf = _outer_bone_surface_world_mm()
    print(f'    {len(surf):,} outer-bone surface voxels')

    tx, ty, tz = target_world
    dorsal = surf[surf[:, 2] > tz]
    if len(dorsal) == 0:
        raise RuntimeError('No dorsal outer-bone surface voxels above target')
    radial = np.hypot(dorsal[:, 0] - tx, dorsal[:, 1] - ty)
    cyl = dorsal[radial < lateral_search_mm]
    if len(cyl) == 0:
        print(f'    no surface within {lateral_search_mm} mm, falling back '
              f'to all dorsal points')
        cyl = dorsal
    apex_world = cyl[np.argmax(cyl[:, 2])].copy()
    print(f'    apex on bone world-mm = {apex_world.round(3)}')

    apex_lifted = apex_world.copy()
    if apex_to_target_mm is not None:
        lat2 = (target_world[0] - apex_world[0])**2 + \
               (target_world[1] - apex_world[1])**2
        if apex_to_target_mm**2 < lat2:
            raise ValueError(
                f'Lateral offset {np.sqrt(lat2):.2f} mm > apex->target '
                f'{apex_to_target_mm} mm; pick a deeper target.')
        apex_z = target_world[2] + np.sqrt(apex_to_target_mm**2 - lat2)
        L = apex_z - apex_world[2]
        if L < 0:
            raise ValueError(
                f'Required apex lift is negative ({L:.2f} mm); '
                f'increase apex_to_target_mm.')
        apex_lifted[2] = apex_z
        scalp_clearance_mm = float(L)
    else:
        if scalp_clearance_mm is None:
            scalp_clearance_mm = 5.0
        apex_lifted[2] += scalp_clearance_mm

    beam = target_world - apex_lifted
    dist = float(np.linalg.norm(beam))
    beam_unit = beam / dist
    print(f'    apex lifted world-mm = {apex_lifted.round(3)}  '
          f'(lift {scalp_clearance_mm:.2f} mm)')
    print(f'    beam unit            = {beam_unit.round(4)}')
    print(f'    apex -> target dist  = {dist:.2f} mm')

    if abs(beam_tilt_deg_yz) > 1e-6:
        pivot = _brain_centre_world_mm()
        theta = np.deg2rad(+beam_tilt_deg_yz)
        apex_lifted = _rotate_about_x_at_pivot(apex_lifted, pivot, theta)
        target_world = _rotate_about_x_at_pivot(target_world, pivot, theta)
        apex_world = _rotate_about_x_at_pivot(apex_world, pivot, theta)
        beam = target_world - apex_lifted
        dist = float(np.linalg.norm(beam))
        beam_unit = beam / dist
        print(f'  applied {beam_tilt_deg_yz:+.1f} deg y-z tilt about '
              f'cavity centre {pivot.round(2)}')
        print(f'    new apex world-mm  = {apex_lifted.round(3)}')
        print(f'    new target         = {target_world.round(3)}')
        print(f'    new beam unit      = {beam_unit.round(4)}')

    placement = {
        'target_name': target_name,
        'target_lps': tuple(float(x) for x in target_world),
        'xdc_center_lps': tuple(float(x) for x in apex_lifted),
        'focus_lps': tuple(float(x) for x in target_world),
        'beam_dir_3d': tuple(float(x) for x in beam_unit),
        'scalp_target_dist_mm': dist,
        'scalp_clearance_mm': float(scalp_clearance_mm),
        'apex_on_bone_voxel_mm': tuple(float(x) for x in apex_world),
        'target_world_mm': tuple(float(x) for x in target_world),
        'beam_tilt_deg_yz': float(beam_tilt_deg_yz),
        'frame': 'macaque_aligned_native_RAS',
    }
    return placement


if __name__ == '__main__':
    target = os.environ.get('MACAQUE_TARGET', 'dACC')
    apex_dist = float(os.environ.get('MACAQUE_APEX_TO_TARGET_MM', 45.0))
    tilt = float(os.environ.get('MACAQUE_TILT_DEG', 0.0))
    p = place_on_macaque_skull(
        target_name=target,
        apex_to_target_mm=apex_dist,
        beam_tilt_deg_yz=tilt)
    print('\nPlacement:')
    print(json.dumps(p, indent=2))
    out = os.path.join(REG_DIR, 'macaque_placement.json')
    with open(out, 'w') as f:
        json.dump(p, f, indent=2)
    print(f'wrote {out}')
