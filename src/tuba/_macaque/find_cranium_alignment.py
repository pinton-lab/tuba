"""Constrained PCA for the cranium-only bone cloud.

The cranium alone is near-axially-symmetric in the L-R / D-V plane
(variance ratio ~ 1.1 : 1.0), so unconstrained PCA gives unstable
secondary eigenvectors. We:

  (i)  Take v_AP from PCA -- the largest eigenvalue is well-separated
       (variance ratio ~ 3.7 : 1).
  (ii) Build a candidate D-V axis perpendicular to v_AP (initialized
       from world +z).
  (iii) Sweep rotation about v_AP from 0 to 2pi. For each candidate
       v_DV, project all bone voxels onto v_DV and compute the count
       asymmetry n(+) - n(-). Maximise: the basicranium (thick bone
       plate) has more voxels per unit area than the parietal vault
       (thin shell), so the +v_DV side that is the basicranium will
       have higher voxel count after centroid subtraction.
  (iv) Once the basicranium direction is identified, set
       v_DV_final = -v_DV_basicranium (so the vault is at +v_DV and
       the basicranium is at -v_DV; in RAS this maps to +z = dorsal,
       -z = ventral).
  (v)  v_LR = v_AP x v_DV (right-handed). The L-R sign is left at
       the +x = right convention; any mirror is caught by the SyN
       registration to NMT.

Output: ZYX intrinsic Euler angles (Rx, Ry, Rz) that rotate the
un-aligned cranium so its anatomical axes coincide with the storage
indices.
"""
import os
import sys
import numpy as np
import nibabel as nib

from .macaque_skull_constants import REG_DIR, _rot_matrix_zyx


def find_cranium_alignment(in_path, bone_thresh=25000,
                              bone_thresh_high=35000, n_sweep=720,
                              verbose=True):
    """Compute the alignment angles from the cranium bone cloud.

    `bone_thresh` is the lower bound (cortical-bone start);
    `bone_thresh_high` is the UPPER bound used to EXCLUDE dental
    enamel (40-44 kHU peak) which would otherwise pull the AP
    eigenvector toward the maxillary teeth. Set bone_thresh_high to
    a large value (>50000) to disable the dental-enamel exclusion.
    """
    img = nib.load(in_path)
    vol = img.get_fdata()
    bone_mask = (vol > bone_thresh) & (vol < bone_thresh_high)
    bone_idx = np.argwhere(bone_mask).astype(np.float64)
    n_bone = len(bone_idx)
    if n_bone == 0:
        raise RuntimeError(f'no bone voxels above {bone_thresh}')
    centroid = bone_idx.mean(axis=0)
    pts = bone_idx - centroid
    cov = (pts.T @ pts) / n_bone

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    if verbose:
        print(f'PCA on {n_bone} bone voxels '
              f'(thresh range [{bone_thresh}, {bone_thresh_high}], '
              f'cortical bone only)')
        print(f'  variances (vox^2): {eigvals}')
        print(f'  variance ratio: {eigvals/eigvals[-1]}')

    # AP axis from PCA (largest eigenvalue)
    v_AP = eigvecs[:, 0]
    e_i, e_j, e_k = np.eye(3)
    if v_AP @ e_j < 0:
        v_AP = -v_AP
    if verbose:
        print(f'  v_AP (from PCA) = {v_AP}')

    # Initialize a candidate D-V direction perpendicular to v_AP, from
    # world +z (storage k axis).
    helper = e_k if abs(v_AP @ e_k) < 0.9 else e_i
    v_DV_init = helper - (helper @ v_AP) * v_AP
    v_DV_init /= np.linalg.norm(v_DV_init)
    v_LR_init = np.cross(v_AP, v_DV_init)
    v_LR_init /= np.linalg.norm(v_LR_init)

    # Sweep rotation about v_AP. For each angle theta, build
    #   v_dv_candidate = cos(theta) * v_DV_init + sin(theta) * v_LR_init
    # Project bone onto it, compute count asymmetry.
    thetas = np.linspace(0, 2 * np.pi, n_sweep, endpoint=False)
    asym = np.zeros(n_sweep)
    pts_T = pts.T  # 3 x N for fast dot products
    for k in range(n_sweep):
        c, s = np.cos(thetas[k]), np.sin(thetas[k])
        v_cand = c * v_DV_init + s * v_LR_init
        proj = v_cand @ pts_T
        n_pos = (proj > 0).sum()
        n_neg = (proj < 0).sum()
        asym[k] = (n_pos - n_neg) / max(1, n_pos + n_neg)

    # Maximum asymmetry: the basicranium is in the +v_dv_candidate
    # direction. Find both the global max and confirm by symmetry that
    # the argmin is at theta+pi.
    k_max = int(np.argmax(asym))
    k_min = int(np.argmin(asym))
    theta_basicranium = thetas[k_max]
    if verbose:
        print(f'  asymmetry sweep: max={asym[k_max]:+.4f} @ '
              f'theta={np.rad2deg(theta_basicranium):.1f} deg, '
              f'min={asym[k_min]:+.4f} @ theta='
              f'{np.rad2deg(thetas[k_min]):.1f} deg')
        delta = (np.rad2deg(thetas[k_min]) -
                 np.rad2deg(thetas[k_max]) + 540) % 360 - 180
        print(f'  min-max angular separation: {delta:+.1f} deg '
              f'(expected ~180)')

    c, s = np.cos(theta_basicranium), np.sin(theta_basicranium)
    v_DV_basicranium = c * v_DV_init + s * v_LR_init
    # We want +v_DV to point toward the dorsal vault (NOT toward
    # basicranium), so v_DV_final = -v_DV_basicranium.
    v_DV = -v_DV_basicranium
    v_DV /= np.linalg.norm(v_DV)
    v_LR = np.cross(v_AP, v_DV)
    v_LR /= np.linalg.norm(v_LR)

    # The L-R sign is determined by the right-hand rule applied to
    # the (v_AP, v_DV) anatomical pair. If the resulting v_LR points
    # in -e_i (storage left), the cranium will be mirrored in storage.
    # We flag this so the caller can apply ROW_FLIP=True after the
    # rotation to restore the +x=right convention. v_AP and v_DV are
    # NOT modified -- they are anatomically correct as-is.
    needs_lr_flip = (v_LR @ e_i < 0)
    if verbose:
        print(f'  v_LR = {v_LR}  (needs_lr_flip = {needs_lr_flip})')
        print(f'  v_AP = {v_AP}')
        print(f'  v_DV = {v_DV}')

    V = np.column_stack([v_LR, v_AP, v_DV])
    if abs(np.linalg.det(V) - 1.0) > 1e-3:
        raise RuntimeError(f'rotation matrix is not a proper rotation '
                            f'(det = {np.linalg.det(V):.3f})')
    R = V.T

    ay = np.arcsin(-R[2, 0])
    ax = np.arctan2(R[2, 1], R[2, 2])
    az = np.arctan2(R[1, 0], R[0, 0])
    ax_d, ay_d, az_d = np.rad2deg([ax, ay, az])
    if verbose:
        print(f'  Euler (ZYX intrinsic): Rx={ax_d:.2f}  Ry={ay_d:.2f}  '
              f'Rz={az_d:.2f} deg')
        R_rt = _rot_matrix_zyx(ax_d, ay_d, az_d)
        print(f'  round-trip ||R - Rz@Ry@Rx||_F = '
              f'{np.linalg.norm(R - R_rt):.2e}')
        if needs_lr_flip:
            print(f'  REQUIRED: set ROW_FLIP=True in constants so that '
                  f'storage axis 1 (L-R) is reversed after rotation')
    return ax_d, ay_d, az_d, needs_lr_flip


if __name__ == '__main__':
    path = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(REG_DIR, 'macaque_skull_250um_cranium.nii.gz'))
    print(f'Computing cranium alignment from {path}')
    ax, ay, az, needs_lr_flip = find_cranium_alignment(path)
    print(f'\nAngles: Rx={ax:+.2f}  Ry={ay:+.2f}  Rz={az:+.2f}'
          f'  (ROW_FLIP needed: {needs_lr_flip})')
