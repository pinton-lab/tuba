"""PCA-based rigid pre-alignment of a skull volume.

After the storage->RAS axis flips, the skull's anatomical axes are
still tilted relative to the index axes (the dorsal vault is not
horizontal, the midsagittal plane is not exactly perpendicular to
R-L, etc.). A single rigid 3D rotation about the bone centroid puts
the cortical-bone principal axes in alignment with the storage index
axes; the cortical-bone centroid is then translated to the volume
centre so the brain cavity sits at world (0, 0, 0).

This is a one-shot choice of coordinates -- it does NOT affect the
SyN registration to the atlas (which finds its own optimum
independently) but it makes all downstream slice plots cleanly
axis-aligned and prevents the working-bounding-box from clipping the
foramen-magnum end of the cavity (see :mod:`tuba.core.cavity`).

Manuscript §5 (mouse case):

* Eigenvectors of the centroid-centred bone cloud are assigned to
  target storage axes by descending variance:

  - largest eigenvalue  -> A-P (the skull's 23.8 mm long axis)
  - middle eigenvalue   -> R-L (15.9 mm width)
  - smallest eigenvalue -> D-V (11.7 mm height)

* A right-handed orthonormal frame V = [v_RL v_AP v_DV] is built; the
  rotation R = V^T sends V's columns onto the storage basis. Closed-
  form intrinsic-ZYX decomposition of R yields the Euler angles used
  by :func:`rot_matrix_zyx`.

* For the mouse Maga scan the PCA-derived angles are
  (a_x, a_y, a_z) = (+8.07, +8.40, -19.51) deg. After the rotation
  the dorsal vault apex lands within +/- 0.5 voxel of the midsagittal
  plane and the basicranial plates are approximately symmetric about
  R-L = 0.

Threshold choice (species-specific): the threshold parameter must be
high enough to exclude both the specimen mount AND the diffuse vault
bone (which is rotationally biased by the basicranium and snout), so
PCA's covariance is dominated by the well-defined bilateral dental
arches and basicranial plates. For mouse Maga, ``ALIGN_PCA_THRESH =
30000`` on the uint16 NRecon scale; species modules pick an analogue
appropriate to their intensity histogram.
"""
import os
import numpy as np
import nibabel as nib
from scipy import ndimage

from . import frame


def rot_matrix_zyx(ax_deg, ay_deg, az_deg):
    """Intrinsic ZYX rotation: ``R = Rz(az) @ Ry(ay) @ Rx(ax)``."""
    ax, ay, az = np.deg2rad([ax_deg, ay_deg, az_deg])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def find_pca_rotation(vol, bone_thresh, verbose=True):
    """PCA on the bone voxel cloud (intensity > ``bone_thresh``) to find
    the rotation that aligns the skull's anatomical axes with storage
    index axes.

    Returns ``(ax_deg, ay_deg, az_deg)`` in the ZYX intrinsic convention
    used by :func:`rot_matrix_zyx` (``R = Rz @ Ry @ Rx`` applied to
    column vectors).

    Sign convention: each eigenvector is flipped, if needed, to have
    positive dot product with its target storage axis (minimises the
    rotation magnitude).
    """
    bone = np.argwhere(vol > bone_thresh).astype(np.float64)
    n_bone = len(bone)
    if n_bone == 0:
        raise RuntimeError(f'no bone voxels above {bone_thresh}')
    # PCA must be about the cloud centroid, not the volume centre.
    centroid = bone.mean(axis=0)
    pts = bone - centroid
    cov = (pts.T @ pts) / n_bone

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    v_AP, v_RL, v_DV = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]
    if verbose:
        print(f'PCA on {n_bone} bone voxels (thresh={bone_thresh})')
        print(f'  variances (vox^2): {eigvals}')
        print(f'  pre-sign: v_AP={v_AP}, v_RL={v_RL}, v_DV={v_DV}')

    e_i, e_j, e_k = np.eye(3)
    if v_AP @ e_j < 0: v_AP = -v_AP
    if v_RL @ e_i < 0: v_RL = -v_RL
    if v_DV @ e_k < 0: v_DV = -v_DV
    V = np.column_stack([v_RL, v_AP, v_DV])
    if np.linalg.det(V) < 0:
        align = np.array([v_RL @ e_i, v_AP @ e_j, v_DV @ e_k])
        worst = int(np.argmin(align))
        V[:, worst] = -V[:, worst]
        if verbose:
            print(f'  det(V) was negative; flipped column {worst}')
    R = V.T

    ay = np.arcsin(-R[2, 0])
    ax = np.arctan2(R[2, 1], R[2, 2])
    az = np.arctan2(R[1, 0], R[0, 0])
    ax_d, ay_d, az_d = np.rad2deg([ax, ay, az])
    if verbose:
        print(f'  Euler (ZYX intrinsic): '
              f'Rx={ax_d:.2f}, Ry={ay_d:.2f}, Rz={az_d:.2f} deg')
        R_rt = rot_matrix_zyx(ax_d, ay_d, az_d)
        print(f'  round-trip ||R - Rz@Ry@Rx||_F = '
              f'{np.linalg.norm(R - R_rt):.2e}')
    return ax_d, ay_d, az_d


def rotate_volume_about_centroid(vol, ax_deg, ay_deg, az_deg,
                                 bone_thresh, pad_voxels, order=1,
                                 verbose=True):
    """Rotate a 3D volume by ZYX intrinsic Euler angles about the
    cortical-bone centroid (``vol > bone_thresh``), padded to fit.

    The output volume is padded by ``pad_voxels`` on every side
    relative to the input; the bone centroid lands at the OUTPUT volume
    centre. This is the pre-alignment step that puts the brain cavity
    at world (0, 0, 0).
    """
    R = rot_matrix_zyx(ax_deg, ay_deg, az_deg)
    R_inv = R.T
    in_shape = np.array(vol.shape)
    out_shape = tuple(int(s + 2 * pad_voxels) for s in in_shape)
    out_centre = (np.array(out_shape) - 1) / 2.0

    bone = np.argwhere(vol > bone_thresh)
    if len(bone) == 0:
        raise RuntimeError(f'no bone voxels above {bone_thresh} for centring')
    c_in = bone.mean(axis=0).astype(np.float64)
    if verbose:
        print(f'  {len(bone):,} bone voxels; centroid = {c_in}')
    del bone

    offset = c_in - R_inv @ out_centre
    if verbose:
        print(f'  rotation R = Rz({az_deg}) @ Ry({ay_deg}) @ Rx({ax_deg}) '
              f'(intrinsic ZYX), pad={pad_voxels} -> out_shape={out_shape}')
    rotated = ndimage.affine_transform(
        vol, R_inv, offset=offset, output_shape=out_shape,
        order=order, mode='constant', cval=0, prefilter=False)
    if verbose:
        print(f'  rotated: shape={rotated.shape}, dtype={rotated.dtype}, '
              f'max={rotated.max()}')
    return rotated


def align_nifti_to_axes(in_path, out_path, ax_deg, ay_deg, az_deg,
                        bone_thresh, pad_voxels=12, voxel_signs=(-1, -1, +1),
                        order=1, force=False, verbose=True):
    """NIfTI-level wrapper: load, rotate about bone centroid, write
    with an updated affine that keeps the output centre at the world
    origin.
    """
    if os.path.exists(out_path) and not force:
        return out_path
    img = nib.load(in_path)
    vol = img.get_fdata().astype(np.float32)
    rotated = rotate_volume_about_centroid(
        vol, ax_deg, ay_deg, az_deg, bone_thresh, pad_voxels,
        order=order, verbose=verbose)
    voxel_mm = float(abs(img.affine[0, 0]))
    out_affine = frame.affine_for_shape(rotated.shape, voxel_mm, signs=voxel_signs)
    if verbose:
        print(f'Rotated {in_path} by (Rx={ax_deg}, Ry={ay_deg}, Rz={az_deg}) deg')
        print(f'  out: shape={rotated.shape} (pad={pad_voxels}), '
              f'max={rotated.max():.0f}')
    nib.save(nib.Nifti1Image(rotated.astype(np.float32), out_affine), out_path)
    if verbose:
        print(f'  wrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)')
    return out_path
