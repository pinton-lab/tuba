"""Outer-bone surface extraction.

Two methods supported:

* ``"close+fill"`` -- binary closing of (intensity > bone_thresh) plus
  a low-intensity head threshold to absorb any adherent mounting
  material, then ``binary_fill_holes`` to close any remaining cavities,
  then boundary voxels (head & ~erode(head)). Used by:

  - mouse Maga placement helper (1 mm closing, head_thresh=500 to
    catch mounting plateau);
  - human Halle pipeline (22 mm closing for foramen magnum + orbits).

* ``"raycast"`` -- ray-cast from the skull centroid outward in spherical
  directions, returning the first-hit bone-above-threshold voxel along
  each ray. Used by the human pipeline's
  ``extract_outer_bone_surface.py`` (the surface mesh is approximately
  the outer table; the surface from "close+fill" is the convex-hull-ish
  envelope that includes any concavities).

The two methods give noticeably different envelopes near the orbits.
"""
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes


def outer_bone_surface_world_mm(skull_path, bone_thresh, close_mm=1.0,
                                head_thresh=500.0, method='close+fill',
                                verbose=True):
    """Return outer-bone surface points in world-mm.

    Parameters
    ----------
    skull_path : str
        NIfTI path to an aligned skull volume (RAS world frame).
    bone_thresh : float
        Bone intensity threshold (typically the cavity extractor's
        ``BONE_LOW``).
    close_mm : float
        Closing radius in mm. 1 mm fills cribriform plate + sub-mm
        sutures (mouse). 22 mm fills foramen magnum + orbits (human).
    head_thresh : float
        Lower threshold for "head mask" inclusion -- catches adherent
        mounting material below the bone threshold but above
        reconstruction-edge noise. Set to None to use ``bone_thresh``
        directly.
    method : {"close+fill", "raycast"}
        Extraction algorithm. ``"raycast"`` is not yet implemented in
        TUBA core; species modules that need it should call out to the
        legacy extractor for now.

    Returns
    -------
    (N, 3) float64 ndarray of world-mm points.
    """
    if method == 'close+fill':
        img = nib.load(skull_path)
        arr = np.asarray(img.dataobj).astype(np.float32)
        aff = img.affine
        bone = arr > bone_thresh
        if head_thresh is not None and head_thresh > 0:
            head = bone | (arr > head_thresh)
        else:
            head = bone
        vox_mm = float(abs(aff[0, 0]))
        close_iters = max(1, int(round(close_mm / vox_mm)))
        head = binary_closing(head, iterations=close_iters)
        head = binary_fill_holes(head)
        surf = head & ~binary_erosion(head, iterations=1)
        ijk = np.argwhere(surf).astype(np.float64)
        if len(ijk) == 0:
            raise RuntimeError(f'No outer-bone surface voxels found in {skull_path}')
        ras = (aff[:3, :3] @ ijk.T).T + aff[:3, 3]
        if verbose:
            print(f'    {len(ras):,} outer-bone surface voxels '
                  f'(close+fill, close_iters={close_iters} @ {vox_mm*1000:.1f} um)')
        return ras
    raise NotImplementedError(
        f'surface method {method!r} not supported by '
        f'outer_bone_surface_world_mm; for "raycast" use '
        f'raycast_outer_bone_surface_world_mm directly (it requires '
        f'a different I/O signature).')


def raycast_outer_bone_surface_world_mm(bone_mask, voxel_mm,
                                        affine_storage_to_world,
                                        n_azim=240, n_elev=120,
                                        return_normals=True,
                                        normal_k=12,
                                        verbose=True):
    """Ray-cast outer-bone-table surface from the head centroid.

    For each of ``n_azim * n_elev`` uniformly-sampled spherical
    directions, walks outward from the bone-mass centroid in 1-voxel
    steps and records the OUTERMOST bone-above-threshold voxel along
    the ray. Optional local-PCA normal estimation on the resulting
    point cloud.

    Used by the human Halle pipeline (the close+fill outer envelope
    sits inside the bone or on the inner table -- not where a
    transducer would be placed; ray-casting from the centroid gives
    the proper outer table).

    Parameters
    ----------
    bone_mask : 3D bool ndarray
        ``volume > bone_threshold``. Already on the working grid.
    voxel_mm : float
        Voxel size (mm), used to set the ray step and convert indices
        to mm.
    affine_storage_to_world : (3, 3) ndarray
        Storage-index -> world-mm rotation+sign matrix (no translation;
        the centroid offset is handled internally). For Halle this is
        ``diag(-voxel_mm, +voxel_mm, +voxel_mm)``.
    n_azim, n_elev : int
        Spherical sampling counts. Default ``240 * 120 = 28800`` rays.
    return_normals : bool
        If True, estimate vertex normals via local PCA on ``normal_k``
        nearest neighbours.
    normal_k : int
        Neighbourhood size for normal estimation.

    Returns
    -------
    verts_world_mm : (N, 3) float64
    normals_world_mm : (N, 3) float64 if ``return_normals``, else None
    """
    from scipy.spatial import cKDTree
    bone = np.asarray(bone_mask, dtype=bool)
    n_L, n_P, n_S = bone.shape
    bone_idx = np.where(bone)
    centroid_idx_mm = np.array([
        float(np.mean(bone_idx[0])) * voxel_mm,
        float(np.mean(bone_idx[1])) * voxel_mm,
        float(np.mean(bone_idx[2])) * voxel_mm,
    ])
    if verbose:
        print(f'    raycast bone centroid (storage-mm) = '
              f'{centroid_idx_mm.round(1)}')

    azims = np.linspace(0, 2*np.pi, n_azim, endpoint=False)
    elevs = np.linspace(-np.pi/2 + np.pi/(2*n_elev),
                        np.pi/2 - np.pi/(2*n_elev), n_elev)
    dirs = []
    for el in elevs:
        for az in azims:
            dirs.append([np.cos(el) * np.cos(az),
                         np.cos(el) * np.sin(az),
                         np.sin(el)])
    dirs = np.asarray(dirs)
    if verbose:
        print(f'    {len(dirs)} ray directions')

    max_steps = int(np.ceil(np.linalg.norm(np.array(bone.shape)) * voxel_mm))
    step_mm = voxel_mm
    surface_pts_storage = []
    for d_idx, direction in enumerate(dirs):
        last_bone = None
        for s in range(1, max_steps):
            r = s * step_mm
            p = centroid_idx_mm + r * direction
            iL = int(round(p[0] / voxel_mm))
            iP = int(round(p[1] / voxel_mm))
            iS = int(round(p[2] / voxel_mm))
            if not (0 <= iL < n_L and 0 <= iP < n_P and 0 <= iS < n_S):
                break
            if bone[iL, iP, iS]:
                last_bone = p
        if last_bone is not None:
            surface_pts_storage.append(last_bone)
        if verbose and d_idx % 5000 == 0:
            print(f'    ray {d_idx}/{len(dirs)}: '
                  f'{len(surface_pts_storage)} hits so far')
    pts_storage_mm = np.asarray(surface_pts_storage, dtype=np.float64)
    if verbose:
        print(f'    raycast surface: {len(pts_storage_mm)} verts')

    # Storage-mm -> world-mm via the species' axis-sign mapping.
    verts_world_mm = pts_storage_mm @ affine_storage_to_world.T

    normals_world_mm = None
    if return_normals:
        tree = cKDTree(verts_world_mm)
        # Centroid in world-mm too (for sign convention).
        centroid_world_mm = centroid_idx_mm @ affine_storage_to_world.T
        normals_world_mm = np.zeros_like(verts_world_mm)
        for i in range(len(verts_world_mm)):
            _, idx = tree.query(verts_world_mm[i], k=normal_k)
            nbr = verts_world_mm[idx] - verts_world_mm[idx].mean(axis=0, keepdims=True)
            cov = nbr.T @ nbr
            _, V = np.linalg.eigh(cov)
            n = V[:, 0]
            if np.dot(n, verts_world_mm[i] - centroid_world_mm) < 0:
                n = -n
            normals_world_mm[i] = n / (np.linalg.norm(n) + 1e-12)
        if verbose:
            print(f'    raycast normals: {len(normals_world_mm)} normals '
                  f'(local-PCA k={normal_k})')
    return verts_world_mm, normals_world_mm
