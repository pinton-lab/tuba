"""ITRUSST benchmark skull -> voxel masks (bone + endocranial cavity).

The ITRUSST transcranial-ultrasound benchmark skull (Aubry et al., JASA
2022) ships as two watertight STL surfaces -- an outer and an inner
(endocranial) table -- distributed from the public
``ucl-bug/transcranial-ultrasound-benchmarks`` repository. See
:mod:`tuba.mini.fetch` for staging.

This module rasterizes those meshes onto a shared 1 mm grid and derives:

* ``bone``   -- ``outer_filled & ~inner_filled`` (the calvarial shell),
* ``cavity`` -- ``inner_filled`` (the endocranial space; the moving image
  for the MNI brain-mask registration in :mod:`tuba.mini.register`).

Coordinate frame
----------------
The benchmark mesh is authored in a conventional **RAS** frame: mesh
axis 0 is L->R, axis 1 is P->A, axis 2 is I->S, and the endocranial
cavity centroid sits at ~``(1, -21, 7)`` mm -- i.e. within ~2 mm of the
MNI152 brain centre of mass ``(0, -22, 9.5)``. The skull is therefore
already MNI-aligned to within an affine, which is why the downstream
SyN starts from a near-identity initialisation. We keep the mesh mm
coordinates as-is and tag the NIfTI with a plain RAS diagonal affine
``diag(pitch, pitch, pitch)`` translated to the grid corner.

Rasterization uses ``trimesh`` (an optional dependency; ``pip install
tuba[mini]``).
"""
import numpy as np


def _require_trimesh():
    try:
        import trimesh  # noqa: F401
        return trimesh
    except ImportError as e:  # pragma: no cover - trivial guard
        raise ImportError(
            'tuba.mini.skull needs trimesh; install with '
            '`pip install "tuba[mini]"` or `pip install trimesh`.') from e


def _fill_onto_grid(mesh, origin, pitch, shape):
    """Voxelize + solid-fill ``mesh``, resample onto the shared index grid
    defined by ``origin`` (RAS-mm corner) / ``pitch`` / ``shape``.

    Both benchmark meshes are watertight, so ``voxelized(pitch).fill()``
    yields a solid interior; we then snap the occupied voxel centres to
    the common lattice with a nearest-index round. The two meshes share
    ``origin``/``pitch`` so the ``outer & ~inner`` subtraction is
    lattice-consistent (any residual sub-voxel phase slip is << the
    3-7 mm bone thickness).
    """
    vg = mesh.voxelized(pitch=pitch).fill()
    idx = np.round((vg.points - origin) / pitch).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
    idx = idx[ok]
    vol = np.zeros(tuple(shape), dtype=bool)
    vol[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return vol


def rasterize(inner_stl, outer_stl, pitch_mm=1.0, margin_mm=4.0,
              verbose=True):
    """Rasterize the inner/outer skull STLs into aligned voxel masks.

    Parameters
    ----------
    inner_stl, outer_stl : str
        Paths to the endocranial (inner) and outer skull STL surfaces.
    pitch_mm : float
        Isotropic voxel size of the output grid (default 1 mm, matching
        the MNI152 registration grid).
    margin_mm : float
        Padding added around the union of the two mesh bounding boxes.

    Returns
    -------
    dict with keys:
        ``bone``   : (X, Y, Z) bool -- calvarial shell
        ``cavity`` : (X, Y, Z) bool -- endocranial space
        ``affine`` : (4, 4) float  -- voxel index -> RAS world mm
        ``pitch_mm``, ``origin_mm``
    """
    trimesh = _require_trimesh()
    inner = trimesh.load(inner_stl)
    outer = trimesh.load(outer_stl)

    lo = np.minimum(inner.bounds[0], outer.bounds[0]) - margin_mm
    hi = np.maximum(inner.bounds[1], outer.bounds[1]) + margin_mm
    shape = (np.ceil((hi - lo) / pitch_mm).astype(int) + 1)
    if verbose:
        print(f'  grid: origin={np.round(lo, 1)} shape={tuple(shape)} '
              f'pitch={pitch_mm} mm')

    outer_f = _fill_onto_grid(outer, lo, pitch_mm, shape)
    inner_f = _fill_onto_grid(inner, lo, pitch_mm, shape)
    bone = outer_f & ~inner_f
    cavity = inner_f

    affine = np.diag([pitch_mm, pitch_mm, pitch_mm, 1.0]).astype(np.float64)
    affine[:3, 3] = lo

    if verbose:
        p3 = pitch_mm ** 3 / 1000.0
        print(f'  bone   = {bone.sum():>8d} vox ({bone.sum()*p3:6.1f} mL)')
        print(f'  cavity = {cavity.sum():>8d} vox ({cavity.sum()*p3:6.1f} mL)')
        cc = np.argwhere(cavity).mean(0)
        world = affine[:3, :3] @ cc + affine[:3, 3]
        print(f'  cavity centroid (RAS mm) = {np.round(world, 1)} '
              f'(cf. MNI brain CoM ~[0, -22, 9.5])')
    return {'bone': bone, 'cavity': cavity, 'affine': affine,
            'pitch_mm': float(pitch_mm), 'origin_mm': lo}


def save_masks(masks, bone_path, cavity_path, verbose=True):
    """Write the ``bone``/``cavity`` masks (from :func:`rasterize`) to
    NIfTI (uint8), tagged with the shared RAS affine."""
    import nibabel as nib
    for arr, path in ((masks['bone'], bone_path),
                      (masks['cavity'], cavity_path)):
        img = nib.Nifti1Image(arr.astype(np.uint8), masks['affine'])
        nib.save(img, path)
        if verbose:
            print(f'  wrote {path}')
    return bone_path, cavity_path
