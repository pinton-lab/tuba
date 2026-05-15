"""Block-max downsampling with NIfTI-affine propagation.

Block-max (rather than mean) preserves cortical-bone peak intensities,
so the same ``BONE_HIGH`` / ``BONE_LOW`` thresholds used at the
native resolution remain valid at any downsampled cache (manuscript
§4, §6).

Caches are always derived from the *aligned native*, never from an
independently-coarsened raw cache. The reason (manuscript §6): a 200
um cache built from raw TIFFs and then PCA-rotated chooses its bone
centroid AFTER block-max has smeared the 4-8 kcts mounting-material
plateau into apparent bone -- biasing the centroid ventrally by ~1
mm. Deriving the working cache from the already-aligned native
sidesteps this bias and ensures the cavity, native skull, and
downstream atlas overlays share an identical world frame; otherwise
a cavity contour overlaid on the native pixel grid appears shifted
~1 mm dorsoventrally relative to the bone vault.

Two streaming entry points cover the mouse Maga case:

* :func:`downsample_tiff_stack_to_nifti` -- factor-N block-max from a
  raw TIFF stack to an axis-flipped NIfTI (200 um and 50 um caches).
* :func:`downsample_nifti_by_factor` -- factor-N block-max from the
  aligned native NIfTI to coarser inheriting caches (25 um and 200 um
  aligned).
"""
import os
import numpy as np
import nibabel as nib

from . import frame


def block_max(a, factor, axis):
    """Reduce length along ``axis`` by ``factor`` via block-max.
    Truncates so that the output length is exactly ``shape[axis] // factor``.
    """
    n = a.shape[axis]
    n_out = n // factor
    if n_out == 0:
        return a.take(indices=[], axis=axis)
    trim_slc = [slice(None)] * a.ndim
    trim_slc[axis] = slice(0, n_out * factor)
    a = a[tuple(trim_slc)]
    new_shape = list(a.shape)
    new_shape[axis:axis + 1] = [n_out, factor]
    return a.reshape(new_shape).max(axis=axis + 1)


def block_max_3d(arr, factor):
    """Block-max along all three axes."""
    arr = block_max(arr, factor, axis=0)
    arr = block_max(arr, factor, axis=1)
    arr = block_max(arr, factor, axis=2)
    return arr


def downsample_nifti_by_factor(in_path, out_path, factor, force=False,
                               verbose=True):
    """Block-max downsample a NIfTI volume by an integer ``factor``.

    The output inherits the input's affine direction and origin (voxel
    (0,0,0) of the output is the same world point as voxel (0,0,0) of
    the input) and has spacing increased by ``factor``. Result dtype is
    ``uint16``.
    """
    if os.path.exists(out_path) and not force:
        return out_path
    import time
    t0 = time.time()
    if verbose:
        print(f'Loading {in_path}...')
    img = nib.load(in_path)
    arr = np.asarray(img.dataobj, dtype=np.uint16)
    in_aff = img.affine
    in_spacing = float(abs(in_aff[0, 0]))
    if verbose:
        print(f'  in shape={arr.shape}, in spacing={in_spacing*1000:.3f} um, '
              f'size={arr.nbytes/1e9:.1f} GB, t={(time.time()-t0):.0f} s')
    out_spacing = in_spacing * factor
    if verbose:
        print(f'Block-max factor={factor} -> out spacing={out_spacing*1000:.3f} um')
    arr = block_max_3d(arr, factor)
    if verbose:
        print(f'  downsampled shape={arr.shape}, t={(time.time()-t0):.0f} s')
    out_aff = in_aff.copy()
    for d in range(3):
        out_aff[d, d] = np.sign(in_aff[d, d]) * out_spacing
    nib.save(nib.Nifti1Image(arr, out_aff), out_path)
    if verbose:
        print(f'  wrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)')
        print(f'  TOTAL wall time: {(time.time()-t0):.0f} s')
    return out_path


def downsample_tiff_stack_to_nifti(tiff_paths, target_voxel_mm,
                                   native_voxel_mm, out_path,
                                   axis_flip_kwargs=None,
                                   voxel_signs=(-1, -1, +1),
                                   force=False, verbose=True):
    """Stream a TIFF stack, block-max-decimate to ``target_voxel_mm``,
    apply storage-axis flips, and write a clean RAS NIfTI.

    ``axis_flip_kwargs`` is a dict of kwargs passed to
    :func:`tuba.core.frame.apply_axis_flips` (e.g.
    ``{'slice_flip': True, 'row_flip': False, 'col_flip': True}``).
    """
    import tifffile
    if os.path.exists(out_path) and not force:
        return out_path
    factor = int(round(target_voxel_mm / native_voxel_mm))
    actual_vox_mm = factor * native_voxel_mm
    if abs(actual_vox_mm - target_voxel_mm) / target_voxel_mm > 0.05 and verbose:
        print(f'  WARNING: requested {target_voxel_mm} mm, factor {factor} gives '
              f'{actual_vox_mm:.4f} mm '
              f'({(actual_vox_mm-target_voxel_mm)/target_voxel_mm*100:.1f}% off)')
    n_slc = len(tiff_paths)
    n_slc_out = n_slc // factor
    if verbose:
        print(f'Downsampling {n_slc} TIFFs ({native_voxel_mm*1e3:.3f} um) '
              f'-> {n_slc_out} slices ({actual_vox_mm*1e3:.1f} um), '
              f'block-max factor {factor}')
    out_slices = []
    for chunk_start in range(0, n_slc_out * factor, factor):
        stack = np.stack([
            tifffile.imread(tiff_paths[chunk_start + s])
            for s in range(factor)
        ], axis=0)
        ds = block_max(stack, factor, axis=1)
        ds = block_max(ds, factor, axis=2)
        ds = ds.max(axis=0, keepdims=False)
        out_slices.append(ds)
        if verbose and (chunk_start // factor) % 16 == 0:
            print(f'  chunk {chunk_start//factor}/{n_slc_out}  '
                  f'out-shape so far {len(out_slices)}x{ds.shape}')
    vol = np.stack(out_slices, axis=0)
    if verbose:
        print(f'  raw downsampled shape: {vol.shape}, dtype {vol.dtype}, '
              f'min={vol.min()}, max={vol.max()}')
    if axis_flip_kwargs is not None:
        vol = frame.apply_axis_flips(vol, **axis_flip_kwargs)
        if verbose:
            print(f'  post-flip/transpose: {vol.shape}')
    affine = frame.affine_for_shape(vol.shape, actual_vox_mm, signs=voxel_signs)
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), out_path)
    if verbose:
        print(f'  wrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)')
        print(f'  affine =\n{affine}')
    return out_path
