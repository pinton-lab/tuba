"""Beam-aligned skull slab extraction in world-mm.

Geometry: ``apex + L*t1 + E*t2 + d*beam`` where ``(t1, t2)`` is built
by Gram-Schmidt from ``beam`` and a seed axis ([0,0,1] unless the
beam is near-vertical, then [1,0,0]). All inputs in TUBA's canonical
world frame (RAS, bone-centroid origin -- the manuscript §5
post-PCA-alignment frame for mouse).

The slab loader samples the species' skull intensity cache and (if
provided) a cavity binary mask via :class:`RegularGridInterpolator`,
then applies an intensity-to-acoustic ramp and optionally overrides
non-bone voxels inside the cavity with parenchymal acoustic
properties.

Acoustic-property mapping (manuscript §8, mouse Maga case): the
piecewise-linear ramp endpoints are anchored on *histogram-derived*
intensities, not on the global maximum -- choosing the cortical-bone
histogram mode as the upper anchor makes the mapping robust to
detector- and reconstruction-dependent absolute scale (the literature
endpoint, c=2900 m/s, rho=1900 kg/m^3, is bound to *this scan's*
cortical peak rather than to a numerically arbitrary saturation
value). This sacrifices absolute calibration -- a hydroxyapatite
phantom would let one bind a specific NRecon count to a specific HU
and thence to a calibrated density -- but yields a deterministic,
reproducible mapping with no free parameters once the histogram is
fixed.

Cavity infill (manuscript §8): the §6 cavity mask is air in the dry
source scan; non-bone voxels inside it are remapped to brain-
parenchymal acoustic properties (c=1540 m/s, rho=1040 kg/m^3 for the
mouse). No skull-distance constraint is required (cf. the DigiMouse
pipeline's 1.2 mm shell), because dry-skull air and cortical bone are
already well-separated by the intensity threshold alone.
"""
from dataclasses import dataclass
import os
import numpy as np
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator

from . import frame


@dataclass
class IntensityToAcousticRamp:
    """Piecewise-linear ``intensity -> (c, rho)`` mapping (manuscript §8).

    Voxels in ``[i_low, i_high]`` interpolate linearly between
    ``(c_water, rho_water)`` and ``(c_bone_max, rho_bone_max)``;
    below ``i_low`` -> water; above ``i_high`` -> cortical bone.

    Anchor on histogram-derived intensities (the cavity extractor's
    ``BONE_LOW`` for ``i_low``; the cortical-bone histogram mode for
    ``i_high``), not on the absolute scale, so the mapping is robust
    to detector / reconstruction differences. Defaults are mouse Maga
    cortical-bone literature values (c=2900 m/s, rho=1900 kg/m^3).
    """
    i_low: float
    i_high: float
    c_water: float = 1540.0
    rho_water: float = 1000.0
    c_bone_max: float = 2900.0
    rho_bone_max: float = 1900.0

    def __call__(self, intensity):
        t = np.clip((intensity - self.i_low) / (self.i_high - self.i_low),
                    0.0, 1.0).astype(np.float32)
        c = (self.c_water + t * (self.c_bone_max - self.c_water)).astype(np.float32)
        rho = (self.rho_water + t * (self.rho_bone_max - self.rho_water)).astype(np.float32)
        return c, rho


@dataclass
class CavityInfill:
    """Override acoustic properties of non-bone voxels inside the cavity.

    Defaults are brain parenchyma at 15 MHz: ``c=1540 m/s`` (matches
    water), ``rho=1040 kg/m^3`` (grey/white-matter mean; mouse §8).
    Attenuation is applied separately by the acoustic solver.
    """
    c_parenchyma: float = 1540.0
    rho_parenchyma: float = 1040.0


def slab_frame(beam_3d):
    """Right-handed orthonormal frame ``(beam, t1, t2)`` from a beam
    direction. Helper-axis seed is ``[0,0,1]`` unless ``|beam_z| > 0.9``,
    in which case ``[1,0,0]``.
    """
    beam = np.asarray(beam_3d, dtype=np.float64)
    beam = beam / np.linalg.norm(beam)
    helper = (np.array([0., 0., 1.]) if abs(beam[2]) < 0.9
              else np.array([1., 0., 0.]))
    t1 = np.cross(beam, helper); t1 /= np.linalg.norm(t1)
    t2 = np.cross(beam, t1)
    return beam, t1, t2


def sample_slab_world_mm(skull_array, skull_world_axes,
                         apex_world_mm, beam_3d,
                         slab_size_m, dx_target_m,
                         ramp: IntensityToAcousticRamp,
                         cavity_array=None, cavity_world_axes=None,
                         cavity_infill: CavityInfill = None,
                         verbose=True):
    """Primitive slab sampler operating on in-memory arrays.

    The skull and cavity arrays must already be in TUBA world-mm via
    their respective ``world_axes`` tuples (each is a 3-tuple of 1D
    ndarrays giving the per-axis world-mm coordinates, as produced by
    :func:`tuba.core.frame.affine_world_axes`).

    Used internally by :func:`load_slab` (NIfTI cache) and directly by
    species modules whose source is not a NIfTI (e.g. the human Halle
    pipeline reads NRRD and builds the affine manually).
    """
    if verbose:
        print(f'  skull shape {skull_array.shape}, '
              f'intensity range [{skull_array.min():.0f}, '
              f'{skull_array.max():.0f}]')
    interp = RegularGridInterpolator(
        skull_world_axes, skull_array,
        method='linear', bounds_error=False, fill_value=np.nan)
    cav_interp = None
    if cavity_array is not None and cavity_world_axes is not None:
        cav_interp = RegularGridInterpolator(
            cavity_world_axes, cavity_array,
            method='linear', bounds_error=False, fill_value=0.0)
        if verbose:
            print(f'  cavity shape {cavity_array.shape}')

    apex = np.asarray(apex_world_mm, dtype=np.float64)
    beam, t1, t2 = slab_frame(beam_3d)

    w_lat, w_elev, w_depth = slab_size_m
    dx = dx_target_m
    n_lat = int(round(w_lat / dx))
    n_elev = int(round(w_elev / dx))
    n_depth = int(round(w_depth / dx))

    s_lat = np.linspace(-w_lat / 2, w_lat / 2, n_lat)
    s_elev = np.linspace(-w_elev / 2, w_elev / 2, n_elev)
    s_depth = np.linspace(0.0, w_depth, n_depth)

    if verbose:
        print(f'  slab grid: {n_lat} x {n_elev} x {n_depth} '
              f'(lat x elev x depth), res={dx*1e3:.3f} mm')
        print(f'  beam={beam.round(3)}  t1={t1.round(3)}  t2={t2.round(3)}')

    hu = np.empty((n_lat, n_elev, n_depth), dtype=np.float32)
    in_cav = (np.zeros((n_lat, n_elev, n_depth), dtype=bool)
              if cav_interp is not None else None)
    n_valid = 0
    L_mm = s_lat[:, None] * 1e3
    E_mm = s_elev[None, :] * 1e3
    base_planar = (L_mm[..., None] * t1[None, None, :]
                   + E_mm[..., None] * t2[None, None, :])
    for kz in range(n_depth):
        d_mm = s_depth[kz] * 1e3
        P_mm_slice = (apex[None, None, :] + base_planar
                      + d_mm * beam[None, None, :])
        flat = P_mm_slice.reshape(-1, 3)
        sl = interp(flat).reshape(n_lat, n_elev)
        n_valid += int(np.count_nonzero(~np.isnan(sl)))
        hu[:, :, kz] = np.nan_to_num(sl, nan=0.0).astype(np.float32)
        if in_cav is not None:
            cs = cav_interp(flat).reshape(n_lat, n_elev)
            in_cav[:, :, kz] = cs > 0.5
    if verbose:
        print(f'  valid voxels: {n_valid}/{hu.size} '
              f'({100*n_valid/hu.size:.1f}%)')

    c_map, rho_map = ramp(hu)

    if in_cav is not None and cavity_infill is not None:
        bone = hu >= ramp.i_low
        parenchyma = in_cav & ~bone
        c_map[parenchyma] = cavity_infill.c_parenchyma
        rho_map[parenchyma] = cavity_infill.rho_parenchyma
        if verbose:
            print(f'  cavity parenchymal-fill fraction: '
                  f'{parenchyma.mean()*100:.2f}%')
            print(f'  bone fraction (>= i_low): {bone.mean()*100:.2f}%')

    if verbose:
        print(f'  c range:   [{c_map.min():.0f}, {c_map.max():.0f}] m/s')
        print(f'  rho range: [{rho_map.min():.0f}, {rho_map.max():.0f}] kg/m^3')

    dz_slab = float(s_depth[1] - s_depth[0])
    slab_frame_dict = {'beam': beam, 't1': t1, 't2': t2, 'apex_mm': apex}
    return c_map, rho_map, dz_slab, slab_frame_dict


def load_slab(cache_path, apex_world_mm, beam_3d,
              slab_size_m, dx_target_m,
              ramp: IntensityToAcousticRamp,
              cavity_mask_path=None,
              cavity_infill: CavityInfill = None,
              verbose=True):
    """Extract a beam-aligned slab from a skull intensity NIfTI cache.

    Reads the cache (and optional cavity NIfTI), constructs world
    axes from the NIfTI affines, then delegates to
    :func:`sample_slab_world_mm`.
    """
    if verbose:
        print(f'Loading cache from {cache_path}...')
    img = nib.load(cache_path)
    arr = np.asarray(img.dataobj).astype(np.float32)
    aff = img.affine
    voxel_mm = float(abs(aff[0, 0]))
    if verbose:
        print(f'  voxel {voxel_mm*1000:.1f} um')
    axes_world_mm = frame.affine_world_axes(arr.shape, aff)

    cav_arr = None
    cav_axes = None
    if cavity_mask_path and os.path.exists(cavity_mask_path):
        if verbose:
            print(f'Loading cavity mask from {cavity_mask_path}...')
        cav_img = nib.load(cavity_mask_path)
        cav_arr = np.asarray(cav_img.dataobj).astype(np.float32)
        cav_aff = cav_img.affine
        cav_axes = frame.affine_world_axes(cav_arr.shape, cav_aff)
        if verbose:
            print(f'  cavity voxel {abs(cav_aff[0,0])*1000:.1f} um, '
                  f'volume {int(cav_arr.sum()) * abs(cav_aff[0,0])**3:.0f} mm^3')

    return sample_slab_world_mm(
        skull_array=arr, skull_world_axes=axes_world_mm,
        apex_world_mm=apex_world_mm, beam_3d=beam_3d,
        slab_size_m=slab_size_m, dx_target_m=dx_target_m,
        ramp=ramp,
        cavity_array=cav_arr, cavity_world_axes=cav_axes,
        cavity_infill=cavity_infill, verbose=verbose,
    )
