"""3D-aware beam-aligned skull slab extraction from the AMNH macaque
microCT.

Direct port of mouse_therapy/registration/maga_skull_slab_3d.py with:
  * piecewise-linear intensity -> (c, rho) ramp anchored at
    I_LOW = 10000  (BONE_THRESH for the AMNH macaque scan, separates
                    bone from the 5000 mounting plateau)
    I_HIGH = 32000 (cortical-bone histogram mode for this scan;
                    higher than the Maga 25000 because adult macaque
                    cortical bone is denser than adult mouse vault)
  * dry-skull endocranial cavity: mounting material is below I_LOW so
    no shell-distance constraint is needed (matched-fluid coupling
    assumption);
  * cavity infill: voxels inside macaque_cranial_cavity that are not
    classified as bone get brain-parenchymal acoustic properties.

Returns (c_map, rho_map, dz_slab, frame). Same signature as the
Maga and DigiMouse loaders so the FDTD/angular-spectrum driver can
remain a one-line skull-source switch.
"""
import os
import sys
import numpy as np
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator

REG_DIR = os.path.dirname(__file__)
from .macaque_skull_constants import MACAQUE_RAS_NATIVE_ALIGNED  # noqa: F401

MACAQUE_RAS_ALIGNED_250UM = os.path.join(
    REG_DIR, 'macaque_skull_aligned_native_250um.nii.gz')
MACAQUE_CAVITY_NII = os.path.join(REG_DIR, 'macaque_cranial_cavity.nii.gz')

# Default slab cache: the 250 um aligned cache (~11 MB float32). At
# 500 kHz the FDTD grid is ~880 um (lambda_brain/3.5), so the 250 um
# cache resolves the bone wall at ~3.5x the simulation grid -- ample
# detail for the slab interpolation. The native 60.6 um cache (~10 GB
# float32) is available for higher-detail sampling but unnecessary
# at this frequency.
DEFAULT_CACHE = MACAQUE_RAS_ALIGNED_250UM

# Piecewise-linear intensity -> acoustic ramp. I_LOW matches the
# cavity extractor's BONE_LOW; I_HIGH is the cortical-bone histogram
# mode determined from intensity_histogram.png (~32000).
I_LOW = 10000.0
I_HIGH = 32000.0

C_WATER = 1540.0
RHO_WATER = 1000.0
C_BONE_MAX = 2900.0       # cortical bone endpoint at I_HIGH
RHO_BONE_MAX = 1900.0

# Endocranial cavity infill (brain parenchyma at 500 kHz).
C_PARENCHYMA = 1540.0
RHO_PARENCHYMA = 1040.0


def _slab_frame(beam_3d):
    beam = np.asarray(beam_3d, dtype=np.float64)
    beam = beam / np.linalg.norm(beam)
    helper = np.array([0., 0., 1.]) if abs(beam[2]) < 0.9 \
        else np.array([1., 0., 0.])
    t1 = np.cross(beam, helper); t1 /= np.linalg.norm(t1)
    t2 = np.cross(beam, t1)
    return beam, t1, t2


def _intensity_to_acoustic(hu):
    t = np.clip((hu - I_LOW) / (I_HIGH - I_LOW), 0.0, 1.0).astype(np.float32)
    c = (C_WATER + t * (C_BONE_MAX - C_WATER)).astype(np.float32)
    rho = (RHO_WATER + t * (RHO_BONE_MAX - RHO_WATER)).astype(np.float32)
    return c, rho


def _affine_world_axes(arr_shape, affine):
    """Per-axis world coordinates implied by a diagonal-affine NIfTI."""
    return tuple(affine[d, 3] + np.arange(arr_shape[d]) * affine[d, d]
                 for d in range(3))


def load_macaque_skull_slab_3d(apex_voxel_mm, beam_3d,
                                  slab_size_m, dx_target_m,
                                  cache_path=DEFAULT_CACHE,
                                  cavity_mask_path=MACAQUE_CAVITY_NII,
                                  chatter=True):
    """Extract a beam-aligned slab from the AMNH macaque skull.

    Parameters
    ----------
    apex_voxel_mm : array-like (3,)
        Slab origin in macaque aligned-native world-mm (origin at the
        skull bone-centroid, axes RAS).
    beam_3d : array-like (3,)
        Unit beam direction in the same world frame.
    slab_size_m : tuple (lateral, elevation, depth) in metres.
    dx_target_m : float, slab voxel size in metres.
    cache_path : str, optional
        Macaque intensity cache to sample. Default 250 um.
    cavity_mask_path : str or None, optional
        Endocranial cavity mask. Voxels inside the cavity not
        classified as bone get parenchymal acoustic properties.

    Returns
    -------
    c_map, rho_map : (n_lat, n_elev, n_depth) float32
    dz_slab : float (m)
    frame : dict with 'beam', 't1', 't2', 'apex_mm'
    """
    if chatter:
        print(f'Loading macaque cache from {cache_path}...')
    img = nib.load(cache_path)
    arr = np.asarray(img.dataobj).astype(np.float32)
    aff = img.affine
    voxel_mm = float(abs(aff[0, 0]))
    if chatter:
        print(f'  shape {arr.shape}, voxel {voxel_mm*1000:.1f} um, '
              f'intensity range [{arr.min():.0f}, {arr.max():.0f}]')

    axes_world_mm = _affine_world_axes(arr.shape, aff)
    interp = RegularGridInterpolator(
        axes_world_mm, arr,
        method='linear', bounds_error=False, fill_value=np.nan)

    cav_interp = None
    if cavity_mask_path and os.path.exists(cavity_mask_path):
        if chatter:
            print(f'Loading cavity mask from {cavity_mask_path}...')
        cav_img = nib.load(cavity_mask_path)
        cav_arr = np.asarray(cav_img.dataobj).astype(np.float32)
        cav_aff = cav_img.affine
        cav_axes = _affine_world_axes(cav_arr.shape, cav_aff)
        cav_interp = RegularGridInterpolator(
            cav_axes, cav_arr,
            method='linear', bounds_error=False, fill_value=0.0)
        if chatter:
            print(f'  cavity shape {cav_arr.shape}, '
                  f'voxel {abs(cav_aff[0,0])*1000:.1f} um, '
                  f'volume {int(cav_arr.sum()) * abs(cav_aff[0,0])**3 / 1000:.1f} mL')

    apex = np.asarray(apex_voxel_mm, dtype=np.float64)
    beam, t1, t2 = _slab_frame(beam_3d)

    w_lat, w_elev, w_depth = slab_size_m
    dx = dx_target_m
    n_lat = int(round(w_lat / dx))
    n_elev = int(round(w_elev / dx))
    n_depth = int(round(w_depth / dx))

    s_lat = np.linspace(-w_lat / 2, w_lat / 2, n_lat)
    s_elev = np.linspace(-w_elev / 2, w_elev / 2, n_elev)
    s_depth = np.linspace(0.0, w_depth, n_depth)

    if chatter:
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
    if chatter:
        print(f'  valid voxels: {n_valid}/{hu.size} '
              f'({100*n_valid/hu.size:.1f}%)')

    c_map, rho_map = _intensity_to_acoustic(hu)

    if in_cav is not None:
        bone = hu >= I_LOW
        parenchyma = in_cav & ~bone
        c_map[parenchyma] = C_PARENCHYMA
        rho_map[parenchyma] = RHO_PARENCHYMA
        if chatter:
            print(f'  cavity parenchymal-fill fraction: '
                  f'{parenchyma.mean()*100:.2f}%')
            print(f'  bone fraction (>= I_LOW): {bone.mean()*100:.2f}%')

    if chatter:
        print(f'  c range:   [{c_map.min():.0f}, {c_map.max():.0f}] m/s')
        print(f'  rho range: [{rho_map.min():.0f}, {rho_map.max():.0f}] kg/m^3')

    dz_slab = float(s_depth[1] - s_depth[0])
    frame = {'beam': beam, 't1': t1, 't2': t2, 'apex_mm': apex}
    return c_map, rho_map, dz_slab, frame
