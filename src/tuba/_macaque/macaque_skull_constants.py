"""Constants and orientation helpers for the AMNH M-87264 adult Macaca
mulatta dry-skull microCT (MorphoSource media 18083/18084, Delson
Primate Scans collection; Copes et al. Sci Data 2016 doi:10.1038/sdata.2016.1).

Source files
------------
  /celerina/gfp/mfs/monkeyCT/mmulatta_mcz/
    morphosource_media-id-000018084_download-c7068658/tiff_extracted/
      AMNH 87264 Macaca mulatta m_0000.tif ... m_2147.tif
  2148 reconstructed slices, 1359 x 1548 uint16, 60.613 um isotropic
  (per the MorphoSource media manifest).

Volume layout (verified by orientation_probe.py + numerical mandible
extent test on the raw TIFF stack):

  axis 0 (2148 slices)  rostro-caudal (A-P). 130.2 mm total. Storage
                        slice 0 = ROSTRAL (snout tip), 2147 = CAUDAL
                        (occipital). Verified from a per-storage-slice
                        TIFF render: clean closed brain-box ring (with
                        empty interior) is visible at storage slc 1500-
                        2050 -- HIGH slice indices. Storage slc 100-
                        1000 shows the snout/face cross-sections (thin
                        bone L-shapes, dental arches, zygomatic). The
                        mandible was scanned in the SAME volume but
                        laid as a separate articulation (storage slc
                        ~1100-1500 region had the mandible-only bone
                        mass peak before the cranium proper).
  axis 1 (1359 rows)    D-V. 82.4 mm. Storage row 0 = VENTRAL,
                        1358 = DORSAL. Verified from per-row bone
                        count summed across brain-box slices 1500-
                        2050: bone is heaviest at rows 300-500 (thick
                        basicranium) and lighter at rows 1000-1200
                        (thin parietal vault), with a U-shaped dip in
                        the middle (the brain cavity void).
  axis 2 (1548 cols)    L-R. 93.8 mm. L-R sign cannot be determined
                        from the unregistered volume; mirror error is
                        caught by the SyN-to-NMT registration.

The convention used downstream
------------------------------
The Maga workflow assumes storage = (slice=AP, row=LR, col=DV). This
macaque stores (slice=AP, row=DV, col=LR) -- rows and cols are swapped.
We swap them once at load time (SWAP_ROW_COL_AT_LOAD) so the rest of
the pipeline (SLICE_FLIP / ROW_FLIP / COL_FLIP semantics, transpose
to (i, j, k) = (LR, AP, DV)) is identical to the Maga case.

The convention used downstream
------------------------------
We write a NIfTI whose RAS world coordinates are anatomical:
  +x = right, +y = anterior, +z = superior (dorsal)

This mirrors mouse_therapy/registration/maga_skull_constants.py so the
downstream cavity extractor, SyN registration, slab loader, and
placement helper can all be ported with the smallest possible change.
"""
import glob
import os
import shutil
import subprocess
import numpy as np
import nibabel as nib
import tifffile
from scipy import ndimage


def _save_nifti_pigz(nii_img, out_path):
    """Save a NIfTI image to ``out_path`` (.nii.gz). Uses parallel gzip
    (pigz) when available -- pays ~30 s for the uncompressed .nii write
    plus ~10-30 s for pigz compression on ~64 threads, vs the ~25 min
    single-thread zlib path that nibabel uses by default for ~80 GB
    uint16 volumes. Falls back to nibabel's built-in gzip when pigz is
    not installed."""
    if shutil.which('pigz') is None or not out_path.endswith('.nii.gz'):
        nib.save(nii_img, out_path)
        return
    resolved = os.path.realpath(out_path)
    tmp_nii = resolved[:-3]  # drop .gz
    nib.save(nii_img, tmp_nii)
    n_threads = min(64, os.cpu_count() or 1)
    subprocess.run(['pigz', '-f', '-p', str(n_threads), tmp_nii], check=True)

REG_DIR = os.environ.get(
    'TUBA_MACAQUE_REG_DIR',
    os.path.expanduser('~/.cache/tuba/macaque/registration'))
os.makedirs(REG_DIR, exist_ok=True)

# --- source -----------------------------------------------------------------
# Raw AMNH M-87264 microCT TIFF stack location. Configurable via
# ``TUBA_MACAQUE_SOURCE_DIR``; falls back to the Copes 2016 layout under
# ``~/.cache/tuba/macaque/source/`` (see fetch_macaque.py).
SOURCE_DIR = os.environ.get(
    'TUBA_MACAQUE_SOURCE_DIR',
    '/celerina/gfp/mfs/monkeyCT/mmulatta_mcz/'
    'morphosource_media-id-000018084_download-c7068658/tiff_extracted')
TIFF_GLOB = os.environ.get(
    'TUBA_MACAQUE_TIFF_GLOB',
    os.path.join(SOURCE_DIR, 'AMNH 87264 Macaca mulatta m_*.tif'))

# --- NMT v2 atlas root (mirrors fetch_nmt.py NMT_DEST) ---------------------
NMT_DEST = os.environ.get(
    'NMT_DEST',
    os.path.expanduser('~/.cache/tuba/macaque/atlas/nmt_v2'))
NMT_DIR = os.path.join(NMT_DEST, 'NMT_v2.0_sym', 'NMT_v2.0_sym')
# Drop back to legacy monkey_brain location if the cache hasn't been
# populated yet, so the existing pipeline keeps working during port.
if not os.path.isfile(os.path.join(NMT_DIR, 'NMT_v2.0_sym.nii.gz')):
    _legacy_nmt = ('/celerina/gfp/mfs/monkey_brain/templates/nmt_v2/'
                   'NMT_v2.0_sym/NMT_v2.0_sym')
    if os.path.isfile(os.path.join(_legacy_nmt, 'NMT_v2.0_sym.nii.gz')):
        NMT_DIR = _legacy_nmt

# --- physical parameters ----------------------------------------------------
NATIVE_VOXEL_MM = 0.060613            # MorphoSource manifest: 0.06061318 mm
# 250 um cache exactly matches NMT v2.0_sym (0.25 mm). At 500 kHz (lambda
# in cortical bone ~5.8 mm) this gives ~23 voxels per wavelength -- enough
# resolution for the cavity contour, SyN moving image, and slab loader
# all from a single cache. The factor (250/60.613 = 4.124) rounds to 4
# giving an actual pitch of 4 * 60.613 = 242.45 um -- 3% off, acceptable.
DOWNSAMPLE_VOXEL_MM = 0.250

# --- orientation flags ------------------------------------------------------
# 1) Storage is (slice=AP, row=DV, col=LR), opposite of the Maga assumption
# (slice=AP, row=LR, col=DV). We swap rows and cols once at load time so
# the rest of the pipeline matches Maga semantics.
SWAP_ROW_COL_AT_LOAD = True
# 2) AFTER the swap (so storage is now slice=AP, row=LR, col=DV), apply
#    these flips to bring (i, j, k) into RAS:
#       i=0 -> right-most, j=0 -> anterior, k=0 -> ventral.
#    SLICE_FLIP=False : storage_slice 0 (rostral, snout tip) is already
#                        at j=0; affine maps j=0 -> +y_max -> anterior.
#    ROW_FLIP=False   : L-R sign undetermined; mirror caught by SyN to NMT.
#    COL_FLIP=False   : pre-swap row 0 (ventral, basicranium per per-row
#                        bone-count probe over brain-box slices 1500-
#                        2050) -> post-swap col 0 -> k=0 -> z_min in
#                        RAS (= ventral, since +z=dorsal).
SLICE_FLIP = False
ROW_FLIP = False
COL_FLIP = False
# Constrained PCA on the cranium-only cropped cache showed v_LR (=
# v_AP x v_DV via right-hand rule) points -e_i in the storage frame.
# A pre-rotation ROW_FLIP would NOT fix this (the i-component of the
# cross product is invariant under storage axis flips that are
# perpendicular to the cross product). We instead apply a post-rotation
# flip of axis 0 in align_*_to_axes when this flag is True.
POST_ROTATE_LR_FLIP = True

# --- bone threshold ---------------------------------------------------------
# uint16 reconstruction. The intensity histogram (intensity_histogram.png)
# resolves three populations:
#   air                  : intensity 0       (~10^8 voxels)
#   mounting / soft      : peak at ~5000    (~10^5 voxels)
#   bone (broad)         : 10k - 45k        (~10^5 voxels per bin)
#     trabecular peak    : ~32000
#     dental enamel tail : 40k - 44k
# A threshold of 10000 cleanly separates bone from mounting material.
BONE_THRESH = 10000

# --- pre-alignment rotation -------------------------------------------------
# Rotation about the volume centre in index space, intrinsic ZYX
# convention, applied to column vectors. Angles set to zero by default;
# find_alignment_rotation() derives them from PCA on the dense bone
# cloud (intensity > ALIGN_PCA_THRESH).
# Derived by find_cranium_alignment.py (constrained PCA on cropped
# cranium-only raw 250 um cache). The cranium-only PCA has near-
# degenerate L-R / D-V eigenvalues (variance ratio 1.14:1.0), so the
# secondary eigenvectors are taken from a basicranium-vs-vault
# voxel-count asymmetry sweep about the AP axis. The L-R chirality of
# the resulting frame requires POST_ROTATE_LR_FLIP=True (above).
#
# (Legacy v1 angles, derived by unconstrained PCA on the cranium+
# mandible volume, were Rx=-9.45, Ry=+10.84, Rz=-4.40 -- the mandible's
# extra bone mass elongated the LR variance enough to make standard PCA
# stable, but the resulting alignment was biased by ~5 deg per axis
# because the mandible was not articulated to the cranium in this scan.)
ALIGN_RX_DEG = +95.94
ALIGN_RY_DEG = -59.69
ALIGN_RZ_DEG = -99.12
# (Computed with ROW_FLIP=False; the L-R chirality is then handled by
# the post-rotation flip POST_ROTATE_LR_FLIP=True above.)

# Threshold for PCA alignment. The Maga adult-mouse case used 30000.
# For this macaque scan (which includes the mandible) we use 25000 to
# capture cortical bone (peak at ~32000) and dense parts of the
# basicranium / dental arch -- bilaterally symmetric structures
# whose centroid gives a clean R-L reference. The mandible's enamel
# tail (40-44k) is INCLUDED, which biases the AP centroid rostrally
# but does not significantly tilt the principal axes because the
# mandible is roughly co-aligned with the cranium's AP/LR/DV directions.
ALIGN_PCA_THRESH = 25000

# --- output paths -----------------------------------------------------------
MACAQUE_RAS_NIFTI = os.path.join(REG_DIR, 'macaque_skull_250um.nii.gz')
MACAQUE_RAS_NIFTI_ALIGNED = os.path.join(REG_DIR, 'macaque_skull_aligned_250um.nii.gz')
MACAQUE_RAS_NATIVE_ALIGNED = os.path.join(REG_DIR, 'macaque_skull_aligned_native.nii.gz')
# 250 um cache derived from the *aligned native* via block-max factor 4.
# This inherits the native's affine origin, so cavity masks extracted from
# this volume share a world coordinate frame with the native and NMT
# overlays. Mirrors the MAGA_RAS_ALIGNED_200UM design.
MACAQUE_RAS_ALIGNED_250UM = os.path.join(REG_DIR,
                                          'macaque_skull_aligned_native_250um.nii.gz')


def _tiff_paths():
    paths = sorted(glob.glob(TIFF_GLOB))
    if not paths:
        raise FileNotFoundError(f'No TIFFs match {TIFF_GLOB}')
    return paths


def _affine_for_shape(shape, voxel_mm):
    """RAS world from native voxel indices.

    Native storage (after applying the FLIP flags):
        i=0 -> right-most   (+x = -i*vox + const)
        j=0 -> anterior     (+y = -j*vox + const, since +y = anterior)
        k=0 -> ventral      (+z = +k*vox + const)
    Centre of the volume is mapped to the world origin.
    """
    n_i, n_j, n_k = shape
    A = np.eye(4)
    A[0, 0] = -voxel_mm
    A[1, 1] = -voxel_mm
    A[2, 2] = +voxel_mm
    A[0, 3] = (n_i - 1) * voxel_mm * 0.5
    A[1, 3] = (n_j - 1) * voxel_mm * 0.5
    A[2, 3] = -(n_k - 1) * voxel_mm * 0.5
    return A


def _apply_axis_flips(arr):
    """Apply SWAP_ROW_COL_AT_LOAD then SLICE_FLIP / ROW_FLIP / COL_FLIP
    to a (slice, row, col) array; finally transpose to (i, j, k).

    The macaque scan stores (slice=AP, row=DV, col=LR), opposite of the
    Maga assumption (slice=AP, row=LR, col=DV). We swap rows and cols
    once at the start so the rest of this function (and the SLICE_FLIP
    semantics in the validation report) match Maga exactly. After the
    swap and flips, axes are (slice=AP, row=LR, col=DV); the closing
    transpose(1, 0, 2) yields (LR, AP, DV) = (i, j, k).
    """
    if SWAP_ROW_COL_AT_LOAD:
        arr = arr.transpose(0, 2, 1)
    if SLICE_FLIP:
        arr = arr[::-1, :, :]
    if ROW_FLIP:
        arr = arr[:, ::-1, :]
    if COL_FLIP:
        arr = arr[:, :, ::-1]
    arr = arr.transpose(1, 0, 2)
    return arr


def _block_max(a, factor, axis):
    """Reduce length along `axis` by `factor` via block-max. Truncates."""
    n = a.shape[axis]
    n_out = n // factor
    if n_out == 0:
        return a.take(indices=[], axis=axis)
    trim_slc = [slice(None)] * a.ndim
    trim_slc[axis] = slice(0, n_out * factor)
    a = a[tuple(trim_slc)]
    new_shape = list(a.shape)
    new_shape[axis:axis+1] = [n_out, factor]
    return a.reshape(new_shape).max(axis=axis+1)


def downsample_to(voxel_mm, out_path, force=False):
    """Stream the TIFF stack, block-max-decimate to `voxel_mm`, write a
    clean RAS NIfTI.  Returns the output path (cached if it already exists).
    """
    if os.path.exists(out_path) and not force:
        return out_path

    factor = int(round(voxel_mm / NATIVE_VOXEL_MM))
    actual_vox_mm = factor * NATIVE_VOXEL_MM
    if abs(actual_vox_mm - voxel_mm) / voxel_mm > 0.05:
        print(f'  WARNING: requested {voxel_mm} mm, factor {factor} gives '
              f'{actual_vox_mm:.4f} mm '
              f'({(actual_vox_mm-voxel_mm)/voxel_mm*100:.1f}% off)')

    paths = _tiff_paths()
    n_slc = len(paths)
    n_slc_out = n_slc // factor
    print(f'Downsampling {n_slc} TIFFs ({NATIVE_VOXEL_MM*1000:.2f} um) -> '
          f'{n_slc_out} slices ({actual_vox_mm*1000:.1f} um), '
          f'block-max factor {factor}')

    out_slices = []
    for chunk_start in range(0, n_slc_out * factor, factor):
        stack = np.stack([
            tifffile.imread(paths[chunk_start + s])
            for s in range(factor)
        ], axis=0)
        ds = _block_max(stack, factor, axis=1)
        ds = _block_max(ds, factor, axis=2)
        ds = ds.max(axis=0, keepdims=False)
        out_slices.append(ds)
        if (chunk_start // factor) % 32 == 0:
            print(f'  chunk {chunk_start//factor}/{n_slc_out}  '
                  f'out-shape so far {len(out_slices)}x{ds.shape}')

    vol = np.stack(out_slices, axis=0)
    print(f'  raw downsampled shape: {vol.shape}, dtype {vol.dtype}, '
          f'min={vol.min()}, max={vol.max()}')

    vol = _apply_axis_flips(vol)
    print(f'  post-flip/transpose: {vol.shape}')

    affine = _affine_for_shape(vol.shape, actual_vox_mm)
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), out_path)
    print(f'  wrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)')
    print(f'  affine =\n{affine}')
    return out_path


def downsample_to_250um(force=False):
    return downsample_to(DOWNSAMPLE_VOXEL_MM, MACAQUE_RAS_NIFTI, force=force)


def _rot_matrix_zyx(ax_deg, ay_deg, az_deg):
    ax, ay, az = np.deg2rad([ax_deg, ay_deg, az_deg])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def find_alignment_rotation(in_path=MACAQUE_RAS_NIFTI,
                              bone_thresh=ALIGN_PCA_THRESH):
    """PCA on the bone voxel cloud to find the rotation that aligns the
    skull's anatomical axes with the storage index axes.

    For a macaque cranium with mandible attached, the eigenvalue ordering
    by anatomical extent (~85-130 mm A-P x ~94 mm L-R x ~82 mm D-V) is
    very close to (largest, middle, smallest) but the largest two are
    nearly degenerate -- watch for axis swapping if the threshold isn't
    high enough to suppress the mandible's contribution.

        largest variance  -> A-P principal axis  -> target j
        middle  variance  -> R-L principal axis  -> target i
        smallest variance -> D-V principal axis  -> target k

    Returns (ax_deg, ay_deg, az_deg) in the ZYX intrinsic convention.
    """
    img = nib.load(in_path)
    vol = img.get_fdata()
    bone = np.argwhere(vol > bone_thresh).astype(np.float64)
    n_bone = len(bone)
    if n_bone == 0:
        raise RuntimeError(f'no bone voxels above {bone_thresh}')
    centroid = bone.mean(axis=0)
    pts = bone - centroid
    cov = (pts.T @ pts) / n_bone

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    v_AP, v_RL, v_DV = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]
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
        print(f'  det(V) was negative; flipped column {worst}')

    R = V.T

    ay = np.arcsin(-R[2, 0])
    ax = np.arctan2(R[2, 1], R[2, 2])
    az = np.arctan2(R[1, 0], R[0, 0])
    ax_d, ay_d, az_d = np.rad2deg([ax, ay, az])
    print(f'  Euler (ZYX intrinsic): Rx={ax_d:.2f}, Ry={ay_d:.2f}, '
          f'Rz={az_d:.2f} deg')
    R_rt = _rot_matrix_zyx(ax_d, ay_d, az_d)
    print(f'  round-trip ||R - Rz@Ry@Rx||_F = {np.linalg.norm(R - R_rt):.2e}')
    return ax_d, ay_d, az_d


def align_skull_to_axes(in_path=MACAQUE_RAS_NIFTI,
                          out_path=MACAQUE_RAS_NIFTI_ALIGNED,
                          ax_deg=ALIGN_RX_DEG, ay_deg=ALIGN_RY_DEG,
                          az_deg=ALIGN_RZ_DEG, force=False, order=1,
                          centre_skull=True, bone_thresh=ALIGN_PCA_THRESH,
                          pad_voxels=12):
    """Rotate a RAS-aligned skull NIfTI so the anatomical axes align with
    the voxel index axes. Same logic as
    maga_skull_constants.align_skull_to_axes."""
    if os.path.exists(out_path) and not force:
        return out_path
    img = nib.load(in_path)
    vol = img.get_fdata().astype(np.float32)
    R = _rot_matrix_zyx(ax_deg, ay_deg, az_deg)
    R_inv = R.T

    in_shape = np.array(vol.shape)
    out_shape = tuple(int(s + 2 * pad_voxels) for s in in_shape)
    out_centre = (np.array(out_shape) - 1) / 2.0

    if centre_skull:
        bone = np.argwhere(vol > bone_thresh).astype(np.float64)
        if len(bone) == 0:
            raise RuntimeError(f'no bone voxels above {bone_thresh}')
        c_in = bone.mean(axis=0)
        offset = c_in - R_inv @ out_centre
    else:
        c_in = (in_shape - 1) / 2.0
        offset = c_in - R_inv @ out_centre

    rotated = ndimage.affine_transform(
        vol, R_inv, offset=offset, output_shape=out_shape,
        order=order, mode='constant', cval=0.0, prefilter=False)

    if POST_ROTATE_LR_FLIP:
        # The constrained-PCA-derived rotation puts the cranium in the
        # output frame but with anatomical-L on the +i side. Flip axis 0
        # so the +x = right convention holds in the output. The affine
        # stays the same (storage i=0 still maps to +x_max world via
        # aff[0,0] < 0); after the flip, what's at storage i=0 is the
        # cranium's anatomical right.
        rotated = rotated[::-1, :, :]
        print(f'  applied post-rotation L-R flip (POST_ROTATE_LR_FLIP=True)')

    voxel_mm = float(abs(img.affine[0, 0]))
    out_affine = _affine_for_shape(out_shape, voxel_mm)

    print(f'Rotated {in_path} by (Rx={ax_deg}, Ry={ay_deg}, Rz={az_deg}) deg')
    print(f'  in: shape={tuple(in_shape)}, max={vol.max():.0f}, '
          f'bone centroid={c_in if centre_skull else "n/a"}')
    print(f'  out: shape={out_shape} (pad={pad_voxels}), max={rotated.max():.0f}')
    nib.save(nib.Nifti1Image(rotated.astype(np.float32), out_affine), out_path)
    print(f'  wrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)')
    return out_path


def align_native_to_axes(out_path=MACAQUE_RAS_NATIVE_ALIGNED,
                           ax_deg=ALIGN_RX_DEG, ay_deg=ALIGN_RY_DEG,
                           az_deg=ALIGN_RZ_DEG, force=False,
                           bone_thresh=ALIGN_PCA_THRESH,
                           pad_voxels=400):
    """Stream the native TIFFs, apply axis flips + transpose to get RAS
    storage order, then rotate by the same ZYX intrinsic angles used for
    the 250 um cache and translate so the bone centroid lands at the
    volume centre. Saves as a uint16 NIfTI at the native 60.6 um pitch.

    pad_voxels=400 (~24 mm at 60.6 um) gives the downstream
    refine_native.py rotation+translation (worst case -27 mm sagittal)
    enough source-FOV headroom to avoid cval=0 truncation of the
    inferior basicranium. The earlier default of 64 vox (~3.9 mm) left
    only ~10 mm of clearance and produced the truncated skull visible
    in the macaque manuscript Fig 5 QC.

    Memory: peak ~3 x 30 GB (input + working float32 + output uint16).
    Wall time dominated by the single-threaded scipy affine_transform
    pass."""
    if os.path.exists(out_path) and not force:
        return out_path
    import time
    t0 = time.time()

    paths = _tiff_paths()
    n_slc = len(paths)
    print(f'[1/4] Loading {n_slc} native TIFFs into a uint16 array...')
    sample = tifffile.imread(paths[0])
    raw = np.empty((n_slc, sample.shape[0], sample.shape[1]), dtype=np.uint16)
    raw[0] = sample
    for k in range(1, n_slc):
        raw[k] = tifffile.imread(paths[k])
        if k % 200 == 0:
            print(f'  {k}/{n_slc}  ({(time.time()-t0):.0f} s elapsed)')
    print(f'  loaded: shape={raw.shape}, dtype={raw.dtype}, '
          f'size={raw.nbytes/1e9:.1f} GB, t={(time.time()-t0):.0f} s')

    print('[2/4] Applying axis flips and transpose to RAS index space...')
    flipped = _apply_axis_flips(raw)
    del raw
    flipped = np.ascontiguousarray(flipped)
    print(f'  flipped/transposed: shape={flipped.shape}, '
          f'size={flipped.nbytes/1e9:.1f} GB, t={(time.time()-t0):.0f} s')

    print(f'[3/4] Finding bone centroid (thresh > {bone_thresh})...')
    bone = np.argwhere(flipped > bone_thresh)
    c_in = bone.mean(axis=0).astype(np.float64)
    print(f'  {len(bone):,} bone voxels; centroid = {c_in}')
    del bone

    R = _rot_matrix_zyx(ax_deg, ay_deg, az_deg)
    R_inv = R.T
    in_shape = np.array(flipped.shape)
    out_shape = tuple(int(s + 2 * pad_voxels) for s in in_shape)
    out_centre = (np.array(out_shape) - 1) / 2.0
    offset = c_in - R_inv @ out_centre
    print(f'  rotation R = Rz({az_deg}) @ Ry({ay_deg}) @ Rx({ax_deg}) '
          f'(intrinsic ZYX), pad={pad_voxels} -> out_shape={out_shape}')

    print('[3/4] Rotating native volume via scipy.ndimage.affine_transform '
          '(uint16 in/out, order=1)...')
    rotated = ndimage.affine_transform(
        flipped, R_inv, offset=offset, output_shape=out_shape,
        order=1, mode='constant', cval=0, prefilter=False)
    print(f'  rotated: shape={rotated.shape}, dtype={rotated.dtype}, '
          f'max={rotated.max()}, t={(time.time()-t0):.0f} s')
    del flipped

    if POST_ROTATE_LR_FLIP:
        rotated = rotated[::-1, :, :]
        print(f'  applied post-rotation L-R flip')

    print('[4/4] Writing NIfTI (pigz if available)...')
    affine = _affine_for_shape(rotated.shape, NATIVE_VOXEL_MM)
    _save_nifti_pigz(nib.Nifti1Image(rotated, affine), out_path)
    print(f'  wrote {out_path}  ({os.path.getsize(out_path)/1e9:.2f} GB)')
    print(f'  TOTAL wall time: {(time.time()-t0)/60:.1f} min')
    return out_path


def downsample_aligned_to(in_path, out_path, factor, force=False):
    """Block-max downsample the aligned native by ``factor`` to produce a
    coarser cache that inherits the native's RAS origin and axis-flip
    signs (so masks/overlays land at the correct world coords)."""
    if os.path.exists(out_path) and not force:
        return out_path
    import time
    t0 = time.time()
    print(f'Loading {in_path}...')
    img = nib.load(in_path)
    arr = np.asarray(img.dataobj, dtype=np.uint16)
    in_aff = img.affine
    in_spacing = float(abs(in_aff[0, 0]))
    print(f'  in shape={arr.shape}, in spacing={in_spacing*1000:.3f} um, '
          f'size={arr.nbytes/1e9:.1f} GB, t={(time.time()-t0):.0f} s')

    out_spacing = in_spacing * factor
    print(f'Block-max factor={factor} -> out spacing={out_spacing*1000:.3f} um')
    arr = _block_max(arr, factor, axis=0)
    arr = _block_max(arr, factor, axis=1)
    arr = _block_max(arr, factor, axis=2)
    print(f'  downsampled shape={arr.shape}, t={(time.time()-t0):.0f} s')

    out_aff = in_aff.copy()
    out_aff[0, 0] = np.sign(in_aff[0, 0]) * out_spacing
    out_aff[1, 1] = np.sign(in_aff[1, 1]) * out_spacing
    out_aff[2, 2] = np.sign(in_aff[2, 2]) * out_spacing
    nib.save(nib.Nifti1Image(arr, out_aff), out_path)
    print(f'  wrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)')
    print(f'  TOTAL wall time: {(time.time()-t0):.0f} s')
    return out_path


def downsample_aligned_to_250um(in_path=MACAQUE_RAS_NATIVE_ALIGNED,
                                  out_path=MACAQUE_RAS_ALIGNED_250UM,
                                  factor=4, force=False):
    """Block-max downsample the aligned native (60.6 um) by factor 4 to
    produce a 242.45 um cache that inherits the native's affine. This
    is the moving-image cache for SyN registration to NMT v2 (0.25 mm)
    and the source cache for the cavity extractor."""
    return downsample_aligned_to(in_path, out_path, factor=factor,
                                  force=force)


if __name__ == '__main__':
    downsample_to_250um()
