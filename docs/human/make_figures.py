"""Generate the three figures for the human-pipeline manuscript at
native Halle resolution (0.125 mm) for fig1 and fig3.

  fig1_halle_native.png    Halle dry-skull orthoslices at NATIVE 0.125 mm.
  fig2_mni_template.png    MNI152 T1 orthoslices with brain-mask contour
                           (1 mm native MNI resolution).
  fig3_skull_atlas.png     Native 0.125 mm Halle bone (greyscale) with
                           warped MNI T1 + brain-mask overlay (the MNI
                           is 1 mm; matplotlib's `extent` keyword
                           handles the resolution mismatch -- both
                           layers are drawn on the same world-mm extent).

Saved at 600 dpi so the native 1700-px-along-AP cortical detail is
preserved when the figure is rasterised into the PDF (mouse manuscript
uses 1200 dpi for its 3882-px-along-AP figures; the human is ~2.3x
coarser native pixel count so 600 dpi gives equivalent fidelity).
"""
import os
import time
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths obey the same TUBA_HUMAN_REG_DIR / TUBA_HUMAN_NRRD_PATH /
# NILEARN_DATA env vars used by tuba.species.human; defaults fall
# back to ~/.cache/tuba/human and ~/nilearn_data.
REG_DIR = os.environ.get(
    'TUBA_HUMAN_REG_DIR',
    os.path.expanduser('~/.cache/tuba/human/registration'))
TEMPLATE_DIR = os.environ.get(
    'TUBA_HUMAN_TEMPLATE_DIR',
    os.path.expanduser('~/.cache/tuba/human/templates/mni152_icbm_2009a'))
NRRD_PATH = os.environ.get(
    'TUBA_HUMAN_NRRD_PATH',
    os.path.expanduser('~/.cache/tuba/human/source/halle_skull.nrrd'))
_NILEARN = os.environ.get('NILEARN_DATA',
                          os.path.expanduser('~/nilearn_data'))
HO_CORTL_PATH = os.path.join(_NILEARN, 'fsl/data/atlases/HarvardOxford/'
                             'HarvardOxford-cortl-maxprob-thr25-1mm.nii.gz')
HO_SUB_PATH = os.path.join(_NILEARN, 'fsl/data/atlases/HarvardOxford/'
                           'HarvardOxford-sub-maxprob-thr25-1mm.nii.gz')
SCHAEFER_400_PATH = os.path.join(_NILEARN, 'schaefer_2018/'
                                 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz')
SCHAEFER_1000_PATH = os.path.join(_NILEARN, 'schaefer_2018/'
                                  'Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_1mm.nii.gz')
AAL3_PATH = os.path.join(_NILEARN, 'aal_SPM12/AAL3/AAL3v1_1mm.nii.gz')
PAULI_2017_PATH = os.path.join(_NILEARN, 'pauli_2017/pauli_2017_det.nii.gz')
YEO_7_PATH = os.path.join(_NILEARN, 'yeo_2011/Yeo_JNeurophysiol11_MNI152/'
                          'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz')

# Warped-on-Halle paths for the three extra parcellations
AAL3_IN_HALLE = os.path.join(REG_DIR, 'aal3_in_halle.nii.gz')
PAULI_2017_IN_HALLE = os.path.join(REG_DIR, 'pauli_2017_in_halle.nii.gz')
YEO_7_IN_HALLE = os.path.join(REG_DIR, 'yeo_7nets_in_halle.nii.gz')

# Warped-on-Halle atlas paths (produced by warp_atlases.py;
# {key}_in_halle.nii.gz follows the canonical keys in
# tuba.atlases.mni152.PARCELLATIONS).
HO_IN_HALLE = os.path.join(REG_DIR, 'harvard_oxford_117_in_halle.nii.gz')
SCH_400_IN_HALLE = os.path.join(REG_DIR, 'schaefer_400_7nets_in_halle.nii.gz')
SCH_1000_IN_HALLE = os.path.join(REG_DIR, 'schaefer_1000_7nets_in_halle.nii.gz')
MNI_BRAIN_IN_HALLE = os.path.join(REG_DIR, 'mni_brain_in_halle.nii.gz')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

SAVE_DPI = 600


def _load_native_halle():
    """Load native Halle NRRD (~5.5 GB int16). Returns (array, voxel_mm)."""
    import nrrd
    print(f'Loading native NRRD ({NRRD_PATH})...')
    t0 = time.time()
    data, header = nrrd.read(NRRD_PATH)
    voxel_mm = float(header['space directions'][0, 0])
    print(f'  shape={data.shape}, voxel={voxel_mm} mm, '
          f'range=[{data.min()},{data.max()}], t={time.time()-t0:.1f}s')
    return data, voxel_mm


def _native_extents(arr_shape, voxel_mm):
    """World-mm extents for sagittal/coronal/axial panels in TUBA Halle RAS.

    Halle NRRD storage is LAS:
      axis 0 (+i = patient-L) -> TUBA RAS x = -i*v   (ranges 0 to -nL*v)
      axis 1 (+j = patient-A) -> TUBA RAS y = +j*v   (ranges 0 to +nA*v)
      axis 2 (+k = patient-S) -> TUBA RAS z = +k*v   (ranges 0 to +nS*v)

    matplotlib imshow extent = (xmin, xmax, ymin, ymax) for origin='lower'.
    """
    nL, nA, nS = arr_shape
    x_min, x_max = -(nL - 1) * voxel_mm, 0.0
    y_min, y_max = 0.0, (nA - 1) * voxel_mm
    z_min, z_max = 0.0, (nS - 1) * voxel_mm
    sag = [y_min, y_max, z_min, z_max]   # (A-P, S-I)
    cor = [x_max, x_min, z_min, z_max]   # (R-L: x_max=0 left, x_min=-nL*v right;
                                          #  note imshow with extent flips for
                                          #  decreasing x_max>x_min so the
                                          #  panel reads patient-right on left)
    ax  = [x_max, x_min, y_min, y_max]   # (R-L, A-P)
    return sag, cor, ax


def _bone_centroid_storage(arr, threshold=700):
    idx = np.argwhere(arr > threshold)
    centroid = idx.mean(axis=0)
    return tuple(int(round(v)) for v in centroid)


def _percentile_clip(arr, low=0.5, high=99.5):
    lo, hi = np.percentile(arr, [low, high])
    return np.clip(arr, lo, hi), lo, hi


def _to_neurological(ax):
    """Flip an Axes' x-axis so patient-L appears on screen-left
    (neurological convention). Called on Halle-frame coronal/axial
    panels whose natural rendering puts patient-R on screen-left
    (radiological), due to the negative aff[0,0] in the Halle affine.
    No-op on sagittal panels (their x-axis is A-P, not R-L)."""
    ax.invert_xaxis()


def fig1_halle_native(halle, voxel_mm):
    """Halle dry-skull orthoslices at NATIVE 0.125 mm."""
    print('fig1_halle_native: computing bone centroid...')
    bi, bj, bk = _bone_centroid_storage(halle, threshold=700)
    print(f'  bone centroid (storage i,j,k) = ({bi},{bj},{bk})')
    print(f'  world-mm (TUBA Halle RAS) = '
          f'({-bi*voxel_mm:.2f}, {+bj*voxel_mm:.2f}, {+bk*voxel_mm:.2f})')

    print('  extracting orthoslices...')
    sag = halle[bi, :, :].astype(np.float32)
    cor = halle[:, bj, :].astype(np.float32)
    axi = halle[:, :, bk].astype(np.float32)

    # Percentile clip on a subsample for speed; apply same window to all.
    sub = halle[::4, ::4, ::4].ravel()
    lo, hi = np.percentile(sub, [0.5, 99.5])
    print(f'  intensity window: [{lo:.0f}, {hi:.0f}] HU')

    e_sag, e_cor, e_ax = _native_extents(halle.shape, voxel_mm)

    print('  rendering...')
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    axes[0].imshow(np.clip(sag.T, lo, hi), cmap='bone', origin='lower',
                   extent=e_sag, vmin=lo, vmax=hi,
                   interpolation='nearest')
    axes[0].set(xlabel='A-P (mm)', ylabel='S-I (mm)',
                title=f'sagittal x = {-bi*voxel_mm:.2f} mm',
                aspect='equal')
    axes[1].imshow(np.clip(cor.T, lo, hi), cmap='bone', origin='lower',
                   extent=e_cor, vmin=lo, vmax=hi,
                   interpolation='nearest')
    axes[1].set(xlabel='L-R (mm)', ylabel='S-I (mm)',
                title=f'coronal y = {+bj*voxel_mm:.2f} mm',
                aspect='equal')
    _to_neurological(axes[1])
    axes[2].imshow(np.clip(axi.T, lo, hi), cmap='bone', origin='lower',
                   extent=e_ax, vmin=lo, vmax=hi,
                   interpolation='nearest')
    axes[2].set(xlabel='L-R (mm)', ylabel='A-P (mm)',
                title=f'axial z = {+bk*voxel_mm:.2f} mm',
                aspect='equal')
    _to_neurological(axes[2])
    fig.suptitle('Halle dry-skull microCT at native 0.125 mm. '
                 'Orthoslices through the cortical-bone centroid.',
                 fontsize=12)
    out = os.path.join(OUT_DIR, 'fig1_halle_native.png')
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)')
    return (bi, bj, bk)


def _harvard_oxford_combined():
    """Load Harvard-Oxford cortical-lateralized + subcortical and combine
    into a single annotation volume. Returns (annot_array, affine, n_labels).

    The two atlases overlap in some subcortical voxels; we prefer the
    cortical-lateralized ID where present, otherwise use the subcortical
    ID (offset by the cortical max so the merged IDs are unique).
    """
    cortl_img = nib.load(HO_CORTL_PATH)
    sub_img = nib.load(HO_SUB_PATH)
    cortl = cortl_img.get_fdata().astype(np.int32)
    sub = sub_img.get_fdata().astype(np.int32)
    assert cortl.shape == sub.shape
    assert np.allclose(cortl_img.affine, sub_img.affine)
    max_cortl = int(cortl.max())
    sub_off = np.where(sub > 0, sub + max_cortl, 0)
    combined = np.where(cortl > 0, cortl, sub_off)
    n_labels = len(np.unique(combined)) - 1   # exclude background
    print(f'  Harvard-Oxford merged: shape={combined.shape}, '
          f'{n_labels} distinct labels (cortl 1..{max_cortl} + sub offset)')
    return combined, cortl_img.affine, n_labels


def _annot_to_rgba(annot_slice, palette='gist_ncar', alpha_fg=0.95,
                   shuffle_prime=104729):
    """Map a 2D integer-label slice to an RGBA image via a deterministic
    ID -> palette hue shuffle (so neighbouring IDs are colour-separated).
    Background (id=0) is transparent.
    """
    cmap = plt.get_cmap(palette)
    unique_ids = np.unique(annot_slice)
    # Hash each ID through a prime modulo a large N for distinct hues
    N = 4096
    rgba = np.zeros((*annot_slice.shape, 4), dtype=np.float32)
    for label_id in unique_ids:
        if label_id == 0:
            continue
        shuf = (int(label_id) * shuffle_prime) % N
        colour = cmap(shuf / N)
        mask = annot_slice == label_id
        rgba[mask, 0] = colour[0]
        rgba[mask, 1] = colour[1]
        rgba[mask, 2] = colour[2]
        rgba[mask, 3] = alpha_fg
    return rgba


def _extents_for_affine(arr, aff):
    """sagittal, coronal, axial world-mm extents from a 3D affine.

    Matplotlib imshow extent convention: ``(left, right, bottom, top)``
    where ``left`` is the world coordinate of the image's left edge
    (== data column 0) and ``right`` is the right edge
    (== data column n_cols-1). For affines with negative axis-0 sign
    (e.g. Halle's ``diag(-vox, +vox, +vox)``), left > right -- this
    correctly gives the radiological convention (patient-R on screen-
    left, i.e. world x=0 on the left) because data column 0 is
    physically at the +x end of the volume.

    A prior version of this function returned the two endpoints in the
    wrong order, which caused the warped atlas overlays in Fig.~4 to
    appear mirror-flipped relative to the native-resolution bone (which
    uses :func:`_native_extents` and was always correct).
    """
    ni, nj, nk = arr.shape
    sag = [aff[1, 3], aff[1, 3] + (nj - 1) * aff[1, 1],
           aff[2, 3], aff[2, 3] + (nk - 1) * aff[2, 2]]
    cor = [aff[0, 3], aff[0, 3] + (ni - 1) * aff[0, 0],
           aff[2, 3], aff[2, 3] + (nk - 1) * aff[2, 2]]
    ax_e = [aff[0, 3], aff[0, 3] + (ni - 1) * aff[0, 0],
            aff[1, 3], aff[1, 3] + (nj - 1) * aff[1, 1]]
    return sag, cor, ax_e


def _world_to_vox(aff, world):
    return np.linalg.solve(aff, np.array([*world, 1]))[:3]


def _atlas_panel(ax, label_vol, aff, slice_idx, axis,
                 extent, title, xlabel, ylabel,
                 background_arr=None, background_extent=None,
                 background_window=None, alpha_fg=0.85):
    """Render one orthoslice of an integer-label volume.

    If background_arr is given, render it (greyscale) underneath and
    blend the labels with alpha. Otherwise the panel is labels on
    black.
    """
    # Extract label slice
    if axis == 0:
        ann = label_vol[slice_idx, :, :].T
    elif axis == 1:
        ann = label_vol[:, slice_idx, :].T
    else:
        ann = label_vol[:, :, slice_idx].T
    if background_arr is not None and background_extent is not None:
        bg_lo, bg_hi = background_window
        if axis == 0:
            bg = background_arr[slice_idx, :, :].T   # caller slicing
        else:
            bg = background_arr
        ax.imshow(np.clip(bg, bg_lo, bg_hi), cmap='bone',
                  origin='lower', extent=background_extent,
                  vmin=bg_lo, vmax=bg_hi, interpolation='nearest')
    else:
        ax.set_facecolor('black')
    ax.imshow(_annot_to_rgba(ann, alpha_fg=alpha_fg), origin='lower',
              extent=extent, interpolation='nearest')
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, aspect='equal')


def fig2_mni_template_and_atlases():
    """3-row x 4-column layout: MNI T1 (col 1) + Harvard-Oxford (col 2)
    + Schaefer-400 (col 3) + Schaefer-1000 (col 4), sagittal/coronal/
    axial at AC.
    """
    print('fig2: loading MNI T1 + brain mask...')
    t1_img = nib.load(os.path.join(TEMPLATE_DIR,
                                    'mni_icbm152_t1_tal_nlin_sym_09a.nii'))
    mask_img = nib.load(os.path.join(TEMPLATE_DIR,
                                      'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii'))
    t1 = t1_img.get_fdata().astype(np.float32)
    mask = mask_img.get_fdata().astype(np.float32)
    t1_aff = t1_img.affine

    print('  loading Harvard-Oxford merged + Schaefer 400 + Schaefer 1000...')
    ho_vol, ho_aff, ho_nlabels = _harvard_oxford_combined()
    s400_img = nib.load(SCHAEFER_400_PATH)
    s400 = s400_img.get_fdata().astype(np.int32)
    s400_aff = s400_img.affine
    s1000_img = nib.load(SCHAEFER_1000_PATH)
    s1000 = s1000_img.get_fdata().astype(np.int32)
    s1000_aff = s1000_img.affine
    print(f'    HO labels: {ho_nlabels}, S400 labels: {len(np.unique(s400))-1}, '
          f'S1000 labels: {len(np.unique(s1000))-1}')

    # Slice indices at AC=(0,0,0) in each volume's own frame
    t1_ac = tuple(int(round(v)) for v in _world_to_vox(t1_aff, (0, 0, 0)))
    ho_ac = tuple(int(round(v)) for v in _world_to_vox(ho_aff, (0, 0, 0)))
    s400_ac = tuple(int(round(v)) for v in _world_to_vox(s400_aff, (0, 0, 0)))
    s1000_ac = tuple(int(round(v)) for v in _world_to_vox(s1000_aff, (0, 0, 0)))

    t1_clip, lo, hi = _percentile_clip(t1, 0.5, 99.5)
    t1_sag, t1_cor, t1_ax = _extents_for_affine(t1, t1_aff)
    ho_sag, ho_cor, ho_ax = _extents_for_affine(ho_vol, ho_aff)
    s400_sag, s400_cor, s400_ax = _extents_for_affine(s400, s400_aff)
    s1000_sag, s1000_cor, s1000_ax = _extents_for_affine(s1000, s1000_aff)

    print('  rendering 3x4 layout...')
    fig, axes = plt.subplots(3, 4, figsize=(20, 13),
                             constrained_layout=True)

    rows = [
        ('sagittal x = 0.0 mm', 0, t1_ac[0], ho_ac[0], s400_ac[0], s1000_ac[0],
         t1_sag, ho_sag, s400_sag, s1000_sag, 'A-P (mm)', 'S-I (mm)'),
        ('coronal y = 0.0 mm', 1, t1_ac[1], ho_ac[1], s400_ac[1], s1000_ac[1],
         t1_cor, ho_cor, s400_cor, s1000_cor, 'L-R (mm)', 'S-I (mm)'),
        ('axial z = 0.0 mm', 2, t1_ac[2], ho_ac[2], s400_ac[2], s1000_ac[2],
         t1_ax, ho_ax, s400_ax, s1000_ax, 'L-R (mm)', 'A-P (mm)'),
    ]

    for r, (slice_title, axis, ti, hi_, s4i, s10i,
            t1ext, hoext, s4ext, s10ext, xlab, ylab) in enumerate(rows):
        # Col 0: T1 template
        if axis == 0:
            sl = t1_clip[ti, :, :].T
        elif axis == 1:
            sl = t1_clip[:, ti, :].T
        else:
            sl = t1_clip[:, :, ti].T
        axes[r, 0].imshow(sl, cmap='gray', origin='lower',
                          extent=t1ext, vmin=lo, vmax=hi)
        if axis == 0:
            msl = mask[ti, :, :].T
        elif axis == 1:
            msl = mask[:, ti, :].T
        else:
            msl = mask[:, :, ti].T
        axes[r, 0].contour(msl, levels=[0.5], colors=['cyan'],
                           linewidths=0.7, extent=t1ext)
        axes[r, 0].set(xlabel=xlab, ylabel=ylab,
                       title=f'{slice_title}\nMNI T1 template (1 mm)',
                       aspect='equal')

        # Col 1: Harvard-Oxford
        _atlas_panel(axes[r, 1], ho_vol, ho_aff, hi_, axis, hoext,
                     f'{slice_title}\nHarvard-Oxford ({ho_nlabels} labels)',
                     xlab, ylab)
        # Col 2: Schaefer 400
        _atlas_panel(axes[r, 2], s400, s400_aff, s4i, axis, s4ext,
                     f'{slice_title}\nSchaefer 400 / 7 networks',
                     xlab, ylab)
        # Col 3: Schaefer 1000
        _atlas_panel(axes[r, 3], s1000, s1000_aff, s10i, axis, s10ext,
                     f'{slice_title}\nSchaefer 1000 / 7 networks',
                     xlab, ylab)

    fig.suptitle('MNI152 ICBM 2009a T1 template and three cortical / '
                 'subcortical parcellations in MNI space. Orthoslices '
                 'at the anterior commissure (world 0,0,0 mm). The '
                 'Harvard-Oxford label IDs and the Schaefer 400 / 1000 '
                 'IDs are mapped through a deterministic ID-hash to a '
                 'gist_ncar palette so neighbouring IDs are colour-'
                 'separated.', fontsize=12)
    out = os.path.join(OUT_DIR, 'fig2_mni_template.png')
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)')


# Back-compat alias
fig2_mni_template = fig2_mni_template_and_atlases
fig2_mni_template_and_atlas = fig2_mni_template_and_atlases


def fig4_warped_atlases_on_halle(halle, voxel_mm):
    """3-row x 4-column layout: native 0.125 mm Halle bone (col 1) +
    warped Harvard-Oxford (col 2) + warped Schaefer 400 (col 3) +
    warped Schaefer 1000 (col 4), at the warped-brain centroid.

    Each parcellation column overlays the warped labels on the native
    Halle bone in greyscale, with the warped MNI brain mask drawn as a
    cyan contour. This is the mouse-manuscript Fig. 6 analogue.
    """
    print('fig4_warped_atlases_on_halle: loading warped atlases + brain mask...')
    ho_img = nib.load(HO_IN_HALLE)
    s400_img = nib.load(SCH_400_IN_HALLE)
    s1000_img = nib.load(SCH_1000_IN_HALLE)
    mask_img = nib.load(MNI_BRAIN_IN_HALLE)
    ho = ho_img.get_fdata().astype(np.int32)
    s400 = s400_img.get_fdata().astype(np.int32)
    s1000 = s1000_img.get_fdata().astype(np.int32)
    mask = mask_img.get_fdata().astype(np.float32)
    halle_1mm_aff = ho_img.affine
    print(f'  warped atlases on Halle 1mm grid: shape={ho.shape}')

    # Slice at warped brain centroid (1mm voxel coords)
    brain_idx = np.argwhere(mask > 0.5)
    mi, mj, mk = tuple(int(round(v)) for v in brain_idx.mean(axis=0))
    bx_world = halle_1mm_aff[0, 3] + mi * halle_1mm_aff[0, 0]
    by_world = halle_1mm_aff[1, 3] + mj * halle_1mm_aff[1, 1]
    bz_world = halle_1mm_aff[2, 3] + mk * halle_1mm_aff[2, 2]
    print(f'  warped brain centroid 1mm voxel ({mi},{mj},{mk}) -> world '
          f'({bx_world:.2f}, {by_world:.2f}, {bz_world:.2f})')

    # Native Halle slice indices for the same world coords
    ni_native = max(0, min(halle.shape[0] - 1,
                           int(round(-bx_world / voxel_mm))))
    nj_native = max(0, min(halle.shape[1] - 1,
                           int(round(+by_world / voxel_mm))))
    nk_native = max(0, min(halle.shape[2] - 1,
                           int(round(+bz_world / voxel_mm))))
    print(f'  native (0.125 mm) slicing indices: '
          f'({ni_native},{nj_native},{nk_native})')

    halle_sag = halle[ni_native, :, :].astype(np.float32)
    halle_cor = halle[:, nj_native, :].astype(np.float32)
    halle_axi = halle[:, :, nk_native].astype(np.float32)
    sub = halle[::4, ::4, ::4].ravel()
    h_lo, h_hi = np.percentile(sub, [0.5, 99.5])

    # Halle native extents (TUBA Halle RAS frame)
    e_sag, e_cor, e_ax = _native_extents(halle.shape, voxel_mm)
    # Atlas extents (1mm Halle grid)
    a_sag, a_cor, a_ax = _extents_for_affine(ho, halle_1mm_aff)

    print('  rendering 3x4 layout...')
    fig, axes = plt.subplots(3, 4, figsize=(20, 13),
                             constrained_layout=True)

    halle_slices = [halle_sag, halle_cor, halle_axi]
    halle_extents = [e_sag, e_cor, e_ax]
    atlas_extents = [a_sag, a_cor, a_ax]
    atlas_slicers = [
        lambda v, i: v[i, :, :].T,
        lambda v, j: v[:, j, :].T,
        lambda v, k: v[:, :, k].T,
    ]
    atlas_idx_native = [ni_native, nj_native, nk_native]
    atlas_idx_1mm = [mi, mj, mk]
    slice_titles = [
        f'sagittal x = {-ni_native*voxel_mm:.1f} mm',
        f'coronal y = {+nj_native*voxel_mm:.1f} mm',
        f'axial z = {+nk_native*voxel_mm:.1f} mm',
    ]
    xls = ['A-P (mm)', 'L-R (mm)', 'L-R (mm)']
    yls = ['S-I (mm)', 'S-I (mm)', 'A-P (mm)']

    atlas_data = [
        ('Harvard-Oxford (117)', ho),
        ('Schaefer 400 / 7 nets', s400),
        ('Schaefer 1000 / 7 nets', s1000),
    ]

    for r in range(3):
        hsl = halle_slices[r]
        hex_ = halle_extents[r]
        aex = atlas_extents[r]
        mask_slice = atlas_slicers[r](mask, atlas_idx_1mm[r])

        # Col 0: bare native bone
        axes[r, 0].imshow(np.clip(hsl.T, h_lo, h_hi), cmap='bone',
                          origin='lower', extent=hex_, vmin=h_lo, vmax=h_hi,
                          interpolation='nearest')
        axes[r, 0].contour(mask_slice, levels=[0.5], colors=['cyan'],
                           linewidths=1.0, extent=aex)
        axes[r, 0].set(xlabel=xls[r], ylabel=yls[r],
                       title=f'{slice_titles[r]}\nHalle CT (native 0.125 mm)',
                       aspect='equal')

        # Cols 1-3: bone + warped atlas overlay + brain mask contour
        for c, (name, vol) in enumerate(atlas_data):
            ax = axes[r, c + 1]
            ax.imshow(np.clip(hsl.T, h_lo, h_hi), cmap='bone',
                      origin='lower', extent=hex_, vmin=h_lo, vmax=h_hi,
                      interpolation='nearest')
            ann_slice = atlas_slicers[r](vol, atlas_idx_1mm[r])
            ax.imshow(_annot_to_rgba(ann_slice, alpha_fg=0.7),
                      origin='lower', extent=aex, interpolation='nearest')
            ax.contour(mask_slice, levels=[0.5], colors=['cyan'],
                       linewidths=1.0, extent=aex)
            ax.set(xlabel=xls[r], ylabel=yls[r],
                   title=f'{slice_titles[r]}\n{name} warped to Halle',
                   aspect='equal')

        # Coronal (r=1) and axial (r=2) have x-axis = R-L; flip to
        # neurological (patient-L on screen-left) to match Fig. 2 / 5.
        if r > 0:
            for c in range(axes.shape[1]):
                _to_neurological(axes[r, c])

    # No suptitle -- explanation moved to the caption so the panels
    # can occupy the full text width.
    out = os.path.join(OUT_DIR, 'fig4_warped_atlases_on_halle.png')
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)')


def fig3_skull_atlas(halle, voxel_mm):
    """Native 0.125 mm Halle bone (greyscale) + warped MNI T1 overlay
    + brain-mask contour (overlay layers at 1 mm; matplotlib's extent
    handles the resolution mismatch)."""
    print('fig3_skull_atlas: loading warped MNI 1 mm volumes...')
    mni_t1_img = nib.load(os.path.join(REG_DIR, 'mni_t1_in_halle.nii.gz'))
    mni_mask_img = nib.load(os.path.join(REG_DIR, 'mni_brain_in_halle.nii.gz'))
    mni_t1 = mni_t1_img.get_fdata().astype(np.float32)
    mni_mask = mni_mask_img.get_fdata().astype(np.float32)
    mni_aff = mni_t1_img.affine
    print(f'  MNI-in-Halle shape={mni_t1.shape}, '
          f'voxel={abs(mni_aff[0,0]):.3f} mm')

    # Slice at the warped-brain centroid (in MNI-in-Halle voxel coords);
    # use the corresponding world coords to slice the native NRRD.
    brain_idx = np.argwhere(mni_mask > 0.5)
    mi, mj, mk = tuple(int(round(v)) for v in brain_idx.mean(axis=0))
    bx_world = mni_aff[0, 3] + mi * mni_aff[0, 0]
    by_world = mni_aff[1, 3] + mj * mni_aff[1, 1]
    bz_world = mni_aff[2, 3] + mk * mni_aff[2, 2]
    print(f'  warped brain centroid: 1mm voxel ({mi},{mj},{mk}) -> world '
          f'({bx_world:.2f}, {by_world:.2f}, {bz_world:.2f})')

    # Convert world coord to native Halle storage index:
    # TUBA RAS x = -i*v -> i = -x/v;  y = +j*v -> j = +y/v;  z = +k*v -> k = +z/v
    ni_native = int(round(-bx_world / voxel_mm))
    nj_native = int(round(+by_world / voxel_mm))
    nk_native = int(round(+bz_world / voxel_mm))
    nL, nA, nS = halle.shape
    ni_native = max(0, min(nL - 1, ni_native))
    nj_native = max(0, min(nA - 1, nj_native))
    nk_native = max(0, min(nS - 1, nk_native))
    print(f'  native (0.125 mm) slicing indices: '
          f'({ni_native},{nj_native},{nk_native})')

    halle_sag = halle[ni_native, :, :].astype(np.float32)
    halle_cor = halle[:, nj_native, :].astype(np.float32)
    halle_axi = halle[:, :, nk_native].astype(np.float32)

    sub = halle[::4, ::4, ::4].ravel()
    h_lo, h_hi = np.percentile(sub, [0.5, 99.5])
    print(f'  Halle intensity window: [{h_lo:.0f}, {h_hi:.0f}] HU')

    # MNI overlay: mask the T1 to its own brain mask so the overlay
    # only shows the brain region (not the warped scalp).
    mni_overlay = np.where(mni_mask > 0.5, mni_t1, np.nan)
    mni_lo, mni_hi = np.nanpercentile(mni_overlay, [5, 99])

    # World-mm extents for the native Halle (used by both bone and
    # MNI overlay, since both are in the same TUBA Halle RAS frame).
    e_sag, e_cor, e_ax = _native_extents(halle.shape, voxel_mm)

    print('  rendering...')
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)

    # Sagittal
    axes[0].imshow(np.clip(halle_sag.T, h_lo, h_hi), cmap='bone',
                   origin='lower', extent=e_sag, vmin=h_lo, vmax=h_hi,
                   interpolation='nearest')
    axes[0].imshow(mni_overlay[mi, :, :].T, cmap='inferno',
                   origin='lower', extent=e_sag,
                   vmin=mni_lo, vmax=mni_hi, alpha=0.55,
                   interpolation='bilinear')
    axes[0].contour(mni_mask[mi, :, :].T, levels=[0.5], colors=['cyan'],
                    linewidths=1.2, extent=e_sag)
    axes[0].set(xlabel='A-P (mm)', ylabel='S-I (mm)',
                title=f'sagittal x = {-ni_native*voxel_mm:.2f} mm',
                aspect='equal')

    # Coronal
    axes[1].imshow(np.clip(halle_cor.T, h_lo, h_hi), cmap='bone',
                   origin='lower', extent=e_cor, vmin=h_lo, vmax=h_hi,
                   interpolation='nearest')
    axes[1].imshow(mni_overlay[:, mj, :].T, cmap='inferno',
                   origin='lower', extent=e_cor,
                   vmin=mni_lo, vmax=mni_hi, alpha=0.55,
                   interpolation='bilinear')
    axes[1].contour(mni_mask[:, mj, :].T, levels=[0.5], colors=['cyan'],
                    linewidths=1.2, extent=e_cor)
    axes[1].set(xlabel='L-R (mm)', ylabel='S-I (mm)',
                title=f'coronal y = {+nj_native*voxel_mm:.2f} mm',
                aspect='equal')
    _to_neurological(axes[1])

    # Axial
    axes[2].imshow(np.clip(halle_axi.T, h_lo, h_hi), cmap='bone',
                   origin='lower', extent=e_ax, vmin=h_lo, vmax=h_hi,
                   interpolation='nearest')
    axes[2].imshow(mni_overlay[:, :, mk].T, cmap='inferno',
                   origin='lower', extent=e_ax,
                   vmin=mni_lo, vmax=mni_hi, alpha=0.55,
                   interpolation='bilinear')
    axes[2].contour(mni_mask[:, :, mk].T, levels=[0.5], colors=['cyan'],
                    linewidths=1.2, extent=e_ax)
    axes[2].set(xlabel='L-R (mm)', ylabel='A-P (mm)',
                title=f'axial z = {+nk_native*voxel_mm:.2f} mm',
                aspect='equal')
    _to_neurological(axes[2])

    fig.suptitle('Halle CT (native 0.125 mm, bone, greyscale) with '
                 'warped MNI T1 (1 mm, brain, warm overlay) and '
                 'MNI brain mask (cyan contour) — ANTs SyN registration QC.',
                 fontsize=12)
    out = os.path.join(OUT_DIR, 'fig3_skull_atlas.png')
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)')


def fig5_extra_parcellations_mni():
    """3-row x 3-column layout: AAL3 (col 1) + Pauli 2017 (col 2) +
    Yeo 7-networks (col 3), sagittal/coronal/axial at AC.

    Companion to Fig. 2 covering the three additional parcellations
    wired into ``tuba.atlases.mni152.PARCELLATIONS``: AAL3 for
    classical whole-brain ROI work, Pauli 2017 for high-resolution
    deep-brain subcortex, Yeo 2011 for functional-network targeting.
    """
    print('fig5_extra_parcellations_mni: loading AAL3, Pauli, Yeo...')
    # nib.as_closest_canonical reorients to RAS so the slicing and
    # extent helpers (which assume i->x, j->y, k->z) work uniformly.
    # Yeo's native storage is LIA, AAL/Pauli are RAS-aligned already
    # but we apply the canonicalisation unconditionally for safety.
    aal_img = nib.as_closest_canonical(nib.load(AAL3_PATH))
    pauli_img = nib.as_closest_canonical(nib.load(PAULI_2017_PATH))
    yeo_img = nib.as_closest_canonical(nib.load(YEO_7_PATH))
    aal = aal_img.get_fdata().astype(np.int32)
    pauli = pauli_img.get_fdata().astype(np.int32)
    yeo = np.squeeze(yeo_img.get_fdata()).astype(np.int32)
    aal_aff = aal_img.affine
    pauli_aff = pauli_img.affine
    yeo_aff = yeo_img.affine
    print(f'  AAL3 shape={aal.shape}, voxel={abs(aal_aff[0,0]):.2f}, '
          f'n_labels={len(np.unique(aal))-1}')
    print(f'  Pauli shape={pauli.shape}, voxel={abs(pauli_aff[0,0]):.2f}, '
          f'n_labels={len(np.unique(pauli))-1}')
    print(f'  Yeo shape={yeo.shape}, voxel={abs(yeo_aff[0,0]):.2f}, '
          f'n_labels={len(np.unique(yeo))-1}')

    # Slice indices at AC=(0,0,0) in each volume's own grid
    aal_ac = tuple(int(round(v)) for v in _world_to_vox(aal_aff, (0, 0, 0)))
    pauli_ac = tuple(int(round(v)) for v in _world_to_vox(pauli_aff, (0, 0, 0)))
    yeo_ac = tuple(int(round(v)) for v in _world_to_vox(yeo_aff, (0, 0, 0)))

    aal_sag, aal_cor, aal_ax = _extents_for_affine(aal, aal_aff)
    pauli_sag, pauli_cor, pauli_ax = _extents_for_affine(pauli, pauli_aff)
    yeo_sag, yeo_cor, yeo_ax = _extents_for_affine(yeo, yeo_aff)

    print('  rendering 3x3 layout...')
    fig, axes = plt.subplots(3, 3, figsize=(15, 13),
                             constrained_layout=True)

    cols = [
        ('AAL3 (166 labels)', aal, aal_aff, aal_ac,
         aal_sag, aal_cor, aal_ax),
        ('Pauli 2017 (16 subcortex)', pauli, pauli_aff, pauli_ac,
         pauli_sag, pauli_cor, pauli_ax),
        ('Yeo 2011 (7 networks)', yeo, yeo_aff, yeo_ac,
         yeo_sag, yeo_cor, yeo_ax),
    ]
    row_titles = ['sagittal x = 0.0 mm', 'coronal y = 0.0 mm',
                  'axial z = 0.0 mm']
    xls = ['A-P (mm)', 'L-R (mm)', 'L-R (mm)']
    yls = ['S-I (mm)', 'S-I (mm)', 'A-P (mm)']

    for r in range(3):
        for c, (name, vol, aff, ac_vox, e_sag, e_cor, e_ax) in enumerate(cols):
            ax = axes[r, c]
            if r == 0:
                axis = 0
                idx = ac_vox[0]
                ext = e_sag
            elif r == 1:
                axis = 1
                idx = ac_vox[1]
                ext = e_cor
            else:
                axis = 2
                idx = ac_vox[2]
                ext = e_ax
            _atlas_panel(ax, vol, aff, idx, axis, ext,
                         f'{row_titles[r]}\n{name}', xls[r], yls[r])

    out = os.path.join(OUT_DIR, 'fig5_extra_parcellations_mni.png')
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)')


def fig6_extra_parcellations_on_halle(halle, voxel_mm):
    """3-row x 4-column layout: native 0.125 mm Halle bone (col 1) +
    warped AAL3 (col 2) + warped Pauli 2017 (col 3) + warped Yeo
    (col 4), at the warped-brain centroid.

    Companion to Fig. 4 covering the three additional parcellations.
    """
    print('fig6_extra_parcellations_on_halle: loading warped atlases...')
    aal_img = nib.load(AAL3_IN_HALLE)
    pauli_img = nib.load(PAULI_2017_IN_HALLE)
    yeo_img = nib.load(YEO_7_IN_HALLE)
    mask_img = nib.load(MNI_BRAIN_IN_HALLE)
    aal = aal_img.get_fdata().astype(np.int32)
    pauli = pauli_img.get_fdata().astype(np.int32)
    yeo = np.squeeze(yeo_img.get_fdata()).astype(np.int32)   # may be 4D
    mask = mask_img.get_fdata().astype(np.float32)
    halle_1mm_aff = aal_img.affine

    brain_idx = np.argwhere(mask > 0.5)
    mi, mj, mk = tuple(int(round(v)) for v in brain_idx.mean(axis=0))
    bx_world = halle_1mm_aff[0, 3] + mi * halle_1mm_aff[0, 0]
    by_world = halle_1mm_aff[1, 3] + mj * halle_1mm_aff[1, 1]
    bz_world = halle_1mm_aff[2, 3] + mk * halle_1mm_aff[2, 2]

    ni_native = max(0, min(halle.shape[0] - 1,
                           int(round(-bx_world / voxel_mm))))
    nj_native = max(0, min(halle.shape[1] - 1,
                           int(round(+by_world / voxel_mm))))
    nk_native = max(0, min(halle.shape[2] - 1,
                           int(round(+bz_world / voxel_mm))))

    halle_sag = halle[ni_native, :, :].astype(np.float32)
    halle_cor = halle[:, nj_native, :].astype(np.float32)
    halle_axi = halle[:, :, nk_native].astype(np.float32)
    sub = halle[::4, ::4, ::4].ravel()
    h_lo, h_hi = np.percentile(sub, [0.5, 99.5])

    e_sag, e_cor, e_ax = _native_extents(halle.shape, voxel_mm)
    a_sag, a_cor, a_ax = _extents_for_affine(aal, halle_1mm_aff)

    halle_slices = [halle_sag, halle_cor, halle_axi]
    halle_extents = [e_sag, e_cor, e_ax]
    atlas_extents = [a_sag, a_cor, a_ax]
    atlas_slicers = [
        lambda v, i: v[i, :, :].T,
        lambda v, j: v[:, j, :].T,
        lambda v, k: v[:, :, k].T,
    ]
    atlas_idx_1mm = [mi, mj, mk]
    slice_titles = [
        f'sagittal x = {-ni_native*voxel_mm:.1f} mm',
        f'coronal y = {+nj_native*voxel_mm:.1f} mm',
        f'axial z = {+nk_native*voxel_mm:.1f} mm',
    ]
    xls = ['A-P (mm)', 'L-R (mm)', 'L-R (mm)']
    yls = ['S-I (mm)', 'S-I (mm)', 'A-P (mm)']

    atlas_data = [
        ('AAL3 (166)', aal),
        ('Pauli 2017 (16 subcortex)', pauli),
        ('Yeo 2011 (7 networks)', yeo),
    ]

    print('  rendering 3x4 layout...')
    fig, axes = plt.subplots(3, 4, figsize=(20, 13),
                             constrained_layout=True)

    for r in range(3):
        hsl = halle_slices[r]
        hex_ = halle_extents[r]
        aex = atlas_extents[r]
        mask_slice = atlas_slicers[r](mask, atlas_idx_1mm[r])

        axes[r, 0].imshow(np.clip(hsl.T, h_lo, h_hi), cmap='bone',
                          origin='lower', extent=hex_, vmin=h_lo, vmax=h_hi,
                          interpolation='nearest')
        axes[r, 0].contour(mask_slice, levels=[0.5], colors=['cyan'],
                           linewidths=1.0, extent=aex)
        axes[r, 0].set(xlabel=xls[r], ylabel=yls[r],
                       title=f'{slice_titles[r]}\nHalle CT (native 0.125 mm)',
                       aspect='equal')

        for c, (name, vol) in enumerate(atlas_data):
            ax = axes[r, c + 1]
            ax.imshow(np.clip(hsl.T, h_lo, h_hi), cmap='bone',
                      origin='lower', extent=hex_, vmin=h_lo, vmax=h_hi,
                      interpolation='nearest')
            ann_slice = atlas_slicers[r](vol, atlas_idx_1mm[r])
            ax.imshow(_annot_to_rgba(ann_slice, alpha_fg=0.7),
                      origin='lower', extent=aex, interpolation='nearest')
            ax.contour(mask_slice, levels=[0.5], colors=['cyan'],
                       linewidths=1.0, extent=aex)
            ax.set(xlabel=xls[r], ylabel=yls[r],
                   title=f'{slice_titles[r]}\n{name} warped to Halle',
                   aspect='equal')

        # Coronal (r=1) and axial (r=2): flip x-axis to neurological
        # to match Fig. 2 / Fig. 5 (patient-L on screen-left).
        if r > 0:
            for c in range(axes.shape[1]):
                _to_neurological(axes[r, c])

    out = os.path.join(OUT_DIR, 'fig6_extra_parcellations_on_halle.png')
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)')


def main():
    halle, voxel_mm = _load_native_halle()
    fig1_halle_native(halle, voxel_mm)
    fig2_mni_template_and_atlases()
    fig3_skull_atlas(halle, voxel_mm)
    fig4_warped_atlases_on_halle(halle, voxel_mm)
    fig5_extra_parcellations_mni()
    fig6_extra_parcellations_on_halle(halle, voxel_mm)
    print(f'\nAll figures written to {OUT_DIR}')


if __name__ == '__main__':
    main()
