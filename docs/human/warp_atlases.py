"""Warp all bundled MNI-space parcellations (from
:data:`tuba.atlases.mni152.PARCELLATIONS`) into Halle skull space using
the saved ANTs SyN forward transform.

Outputs land in ``REG_DIR`` as ``{key}_in_halle.nii.gz`` (one file per
parcellation key in the catalogue). Uses ``ants.apply_transforms`` with
nearestNeighbor interpolation (preserves integer label IDs); ANTs
resamples in physical world coordinates so source-grid differences
between FSL-MNI152 / ICBM-2009a / Pauli's 0.7 mm grid / Yeo's
FreeSurfer-conformed grid are handled internally.

Special-case: Harvard-Oxford ships as a cortical-lateralized + a
subcortical NIfTI which we merge into a single 117-label volume
before warping.
"""
import os
import sys
import numpy as np
import nibabel as nib
import ants

TUBA_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'src')
sys.path.insert(0, TUBA_SRC)
from tuba.atlases.mni152 import PARCELLATIONS, get_parcellation  # noqa: E402

# Paths obey the same TUBA_HUMAN_REG_DIR env var as
# tuba.species.human; falls back to the same ~/.cache/tuba/human
# location so a fresh install works without extra config.
REG_DIR = os.environ.get(
    'TUBA_HUMAN_REG_DIR',
    os.path.expanduser('~/.cache/tuba/human/registration'))
HALLE_REF = os.path.join(REG_DIR, 'halle_ct_1mm.nii.gz')
FWD_TXT = os.path.join(REG_DIR, 'ants_syn_forward.txt')
MNI_BRAIN_IN_HALLE = os.path.join(REG_DIR, 'mni_brain_in_halle.nii.gz')

# Harvard-Oxford merge sources (cortl + sub -> 117 labels). The
# catalogue entry in tuba.atlases.mni152 points at cortl; we override
# here to feed the merged volume to ANTs. NILEARN_DATA env var (used
# by nilearn) controls the cache dir; default is ~/nilearn_data.
_NILEARN = os.environ.get('NILEARN_DATA',
                          os.path.expanduser('~/nilearn_data'))
HO_CORTL = os.path.join(_NILEARN, 'fsl/data/atlases/HarvardOxford/'
                                  'HarvardOxford-cortl-maxprob-thr25-1mm.nii.gz')
HO_SUB = os.path.join(_NILEARN, 'fsl/data/atlases/HarvardOxford/'
                                'HarvardOxford-sub-maxprob-thr25-1mm.nii.gz')


def _read_fwd_transforms():
    with open(FWD_TXT) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _build_harvard_oxford_merged():
    """Combine cortl + sub into a single 117-label atlas. Cached."""
    out = os.path.join(REG_DIR, 'harvard_oxford_merged_mni.nii.gz')
    if os.path.exists(out):
        return out
    cortl_img = nib.load(HO_CORTL)
    sub_img = nib.load(HO_SUB)
    cortl = cortl_img.get_fdata().astype(np.int32)
    sub = sub_img.get_fdata().astype(np.int32)
    max_cortl = int(cortl.max())
    sub_off = np.where(sub > 0, sub + max_cortl, 0)
    combined = np.where(cortl > 0, cortl, sub_off).astype(np.int16)
    nib.save(nib.Nifti1Image(combined, cortl_img.affine), out)
    print(f'  merged Harvard-Oxford ({len(np.unique(combined))-1} labels): {out}')
    return out


def warp_atlas_to_halle(atlas_path, out_name, mask_to_brain=False, verbose=True):
    """Apply ants_syn_forward to bring atlas onto the Halle 1 mm grid.

    If ``mask_to_brain=True``, the warped output is intersected with
    the warped MNI brain mask before being saved (zeroes label voxels
    that fall outside the brain envelope). Used for atlases with
    intrinsic asymmetry whose lateral labels spill into the cortical
    bone shell after warping; see the Parcellation dataclass docstring.
    """
    out_path = os.path.join(REG_DIR, out_name)
    if os.path.exists(out_path):
        if verbose:
            print(f'  cached: {out_path}')
        return out_path
    fwd = _read_fwd_transforms()
    fixed = ants.image_read(HALLE_REF)
    moving = ants.image_read(atlas_path)
    if verbose:
        print(f'  warping {os.path.basename(atlas_path)} -> {out_name}')
        print(f'    moving shape={moving.shape} spacing={moving.spacing}')
    warped = ants.apply_transforms(
        fixed=fixed, moving=moving, transformlist=fwd,
        interpolator='nearestNeighbor')
    if mask_to_brain:
        if not os.path.exists(MNI_BRAIN_IN_HALLE):
            raise FileNotFoundError(
                f'mask_to_brain=True but {MNI_BRAIN_IN_HALLE} is missing. '
                f'Run register_ants_syn.py first to produce the warped '
                f'brain mask.')
        brain = nib.load(MNI_BRAIN_IN_HALLE).get_fdata() > 0.5
        warped_arr = warped.numpy().astype(np.int16)
        n_before = int((warped_arr > 0).sum())
        warped_arr = np.where(brain, warped_arr, 0)
        n_after = int((warped_arr > 0).sum())
        if verbose:
            print(f'    masked to brain envelope: '
                  f'{n_before:,} -> {n_after:,} label voxels '
                  f'(removed {n_before - n_after:,}, '
                  f'{(n_before - n_after)/n_before*100:.1f}% of pre-mask)')
        nib.save(nib.Nifti1Image(warped_arr,
                                  nib.load(HALLE_REF).affine),
                 out_path)
    else:
        ants.image_write(warped, out_path)
    if verbose:
        import nibabel as _nib
        final_arr = _nib.load(out_path).get_fdata().astype(int)
        n_labels = len(np.unique(final_arr)) - 1
        print(f'    -> {out_path} ({n_labels} labels in Halle frame)')
    return out_path


def main():
    print(f'Warping {len(PARCELLATIONS)} catalogued parcellations '
          f'to Halle space...\n')
    for i, (key, parc) in enumerate(PARCELLATIONS.items(), start=1):
        print(f'[{i}/{len(PARCELLATIONS)}] {key} ({parc.scope}, '
              f'{parc.n_labels} labels):')
        # Harvard-Oxford is the only key that needs a pre-merge.
        if key == 'harvard_oxford_117':
            src = _build_harvard_oxford_merged()
        else:
            src = parc.path
            if not os.path.exists(src):
                print(f'  SKIP: source file missing ({src})')
                continue
        warp_atlas_to_halle(src, f'{key}_in_halle.nii.gz',
                            mask_to_brain=parc.mask_to_brain_envelope)
        print()
    print('Done.')


if __name__ == '__main__':
    main()
