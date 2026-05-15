"""Allen Mouse Brain Common Coordinate Framework v3 (CCFv3) binding.

Atlas files (expected to be pre-cached by an external fetcher script
like ``mouse_therapy/templates/fetch_allen_ccfv3.py``):

* ``allen_template_10um.nii.gz``    -- 10 um intensity template
* ``allen_template_25um.nii.gz``    -- 25 um template (used as the
                                       moving image in SyN registration
                                       for speed; ~70 s)
* ``allen_annotation_10um.nii.gz``  -- 10 um structure annotation
                                       (594 labels after cavity restriction)
* ``allen_annotation_25um.nii.gz``  -- 25 um annotation

Registration mode: ``"cavity_binary"``. The brain mask is derived from
the template by thresholding at 5% of the template maximum.
"""
from dataclasses import dataclass
import os


# Structure IDs used by mouse-FUS placement. See the Allen Brain Atlas
# API for the full ontology.
STRUCTURES = {
    'CC_body':   484682516,   # corpus callosum, body
    'CC_top':    776,         # alias used in some annotation revisions
}


@dataclass(frozen=True)
class AllenCCFv3:
    """Path-binding for a cached Allen CCFv3 directory.

    Parameters
    ----------
    atlas_dir : str
        Directory containing ``allen_template_*um.nii.gz`` and
        ``allen_annotation_*um.nii.gz``.
    brain_mask_threshold_frac : float
        Fraction of template max used to derive a binary brain mask.
        The mouse Maga pipeline uses 0.05 (5%); change here only if
        the upstream atlas representation changes.
    """
    atlas_dir: str
    brain_mask_threshold_frac: float = 0.05
    mode: str = 'cavity_binary'
    native_um: int = 10
    working_um: int = 25

    @property
    def template_10um(self):
        return os.path.join(self.atlas_dir, 'allen_template_10um.nii.gz')

    @property
    def template_25um(self):
        return os.path.join(self.atlas_dir, 'allen_template_25um.nii.gz')

    @property
    def annotation_10um(self):
        return os.path.join(self.atlas_dir, 'allen_annotation_10um.nii.gz')

    @property
    def annotation_25um(self):
        return os.path.join(self.atlas_dir, 'allen_annotation_25um.nii.gz')

    @property
    def brain_mask_path(self):
        return os.path.join(self.atlas_dir, 'allen_brain_mask.nii.gz')

    def derive_brain_mask(self, force=False, verbose=True):
        """Threshold ``allen_template_25um`` at ``brain_mask_threshold_frac``
        of its max, save to ``allen_brain_mask.nii.gz``, return path.
        This is the SyN fixed image for ``mode='cavity_binary'``.
        """
        import ants
        import numpy as np
        if os.path.exists(self.brain_mask_path) and not force:
            return self.brain_mask_path
        if verbose:
            print(f'Deriving Allen brain mask from {self.template_25um}...')
        t = ants.image_read(self.template_25um)
        arr = t.numpy()
        mask = (arr > arr.max() * self.brain_mask_threshold_frac).astype(np.float32)
        mask_img = t.new_image_like(mask)
        ants.image_write(mask_img, self.brain_mask_path)
        if verbose:
            print(f'  Allen brain mask: {int(mask.sum())} voxels')
            print(f'  wrote {self.brain_mask_path}')
        return self.brain_mask_path

    def structure_id(self, name):
        """Look up a structure ID by name (e.g. ``'CC_body'``)."""
        return STRUCTURES[name]
