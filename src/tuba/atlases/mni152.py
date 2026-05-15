"""MNI152 ICBM 2009a non-linear symmetric T1 atlas binding.

Atlas files (expected from the upstream MNI152 release):

* ``mni_icbm152_t1_tal_nlin_sym_09a.nii``       -- 1 mm T1 template
* ``mni_icbm152_t1_tal_nlin_sym_09a_mask.nii``  -- brain mask

Registration mode: ``"intensity"``. The human Halle pipeline runs
ANTs SyN with the subject Halle CT as ``fixed`` and the MNI T1 as
``moving`` (cross-modality mutual-information cost). The forward
transforms therefore bring MNI -> Halle directly (image fwd, point
inv, per ANTs convention).

This module also publishes:

* :data:`ANATOMICAL_TARGETS_MNI` -- a name -> (x, y, z) RAS-mm dict of
  standard atlas centroids (S1, M1, V1, V5, STN, VIM thalamus, ACC,
  PCC, DLPFC, ...) used by the human placement helper.

* :data:`EEG_SITES_MNI` -- a name -> (x, y, z) RAS-mm dict of EEG
  10-20 sites (Jurcak / Koessler 2007), used to constrain the
  perpendicular-to-skull placement to a region around an experimental
  EEG location.

* :func:`available_parcellations` -- catalogue of nilearn-fetchable
  parcellation atlases that ship in MNI space and can be warped through
  the saved SyN transforms by ``warp_atlases.py``: Harvard-Oxford
  (117), Schaefer 400 / 1000, AAL3 (166), Pauli 2017 (16 subcortical),
  and Yeo 2011 (7 networks). All live on (slightly different) MNI
  grids (1.0 mm for most, 0.7 mm for Pauli); ANTs resamples in
  physical world coordinates so the source-grid mismatch is handled
  internally.

  Five of the six fetch cleanly via ``nilearn.datasets.fetch_atlas_*``.
  AAL3 is hosted at ``www.gin.cnrs.fr`` whose TLS cert is
  intermittently misconfigured (verifies as untrusted at CA-bundle
  level); one-shot manual fetch with ``curl -Lk`` followed by ``tar
  xzf`` into ``~/nilearn_data/aal_SPM12/AAL3/`` is the documented
  workaround. Subsequent loads bypass the network.
"""
from dataclasses import dataclass
import os
from typing import Optional


EEG_SITES_MNI = {
    'Fp1': (-21.5,  70.2,  -0.1),
    'Fp2': ( 28.4,  69.1,  -0.4),
    'F3':  (-39.7,  34.4,  53.9),
    'F4':  ( 41.9,  37.6,  53.4),
    'F7':  (-54.8,  33.9,  -3.5),
    'F8':  ( 56.6,  35.7,  -3.7),
    'C3':  (-52.2,  -16.4, 57.8),
    'C4':  ( 54.1,  -18.0, 57.5),
    'Cz':  (  0.4, -21.3,  87.0),
    'CP3': (-55.9, -44.5,  56.8),
    'CP4': ( 56.6, -45.8,  56.6),
    'P3':  (-52.4, -62.6,  46.4),
    'P4':  ( 53.7, -64.7,  47.6),
    'O1':  (-23.8, -94.0,  -1.3),
    'O2':  ( 25.4, -93.0,  -1.6),
    'Oz':  (  0.0, -101.6, -2.7),
    'T3':  (-72.0, -22.6, -32.4),
    'T4':  ( 72.4, -23.6, -33.5),
    'T5':  (-65.0, -68.0, -10.6),
    'T6':  ( 65.0, -68.0, -10.6),
    'AF3': (-29.3,  60.6,  35.0),
    'AF4': ( 32.4,  59.6,  35.7),
}


ANATOMICAL_TARGETS_MNI = {
    'S1_left':           (-40, -25,  55),
    'S1_right':          ( 40, -25,  55),
    'M1_left':           (-37, -21,  58),
    'M1_right':          ( 37, -21,  58),
    'V1':                (  0, -90,   0),
    'V5_left':           (-44, -67,   2),
    'V5_right':          ( 44, -67,   2),
    'STN_left':          (-12, -14,  -7),
    'STN_right':         ( 12, -14,  -7),
    'VIM_thalamus_left': (-13, -15,  10),
    'VIM_thalamus_right':( 13, -15,  10),
    'thalamus_central':  (  0, -16,   8),
    'caudate_left':      (-12,  15,  11),
    'caudate_right':     ( 12,  15,  11),
    'DLPFC_left':        (-40,  35,  35),
    'DLPFC_right':       ( 40,  35,  35),
    'ACC':               (  0,  20,  30),
    'PCC':               (  0, -50,  30),
    'insula_left':       (-38,   5,   2),
    'insula_right':      ( 38,   5,   2),
    'amygdala_left':     (-22,  -6, -18),
    'amygdala_right':    ( 22,  -6, -18),
    'entorhinal_left':   (-22, -10, -28),
    'entorhinal_right':  ( 22, -10, -28),
    'pre_SMA':           (  0,  10,  60),
    'cerebellum_central':(  0, -60, -30),
    'auditory_left':     (-50, -22,   8),
    'auditory_right':    ( 50, -22,   8),
    'temporal_lobe_left':(-50, -10, -20),
    'temporal_lobe_right':( 50, -10, -20),
    'hippocampus_left':  (-25, -20, -15),     # Li 2017 (literature MNI)
    'hippocampus_right': ( 25, -20, -15),
    'rPFC':              ( 30,  50,  30),     # Sanguinetti 2020 right PFC
    'lPFC':              (-30,  50,  30),
    'vmPFC':             (  0,  50, -10),
    'NAcc_left':         (-10,  10,  -8),
    'NAcc_right':        ( 10,  10,  -8),
    'sciatic_nerve':     None,                # handled separately
    # Additional targets used by the human-skull sweep
    'V1_left':           (-12, -90,   5),
    'V1_right':          ( 12, -90,   5),
    'A1_left':           (-50, -22,   8),     # alias for auditory_left
    'A1_right':          ( 50, -22,   8),
    'ACC_left':          ( -7,  35,  20),
    'ACC_right':         (  7,  35,  20),
    'PCC_left':          ( -7, -50,  30),
    'PCC_right':         (  7, -50,  30),
    'preSMA_left':       ( -5,  15,  65),
    'preSMA_right':      (  5,  15,  65),
    'cerebellum_left':   (-25, -65, -30),
    'cerebellum_right':  ( 25, -65, -30),
    'thalamus_left':     (-12, -18,   8),
    'thalamus_right':    ( 12, -18,   8),
}


@dataclass(frozen=True)
class MNI152:
    """Path-binding for the MNI152 ICBM 2009a non-linear symmetric atlas.

    Parameters
    ----------
    atlas_dir : str
        Directory containing the ``mni_icbm152_*`` template + mask.
    """
    atlas_dir: str
    mode: str = 'intensity'
    native_mm: float = 1.0

    @property
    def t1_path(self):
        return os.path.join(self.atlas_dir,
                            'mni_icbm152_t1_tal_nlin_sym_09a.nii')

    @property
    def brain_mask_path(self):
        return os.path.join(self.atlas_dir,
                            'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii')

    def target_ras(self, name):
        v = ANATOMICAL_TARGETS_MNI[name]
        if v is None:
            raise ValueError(f'Target {name!r} has no MNI coords '
                             f'(handled separately)')
        return v

    def eeg_ras(self, name):
        return EEG_SITES_MNI[name]


# ---------------------------------------------------------------------------
# Parcellation atlases (all live in MNI space; can be warped through the
# saved SyN transforms in ``warp_atlases.py``)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Parcellation:
    """Path-binding for an MNI-space parcellation atlas.

    ``path`` is the volumetric NIfTI on disk (typically in the nilearn
    cache or an FSL distribution). ``n_labels`` is the number of
    distinct non-background labels. ``citation`` is a short
    bibliographic anchor. ``scope`` is a single word describing what's
    parcellated (``"cortex"``, ``"cortex+subcortex+cerebellum"``,
    ``"subcortex"``, ``"networks"``). ``fetcher`` is the
    ``nilearn.datasets.*`` function that produces this file (or
    ``None`` if no auto-fetcher exists).

    ``mask_to_brain_envelope`` controls a post-warp cleanup step:
    when True, the warped Halle-space output is intersected with the
    warped MNI brain mask before being saved, zeroing out any label
    voxels that fall outside the brain envelope. AAL3 needs this
    because its native source is ~8% R-biased (more lateral cover on
    the right), and after warping into Halle the R-side temporal-pole
    / Crus2 labels otherwise spill past the brain mask and into the
    cortical bone shell (~1.7% of label volume; 90% on patient-R
    side). Other atlases (Schaefer, Pauli, Yeo, Harvard-Oxford) stop
    well inside the brain mask and don't need this cleanup.
    """
    name: str
    path: str
    n_labels: int
    scope: str
    citation: str
    fetcher: Optional[str] = None
    mask_to_brain_envelope: bool = False


def _user_cache(rel):
    return os.path.join(os.path.expanduser('~/nilearn_data'), rel)


# Bundled atlases. Each is keyed by a short canonical name.
PARCELLATIONS = {
    'harvard_oxford_117': Parcellation(
        name='Harvard-Oxford (cortl + sub merged, 117 labels)',
        path=_user_cache('fsl/data/atlases/HarvardOxford/'
                          'HarvardOxford-cortl-maxprob-thr25-1mm.nii.gz'),
        n_labels=117,
        scope='cortex+subcortex',
        citation='Desikan 2006 + FSL CMA subcortical (FSL bundle)',
        fetcher='nilearn.datasets.fetch_atlas_harvard_oxford',
    ),
    'schaefer_400_7nets': Parcellation(
        name='Schaefer 400 (7 networks)',
        path=_user_cache('schaefer_2018/'
                          'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz'),
        n_labels=400,
        scope='cortex',
        citation='Schaefer et al., Cereb Cortex 2018',
        fetcher='nilearn.datasets.fetch_atlas_schaefer_2018',
    ),
    'schaefer_1000_7nets': Parcellation(
        name='Schaefer 1000 (7 networks)',
        path=_user_cache('schaefer_2018/'
                          'Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_1mm.nii.gz'),
        n_labels=1000,
        scope='cortex',
        citation='Schaefer et al., Cereb Cortex 2018',
        fetcher='nilearn.datasets.fetch_atlas_schaefer_2018',
    ),
    'aal3': Parcellation(
        name='AAL3 (Automated Anatomical Labelling, v3)',
        path=_user_cache('aal_SPM12/AAL3/AAL3v1_1mm.nii.gz'),
        n_labels=166,
        scope='cortex+subcortex+cerebellum',
        citation='Rolls et al., NeuroImage 2020 (extends Tzourio-Mazoyer 2002)',
        fetcher='nilearn.datasets.fetch_atlas_aal',
        mask_to_brain_envelope=True,
    ),
    'pauli_2017': Parcellation(
        name='Pauli 2017 deep-brain subcortical (deterministic)',
        path=_user_cache('pauli_2017/pauli_2017_det.nii.gz'),
        n_labels=16,
        scope='subcortex',
        citation='Pauli et al., Scientific Data 2018',
        fetcher='nilearn.datasets.fetch_atlas_pauli_2017(atlas_type="deterministic")',
    ),
    'yeo_7nets': Parcellation(
        name='Yeo 2011 (7 functional networks)',
        path=_user_cache('yeo_2011/Yeo_JNeurophysiol11_MNI152/'
                          'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz'),
        n_labels=7,
        scope='networks',
        citation='Yeo et al., J Neurophysiol 2011',
        fetcher='nilearn.datasets.fetch_atlas_yeo_2011',
    ),
}


def available_parcellations():
    """Return the catalogue of bundled MNI parcellations as a list of
    :class:`Parcellation` objects. Use :func:`get_parcellation` to look
    up by canonical key."""
    return list(PARCELLATIONS.values())


def get_parcellation(key):
    """Look up a bundled parcellation by canonical key (e.g.
    ``'harvard_oxford_117'``, ``'aal3'``, ``'pauli_2017'``,
    ``'yeo_7nets'``, ``'schaefer_400_7nets'``,
    ``'schaefer_1000_7nets'``).
    """
    if key not in PARCELLATIONS:
        raise KeyError(
            f'Unknown parcellation {key!r}. Known: '
            f'{sorted(PARCELLATIONS)}.')
    return PARCELLATIONS[key]
