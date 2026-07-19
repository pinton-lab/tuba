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

* :func:`available_parcellations` -- catalogue of parcellation atlases
  that ship in MNI space and can be warped through the saved SyN
  transforms by ``warp_atlases.py``: Harvard-Oxford (117), Schaefer
  400 / 1000, AAL3 (166), Pauli 2017 (16 subcortical), Yeo 2011 (7
  networks), and the Diedrichsen cerebellar atlas (34 labels: 28
  lobules + 6 deep cerebellar nuclei). All live on (slightly different)
  MNI grids (1.0 mm for most, 0.7 mm for Pauli); ANTs resamples in
  physical world coordinates so the source-grid mismatch is handled
  internally.

  Five of the six nilearn-style atlases fetch cleanly via
  ``nilearn.datasets.fetch_atlas_*``. AAL3 is hosted at
  ``www.gin.cnrs.fr`` whose TLS cert is intermittently misconfigured
  (verifies as untrusted at CA-bundle level); one-shot manual fetch
  with ``curl -Lk`` followed by ``tar xzf`` into
  ``~/nilearn_data/aal_SPM12/AAL3/`` is the documented workaround.
  Subsequent loads bypass the network. The Diedrichsen cerebellar
  atlas is fetched by :mod:`tuba.data.fetch_cerebellar_atlases`; its
  deep cerebellar nuclei (dentate / interposed / fastigial) are
  exposed through :class:`DiedrichsenDCN` and the module-level
  :func:`dentate_mask` helper (a probabilistic atlas, thresholded to a
  binary mask per nucleus).
"""
from dataclasses import dataclass, field
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
    'amPFC':             ( -5,  45,  -3),     # Schachtner 2026 left anterior medial PFC (MDD)
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
    ``nilearn.datasets.*`` function or tuba fetcher module (e.g.
    ``"tuba.data.fetch_cerebellar_atlases"``) that produces this file
    (or ``None`` if no auto-fetcher exists).

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
    'diedrichsen_dcn': Parcellation(
        name='Diedrichsen cerebellar atlas (28 lobules + 6 deep nuclei; '
             'dentate / interposed / fastigial, L+R)',
        path=_user_cache('diedrichsen_2009/atl-Anatom_space-MNI_dseg.nii'),
        n_labels=34,
        scope='cerebellum+nuclei',
        citation='Diedrichsen et al., NeuroImage 54(3):1786-1794, 2011 '
                 '(deep cerebellar nuclei); Diedrichsen et al., '
                 'NeuroImage 2009 (lobules)',
        fetcher='tuba.data.fetch_cerebellar_atlases',
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
    ``'schaefer_1000_7nets'``, ``'diedrichsen_dcn'``).
    """
    if key not in PARCELLATIONS:
        raise KeyError(
            f'Unknown parcellation {key!r}. Known: '
            f'{sorted(PARCELLATIONS)}.')
    return PARCELLATIONS[key]


# ---------------------------------------------------------------------------
# Diedrichsen deep cerebellar nuclei (probabilistic, MNI space)
# ---------------------------------------------------------------------------
# The ``diedrichsen_dcn`` parcellation above points at the discrete dseg
# (warpable like the others). The probabilistic per-nucleus maps and the
# structure-name lookups live on the :class:`DiedrichsenDCN` binding
# below, which mirrors the whs / allen pattern.

_DIEDRICHSEN_SUBDIR = 'diedrichsen_2009'
_DIEDRICHSEN_DSEG = 'atl-Anatom_space-MNI_dseg.nii'
_DIEDRICHSEN_PROBSEG = 'atl-Anatom_space-MNI_probseg.nii'
_DIEDRICHSEN_LABELS = 'atl-Anatom.tsv'
_DIEDRICHSEN_LUT = 'atl-Anatom.lut'

# The six deep cerebellar nuclei (labels 29-34 in atl-Anatom). The atlas
# .tsv spells these ``<Side>_<Nucleus>``; we also accept the friendlier
# ``<Nucleus>_<Side>`` aliases used by tuba consumers and the manuscript.
DIEDRICHSEN_DCN_LABELS = {
    'Dentate_L': 29, 'Dentate_R': 30,
    'Interposed_L': 31, 'Interposed_R': 32,
    'Fastigial_L': 33, 'Fastigial_R': 34,
}

# ``<Nucleus>_<Side>`` alias -> the canonical ``<Side>_<Nucleus>`` .tsv
# spelling. ``structure_id`` resolves an alias to its .tsv name and then
# requires that name in the parsed table, so a truncated/corrupt .tsv
# surfaces as a KeyError instead of silently returning a hardcoded ID
# (the .tsv stays the source of truth).
_DIEDRICHSEN_DCN_ALIASES = {
    f'{nuc}_{abbr}': f'{word}_{nuc}'
    for nuc in ('Dentate', 'Interposed', 'Fastigial')
    for abbr, word in (('L', 'Left'), ('R', 'Right'))
}

# Case-insensitive view of the alias map so ``structure_id('dentate_l')``
# resolves the same as ``structure_id('Dentate_L')`` (the .tsv spellings
# already get a case-insensitive fallback in structure_id).
_DIEDRICHSEN_DCN_ALIASES_LOWER = {
    k.lower(): v for k, v in _DIEDRICHSEN_DCN_ALIASES.items()
}


def _norm_side(side):
    """Normalise a side argument to ``'L'`` or ``'R'``."""
    s = str(side).strip().lower()
    if s in ('l', 'left'):
        return 'L'
    if s in ('r', 'right'):
        return 'R'
    raise ValueError(
        f"side must be 'L'/'R' (or 'left'/'right'); got {side!r}")


@dataclass
class DiedrichsenDCN:
    """Binding for the Diedrichsen probabilistic cerebellar atlas in
    MNI152 space, with first-class access to the six deep cerebellar
    nuclei (dentate / interposed / fastigial, left + right; labels
    29-34).

    The atlas ships two co-registered MNI NIfTIs (1 mm, infratentorial
    FOV, affine origin ``[76, -108, -72]``), staged by
    :mod:`tuba.data.fetch_cerebellar_atlases`:

    * ``atl-Anatom_space-MNI_dseg.nii`` -- discrete "most probable
      compartment" label volume (34 labels: 28 lobules + 6 nuclei).
      This is the volume registered in :data:`PARCELLATIONS` and warped
      to Halle by ``warp_atlases.py`` (nearestNeighbor preserves IDs).
    * ``atl-Anatom_space-MNI_probseg.nii`` -- 4-D per-compartment
      probability maps (``153x103x84x34``, prob in ``[0, 1]``); volume
      ``v`` holds label ``v + 1``. :meth:`nucleus_mask` thresholds
      these into a binary ROI.

    Measured dentate centroids (probseg, thr 0.5): left ~``[-14.6,
    -60.1, -34.3]`` mm (0.93 cm^3), right ~``[15.6, -59.5, -34.6]`` mm
    (1.07 cm^3) -- ~4-5 mm from the manuscript-cited ``[+/-12, -57,
    -34..-36]`` (a slightly medial/superior approximation; the offset
    is real and reflects the Diedrichsen FNIRT-MNI normalization).

    Space caveat: SUIT-space variants (``*_space-SUIT_*``) also exist
    upstream; tuba binds the MNI files so the existing MNI -> Halle SyN
    warp applies.

    Parameters
    ----------
    atlas_dir : str
        Directory holding the ``atl-Anatom_space-MNI_*`` NIfTIs plus the
        ``atl-Anatom.tsv`` label table. :func:`diedrichsen_dcn` resolves
        the default ``~/nilearn_data/diedrichsen_2009`` for you.
    """
    atlas_dir: str
    space: str = 'MNI152'
    _structure_cache: dict = field(default_factory=dict, repr=False,
                                   compare=False)

    @property
    def dseg_path(self) -> str:
        return os.path.join(self.atlas_dir, _DIEDRICHSEN_DSEG)

    @property
    def probseg_path(self) -> str:
        return os.path.join(self.atlas_dir, _DIEDRICHSEN_PROBSEG)

    @property
    def label_table_path(self) -> str:
        return os.path.join(self.atlas_dir, _DIEDRICHSEN_LABELS)

    @property
    def lut_path(self) -> str:
        return os.path.join(self.atlas_dir, _DIEDRICHSEN_LUT)

    def _parse_label_table(self) -> dict:
        """Parse ``atl-Anatom.tsv`` into a ``name -> id`` dict (34
        entries). Format: a header line ``index<TAB>name<TAB>color``
        followed by one tab-separated row per label."""
        if self._structure_cache:
            return self._structure_cache
        path = self.label_table_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Diedrichsen label table not found: {path}. '
                f'Run tuba.data.fetch_cerebellar_atlases first.')
        out: dict = {}
        with open(path) as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2 or not parts[0].strip().isdigit():
                    continue  # header / blank line
                out[parts[1].strip()] = int(parts[0])
        self._structure_cache.update(out)
        return self._structure_cache

    def structure_id(self, name: str) -> int:
        """Look up a label ID by structure name. Accepts the atlas .tsv
        spelling (``'Left_Dentate'``) and the ``'Dentate_L'`` aliases
        for the six deep nuclei; falls back to a case-insensitive match,
        then raises ``KeyError`` with near-miss hints.

        Aliases resolve to their canonical .tsv spelling and are then
        looked up in the parsed table, so the .tsv remains the source of
        truth -- a missing/corrupt table raises rather than returning a
        hardcoded ID.
        """
        table = self._parse_label_table()
        canonical = _DIEDRICHSEN_DCN_ALIASES.get(
            name, _DIEDRICHSEN_DCN_ALIASES_LOWER.get(name.lower(), name))
        if canonical in table:
            return table[canonical]
        lower = {k.lower(): v for k, v in table.items()}
        if canonical.lower() in lower:
            return lower[canonical.lower()]
        hits = [k for k in table if name.lower() in k.lower()]
        hint = f'  similar: {hits[:5]}' if hits else ''
        raise KeyError(
            f'Structure {name!r} not in Diedrichsen atlas.{hint}')

    def all_structures(self) -> dict:
        """Return the full ``name -> id`` map (34 entries) from the
        ``atl-Anatom.tsv`` label table."""
        return dict(self._parse_label_table())

    def nucleus_mask(self, name, threshold: float = 0.5):
        """Return ``(mask_bool_3d, affine_4x4)`` for a named compartment,
        thresholding its probabilistic map at ``threshold`` (prob in
        ``[0, 1]``; default 0.5 keeps voxels more likely than not to
        belong to the compartment).

        ``name`` accepts any label name (e.g. ``'Left_Dentate'`` or the
        ``'Dentate_L'`` alias). For the dentate specifically see
        :meth:`dentate_mask`.

        Note: the dentate and interposed nuclei are well populated at
        thr 0.5, but the small fastigial nuclei peak below 0.5 (max
        ~0.23), so ``nucleus_mask('Fastigial_L')`` is empty at the
        default threshold -- pass a lower ``threshold`` (e.g. 0.1-0.2)
        for those, and guard against an empty mask before taking a
        centroid (``argwhere(mask).mean(0)`` of an empty mask is NaN).
        """
        import numpy as np
        import nibabel as nib
        label_id = self.structure_id(name)
        path = self.probseg_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Diedrichsen probseg not found: {path}. '
                f'Run tuba.data.fetch_cerebellar_atlases first.')
        img = nib.load(path)
        if img.ndim != 4:
            raise ValueError(
                f'expected a 4-D probseg at {path}, got shape {img.shape}')
        # Volume index v holds label v + 1; slicing the proxy applies the
        # scl_slope/inter so values land in [0, 1] and only one volume is
        # read into memory.
        vol = np.asanyarray(img.dataobj[..., label_id - 1], dtype=np.float32)
        mask = vol >= float(threshold)
        return mask, img.affine

    def dentate_mask(self, side='L', threshold: float = 0.5):
        """Return ``(mask_bool_3d, affine_4x4)`` for the left or right
        dentate nucleus. ``side`` is ``'L'``/``'R'`` (also
        ``'left'``/``'right'``, case-insensitive)."""
        return self.nucleus_mask(f'Dentate_{_norm_side(side)}',
                                 threshold=threshold)


def diedrichsen_dcn(atlas_dir: Optional[str] = None) -> DiedrichsenDCN:
    """Return a :class:`DiedrichsenDCN` bound to the cached MNI atlas.

    With ``atlas_dir=None`` (the default) it resolves
    ``~/nilearn_data/diedrichsen_2009`` -- the location
    :mod:`tuba.data.fetch_cerebellar_atlases` stages into and the one
    the :data:`PARCELLATIONS` ``diedrichsen_dcn`` entry points at.
    """
    if atlas_dir is None:
        atlas_dir = _user_cache(_DIEDRICHSEN_SUBDIR)
    return DiedrichsenDCN(atlas_dir=atlas_dir)


def dentate_mask(side='L', threshold: float = 0.5,
                 atlas_dir: Optional[str] = None):
    """Return ``(mask_bool_3d, affine_4x4)`` for the MNI dentate nucleus.

    This is the downstream entry point for the human Fig. 1 renderer::

        from tuba.atlases import mni152
        mask, aff = mni152.dentate_mask('L')          # left dentate, thr 0.5

    ``mask`` is a boolean array on the atlas's 1 mm MNI grid; ``aff`` is
    its voxel -> world (MNI RAS mm) affine, so ``nibabel.affines.
    apply_affine(aff, np.argwhere(mask).mean(0))`` gives the centroid in
    MNI mm. The Diedrichsen 2011 dentate centroid lands at ~``[-14.6,
    -60.1, -34.3]`` mm (left) / ~``[15.6, -59.5, -34.6]`` mm (right) at
    threshold 0.5.
    """
    return diedrichsen_dcn(atlas_dir).dentate_mask(side, threshold=threshold)
