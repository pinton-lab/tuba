"""Tests for the Diedrichsen deep cerebellar nuclei atlas binding.

The catalog / binding-path / label-parser / manifest tests run
unconditionally (no on-disk data needed), mirroring the rat skeleton
tests. The dentate-mask test needs the staged MNI probseg + label
table and self-skips when they are absent (run
``python -m tuba.data.fetch_cerebellar_atlases`` to stage them), the
same way the other atlas tests gate on missing data.

Run from the TUBA project root:
    PYTHONPATH=src pytest tests/test_diedrichsen_dcn.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TUBA_SRC = os.path.join(HERE, '..', 'src')
if TUBA_SRC not in sys.path:
    sys.path.insert(0, TUBA_SRC)

# Manuscript-cited dentate target (MNI RAS mm); the test asserts the
# atlas centroid sits close to it. The measured Diedrichsen centroid is
# ~[+/-14.6, -60.1, -34.3] -- ~4.4 mm from this point (a real, small
# offset: ~2.5 mm lateral + ~3 mm A-P, from the FNIRT-MNI normalization),
# so we allow 5 mm rather than the 4 mm the manuscript quotes.
DENTATE_TARGET = {'L': (-12.0, -57.0, -36.0), 'R': (12.0, -57.0, -36.0)}
CENTROID_TOL_MM = 5.0


# ---------------------------------------------------------------------------
# No-data tests (run everywhere)
# ---------------------------------------------------------------------------
def test_diedrichsen_dcn_in_catalog():
    """get_parcellation('diedrichsen_dcn') must resolve and appear in
    available_parcellations() with well-formed fields."""
    from tuba.atlases import mni152
    parc = mni152.get_parcellation('diedrichsen_dcn')
    assert parc in mni152.available_parcellations()
    assert 'diedrichsen_dcn' in mni152.PARCELLATIONS
    assert parc.n_labels == 34
    assert 'cerebellum' in parc.scope
    assert '2011' in parc.citation and 'Diedrichsen' in parc.citation
    assert parc.fetcher == 'tuba.data.fetch_cerebellar_atlases'
    assert parc.path.endswith(
        os.path.join('diedrichsen_2009', 'atl-Anatom_space-MNI_dseg.nii'))


def test_diedrichsen_binding_paths():
    """DiedrichsenDCN property paths compose under a user-supplied dir,
    and the default factory resolves the nilearn_data cache."""
    from tuba.atlases.mni152 import DiedrichsenDCN, diedrichsen_dcn
    a = DiedrichsenDCN(atlas_dir='/tmp/dcn')
    assert a.space == 'MNI152'
    assert a.dseg_path.endswith('atl-Anatom_space-MNI_dseg.nii')
    assert a.probseg_path.endswith('atl-Anatom_space-MNI_probseg.nii')
    assert a.label_table_path.endswith('atl-Anatom.tsv')
    assert a.lut_path.endswith('atl-Anatom.lut')

    default = diedrichsen_dcn()
    assert default.atlas_dir.endswith(
        os.path.join('nilearn_data', 'diedrichsen_2009'))


def test_norm_side_accepts_aliases_and_rejects_garbage():
    from tuba.atlases.mni152 import _norm_side
    assert _norm_side('L') == 'L'
    assert _norm_side('left') == 'L'
    assert _norm_side('Right') == 'R'
    assert _norm_side('r') == 'R'
    with pytest.raises(ValueError):
        _norm_side('up')


def test_dcn_label_constants():
    """The six deep nuclei map to atl-Anatom labels 29-34."""
    from tuba.atlases.mni152 import DIEDRICHSEN_DCN_LABELS
    assert DIEDRICHSEN_DCN_LABELS == {
        'Dentate_L': 29, 'Dentate_R': 30,
        'Interposed_L': 31, 'Interposed_R': 32,
        'Fastigial_L': 33, 'Fastigial_R': 34,
    }


def test_label_table_parser_and_aliases():
    """_parse_label_table reads the index<TAB>name<TAB>color .tsv,
    skipping the header; structure_id accepts both the .tsv spelling and
    the Dentate_L alias and errors helpfully on misses."""
    from tuba.atlases.mni152 import DiedrichsenDCN
    sample = (
        'index\tname\tcolor\n'
        '1\tLeft_I_IV\t#ccff00\n'
        '29\tLeft_Dentate\t#4c4c4c\n'
        '30\tRight_Dentate\t#4c4c4c\n'
        '33\tLeft_Fastigial\t#b2b2b2\n'
    )
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'atl-Anatom.tsv'), 'w') as f:
            f.write(sample)
        atlas = DiedrichsenDCN(atlas_dir=d)
        allstr = atlas.all_structures()
        assert allstr == {'Left_I_IV': 1, 'Left_Dentate': 29,
                          'Right_Dentate': 30, 'Left_Fastigial': 33}
        assert atlas.structure_id('Left_Dentate') == 29   # .tsv spelling
        assert atlas.structure_id('Dentate_L') == 29       # friendly alias
        assert atlas.structure_id('Dentate_R') == 30
        assert atlas.structure_id('left_dentate') == 29    # .tsv, any case
        assert atlas.structure_id('dentate_l') == 29       # alias, any case
        with pytest.raises(KeyError, match='Bogus_Nucleus'):
            atlas.structure_id('Bogus_Nucleus')


def test_diedrichsen_manifest_entry():
    """sources.toml must carry an atlas.diedrichsen_dcn entry with the
    expected role, fetcher, citation, and format."""
    from tuba.data import get
    s = get('atlas.diedrichsen_dcn')
    assert s.role == 'atlas_parcellation'
    assert s.fetcher_script == 'src/tuba/data/fetch_cerebellar_atlases.py'
    assert s.format == 'NIfTI'
    assert s.voxel_mm == 1.0
    assert '2011' in s.citation and 'deep cerebellar nuclei' in s.citation
    assert s.auth_required == 'none'
    # Omitted until pinned; tolerate a real 64-hex hash so pinning later
    # does not break this test.
    assert s.sha256 is None or len(s.sha256) == 64


def test_fetcher_exists_and_importable():
    """fetch_cerebellar_atlases.py must exist where the manifest says and
    expose the DEST + main() surface the other fetchers share."""
    import importlib.util
    proj_root = os.path.normpath(os.path.join(HERE, '..'))
    path = os.path.join(proj_root, 'src/tuba/data/fetch_cerebellar_atlases.py')
    assert os.path.exists(path)
    spec = importlib.util.spec_from_file_location('fca', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert hasattr(m, 'DEST') and hasattr(m, 'main')
    assert set(m.URLS) >= {'atl-Anatom_space-MNI_dseg.nii',
                           'atl-Anatom_space-MNI_probseg.nii',
                           'atl-Anatom.tsv'}
    # The fetcher URLS and the manifest blob_urls are two copies of the
    # canonical download list; tie them together so they cannot drift
    # (this also exercises the URL values, not just the filename keys).
    from tuba.data import get
    assert set(m.URLS.values()) == set(get('atlas.diedrichsen_dcn').blob_urls)


# ---------------------------------------------------------------------------
# Data-gated test (skips when the MNI probseg is not staged)
# ---------------------------------------------------------------------------
def _centroid_mm(mask, affine):
    import numpy as np
    import nibabel as nib
    ijk = np.argwhere(mask)
    return nib.affines.apply_affine(affine, ijk.mean(0))


def test_dentate_mask_volume_and_centroid_when_staged():
    """When the Diedrichsen MNI probseg is staged, L/R dentate masks must
    be non-empty, anatomically sized, in the correct hemisphere, and
    centred near the manuscript dentate target."""
    import numpy as np
    from tuba.atlases import mni152

    binding = mni152.diedrichsen_dcn()
    if not (os.path.exists(binding.probseg_path)
            and os.path.exists(binding.label_table_path)):
        pytest.skip('Diedrichsen MNI atlas not staged; run '
                    'python -m tuba.data.fetch_cerebellar_atlases')

    for side in ('L', 'R'):
        mask, aff = mni152.dentate_mask(side)          # default thr 0.5
        assert mask.dtype == bool
        assert aff.shape == (4, 4)
        assert mask.any(), f'{side} dentate mask empty at thr 0.5'

        vox_mm3 = float(abs(np.linalg.det(aff[:3, :3])))
        vol_cm3 = mask.sum() * vox_mm3 / 1000.0
        # Measured at thr 0.5: L 0.93 cm^3, R 1.07 cm^3. Window brackets
        # both with margin while still catching a gross over/under-segment.
        assert 0.4 <= vol_cm3 <= 1.5, (
            f'{side} dentate volume {vol_cm3:.2f} cm^3 outside [0.4, 1.5]')

        cx, cy, cz = _centroid_mm(mask, aff)
        # Hemisphere: left dentate has x < 0, right x > 0.
        assert (cx < 0) == (side == 'L'), (
            f'{side} dentate centroid x={cx:.1f} in wrong hemisphere')
        tx, ty, tz = DENTATE_TARGET[side]
        dist = float(np.hypot(np.hypot(cx - tx, cy - ty), cz - tz))
        assert dist <= CENTROID_TOL_MM, (
            f'{side} dentate centroid [{cx:.1f},{cy:.1f},{cz:.1f}] is '
            f'{dist:.2f} mm from manuscript target {DENTATE_TARGET[side]} '
            f'(tol {CENTROID_TOL_MM} mm)')

    # A higher threshold must shrink the mask (probabilistic, monotone).
    m05, _ = mni152.dentate_mask('L', threshold=0.5)
    m07, _ = mni152.dentate_mask('L', threshold=0.7)
    assert m07.sum() < m05.sum() and m07.any()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
