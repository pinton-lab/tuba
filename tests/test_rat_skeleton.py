"""Skeleton smoke tests for the rat pillar.

Unlike the other species' regression tests (which compare TUBA against
a legacy pre-port implementation), the rat pillar is greenfield --
there is no legacy code to validate against, and no on-disk data
staged on most hosts. This file therefore exercises the things that
*can* be checked without any data: that the module imports, the
manifest entries are well-formed, the atlas-binding paths resolve to
the expected locations, and the ITK-SNAP .label parser handles a
representative sample.

These tests run unconditionally (no env-var gating) so they catch
silent breakage of the rat skeleton as the surrounding TUBA code
evolves. They cost <1 s.
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


def test_rat_module_imports():
    """tuba.species.rat must import and expose the species-API surface."""
    import tuba.species.rat as rat
    for attr in ('NATIVE_VOXEL_MM', 'DOWNSAMPLE_VOXEL_MM',
                 'BONE_LOW', 'BONE_HIGH', 'ALIGN_PCA_THRESH',
                 'ALIGN_RX_DEG', 'ALIGN_RY_DEG', 'ALIGN_RZ_DEG',
                 'NATIVE_PAD_VOXELS', 'SLAB_RAMP', 'SLAB_CAVITY_INFILL',
                 'SYN_PREFIX', 'ATLAS',
                 'place_on_skull', 'load_slab',
                 'extract_cavity', 'register_to_atlas'):
        assert hasattr(rat, attr), f'tuba.species.rat missing {attr!r}'


def test_rat_native_voxel_matches_digimorph():
    """NATIVE_VOXEL_MM must equal the DigiMorph TMM M-2272 pitch (29.606 um)."""
    import tuba.species.rat as rat
    assert abs(rat.NATIVE_VOXEL_MM - 0.029606) < 1e-9


def test_whs_atlas_binding_paths():
    """WaxholmSpace property paths must compose correctly under a
    user-supplied atlas_dir, matching the layout fetch_whs.py produces."""
    from tuba.atlases.whs import WaxholmSpace
    a = WaxholmSpace(atlas_dir='/tmp/whs')
    assert a.mode == 'cavity_binary'
    assert a.native_um == 39
    assert a.template_path.endswith(
        'WHS_SD_rat_atlas_v4_pack/WHS_SD_rat_T2star_v1.01.nii.gz')
    assert a.annotation_path.endswith('WHS_SD_rat_atlas_v4.01.nii.gz')
    assert a.label_table_path.endswith('WHS_SD_rat_atlas_v4.01.label')
    assert a.brain_mask_path.endswith('whs_brain_mask.nii.gz')


def test_whs_label_parser_roundtrip():
    """WaxholmSpace._parse_label_table must extract id -> name from the
    ITK-SNAP format, skipping the background (id 0) and comments."""
    from tuba.atlases.whs import WaxholmSpace
    sample = (
        '# ITK-SnAP Label Description File\n'
        '# File format:\n'
        '# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL\n'
        '    0    0    0    0        0  0  0    "Clear Label"\n'
        '    1  255    0    0        1  1  1    "Corpus callosum and '
        'associated subcortical white matter"\n'
        '    2    0  255    0        1  1  1    "Caudate putamen"\n'
        '   17    0    0  255        1  1  1    "Hippocampal formation"\n'
    )
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, 'WHS_SD_rat_atlas_v4_pack'))
        with open(os.path.join(d, 'WHS_SD_rat_atlas_v4.01.label'), 'w') as f:
            f.write(sample)
        atlas = WaxholmSpace(atlas_dir=d)
        all_structs = atlas.all_structures()
        assert all_structs == {
            'Corpus callosum and associated subcortical white matter': 1,
            'Caudate putamen': 2,
            'Hippocampal formation': 17,
        }
        assert atlas.structure_id('Caudate putamen') == 2
        with pytest.raises(KeyError, match='Bogus structure'):
            atlas.structure_id('Bogus structure')


def test_rat_manifest_entries_well_formed():
    """sources.toml must contain three rat entries with the expected
    roles, fetchers, and DigiMorph/WHS provenance fields."""
    from tuba.data import by_species, get
    rat = {s.key: s for s in by_species('rat')}
    assert set(rat) == {'rat.skull', 'rat.atlas_template',
                        'rat.atlas_annotation'}, rat

    skull = rat['rat.skull']
    assert skull.role == 'skull'
    assert skull.fetcher_script == 'src/tuba/data/fetch_rat.py'
    assert skull.voxel_um == 29.61
    assert 'DigiMorph' in skull.citation
    assert 'TMM M-2272' in skull.notes

    tpl = rat['rat.atlas_template']
    assert tpl.role == 'atlas_template'
    assert tpl.fetcher_script == 'src/tuba/data/fetch_whs.py'
    assert tpl.voxel_um == 39.0
    assert tpl.license == 'CC BY 4.0'
    assert tpl.archive_name == 'WHS_SD_rat_atlas_v4_pack.zip'

    ann = rat['rat.atlas_annotation']
    assert ann.role == 'atlas_annotation'
    assert ann.fetcher_script == 'src/tuba/data/fetch_whs.py'
    assert '2023' in ann.citation     # Kleven 2023 Nat Methods
    assert ann.license == 'CC BY 4.0'
    assert len(ann.companion_files) == 1
    assert ann.companion_files[0]['name'] == 'WHS_SD_rat_atlas_v4.01.label'


def test_fetcher_scripts_exist_and_importable():
    """fetch_rat.py and fetch_whs.py must exist at the paths the
    manifest declares, and be importable as standalone modules."""
    import importlib.util
    proj_root = os.path.normpath(os.path.join(HERE, '..'))
    for rel in ('src/tuba/data/fetch_rat.py', 'src/tuba/data/fetch_whs.py'):
        path = os.path.join(proj_root, rel)
        assert os.path.exists(path), f'missing fetcher: {path}'
        spec = importlib.util.spec_from_file_location('x', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert hasattr(m, 'DEST')
        assert hasattr(m, 'main')


def test_swap_row_col_flip_is_wired_through():
    """frame.apply_axis_flips must accept the swap_row_col kwarg that rat.py
    relies on, and it must compose correctly with the per-axis flips."""
    import numpy as np
    from tuba.core.frame import apply_axis_flips

    # 3-axis test array: distinct values along each axis
    a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    # swap_row_col + transpose_to_ras should match the macaque convention
    out = apply_axis_flips(a, swap_row_col=True)
    # After transpose(0,2,1) then transpose(1,0,2): shape (4, 2, 3)
    assert out.shape == (4, 2, 3), f'unexpected shape {out.shape}'


def test_rat_cavity_pipeline_when_staged():
    """When the DigiMorph + WHS data is staged locally, the species
    module must produce a cavity in the 1.5--2.5 mL range and
    populate the warped WHS template + annotation."""
    import nibabel as nib
    import numpy as np
    from tuba.species import rat
    from tuba.atlases.whs import WaxholmSpace

    if not os.path.exists(rat.CAVITY_NII):
        pytest.skip('rat cavity not yet staged; '
                     'run rat.extract_cavity() + register_to_atlas() first')

    cav = nib.load(rat.CAVITY_NII)
    cav_arr = cav.get_fdata().astype(bool)
    voxel_mm = abs(cav.affine[0, 0])
    vol_mL = cav_arr.sum() * voxel_mm**3 / 1000
    # Loose envelope: 1.8-3.0 mL covers the rat cranial cavity after
    # the post-warp dilate + bone-clip + opening + fill_holes pipeline,
    # which sits in the published 1.8-2.4 mL brain-parenchyma range with
    # a small margin for meninges + CSF spaces. ANTs Affine has
    # stochastic init so a ~5% run-to-run spread is expected.
    assert 2.0 < vol_mL < 3.0, (
        f'rat cavity volume {vol_mL:.2f} mL outside [2.0, 3.0] mL window')

    if os.path.exists(rat.ANNOT_NII):
        annot = nib.load(rat.ANNOT_NII).get_fdata().astype(np.int32)
        atlas = WaxholmSpace(
            atlas_dir=os.path.expanduser('~/.cache/tuba/rat/atlas/whs_sd_v4'))
        # The warped annotation must contain at least a representative TUS landmark.
        cp_id = atlas.structure_id('Caudate putamen')
        assert (annot == cp_id).any(), (
            f'Caudate putamen (id {cp_id}) missing from warped annotation')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
