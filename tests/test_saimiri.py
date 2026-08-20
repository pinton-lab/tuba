"""Offline unit tests for the squirrel-monkey (Saimiri) pillar.

No network, no antspyx, no staged scan. Covers the deliverable-1/2 logic
that must hold regardless of whether the heavy data/deps are present:

1. importing the pillar must NOT import antspyx (ANTs-free consumer
   contract -- ANTs is lazy inside tuba.core.warp / the atlas binding);
2. the HU->acoustics calibration guard refuses uncalibrated input and
   the calibrated Aubry mapping is physically monotonic;
3. the points-per-wavelength + time-of-flight-aberration + grid-
   convergence maths (deliverable 1);
4. the VALiDATe29 file/label resolution + S1/M1 target keyword mapping
   (deliverable 2), and a synthetic affine registration round-trip;
5. the data manifest carries the three Saimiri entries with the expected
   calibration/license semantics.
"""
import os
import subprocess
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. ANTs-free import contract
# ---------------------------------------------------------------------------
def test_import_does_not_pull_in_antspyx():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    code = (
        'import sys; '
        'import tuba.species.saimiri, tuba.atlases.validate29, '
        'tuba.core.hu_acoustics, tuba.data.fetch_saimiri, '
        'tuba.data.fetch_validate29; '
        'assert "ants" not in sys.modules, sorted(m for m in sys.modules '
        'if "ant" in m.lower()); '
        'print("ok")')
    r = subprocess.run([sys.executable, '-c', code], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert 'ok' in r.stdout


# ---------------------------------------------------------------------------
# 2. HU->acoustics calibration guard + calibrated mapping physics
# ---------------------------------------------------------------------------
def test_uncalibrated_mapping_refuses():
    from tuba.core import hu_acoustics as ha
    m = ha.AubryHUMapping(calibration=ha.UNCALIBRATED_MICROCT)
    for call in (lambda: m.properties(1500.0),
                 lambda: m(1500.0),
                 lambda: m.porosity(1500.0)):
        with pytest.raises(ha.UncalibratedInputError):
            call()


def test_calibrated_mapping_is_physical_and_monotonic():
    from tuba.core import hu_acoustics as ha
    cal = ha.CTCalibration(kind='hu', hu_water=0.0, hu_bone_max=2000.0,
                           source='synthetic phantom')
    m = ha.AubryHUMapping(calibration=cal)

    # Endpoints: water and dense cortical bone.
    rho_w, c_w, a_w = m.properties(0.0)
    rho_b, c_b, a_b = m.properties(2000.0)
    assert (rho_w, c_w) == pytest.approx((1000.0, 1500.0), abs=1e-3)
    assert (rho_b, c_b) == pytest.approx((1900.0, 2900.0), abs=1e-3)
    assert a_w == pytest.approx(0.5, abs=1e-3)     # soft-tissue floor
    assert a_b == pytest.approx(8.0, abs=1e-3)     # cortical bone

    # Porosity 1 (water) -> 0 (dense bone); properties monotone up in HU.
    hu = np.array([0, 500, 1000, 1500, 2000], float)
    rho, c, alpha = m.properties(hu)
    assert np.all(np.diff(m.porosity(hu)) < 0)
    for arr in (rho, c, alpha):
        assert np.all(np.diff(arr) > 0)

    # Above-cortical HU clamps (porosity floored at 0), doesn't extrapolate.
    rho_hi, c_hi, _ = m.properties(5000.0)
    assert (rho_hi, c_hi) == pytest.approx((1900.0, 2900.0), abs=1e-3)

    # __call__ is the slab-ramp (c, rho) adapter.
    c2, rho2 = m(2000.0)
    assert (c2, rho2) == pytest.approx((2900.0, 1900.0), abs=1e-3)


def test_placeholder_ramp_is_flagged():
    from tuba.core import hu_acoustics as ha
    ramp = ha.placeholder_ramp(8000, 32000, reason='unit test')
    assert getattr(ramp, 'placeholder', False) is True
    assert 'unit test' in ramp.placeholder_reason
    c_lo, rho_lo = ramp(0.0)          # below i_low -> water
    c_hi, rho_hi = ramp(32000.0)      # at i_high -> cortical bone
    assert (c_lo, rho_lo) == pytest.approx((1500.0, 1000.0), abs=1e-3)
    assert (c_hi, rho_hi) == pytest.approx((2900.0, 1900.0), abs=1e-3)


# ---------------------------------------------------------------------------
# 3. PPW + time-of-flight aberration + grid convergence (deliverable 1)
# ---------------------------------------------------------------------------
def test_points_per_wavelength():
    from tuba.core import hu_acoustics as ha
    rows = ha.points_per_wavelength(60e-6, (2e6, 3e6, 4e6), c_min_m_s=1500.0)
    ppw = {round(f / 1e6): p for f, _lam, p in rows}
    assert ppw[2] == pytest.approx(12.5, rel=1e-6)
    assert ppw[3] == pytest.approx(8.3333, rel=1e-4)
    assert ppw[4] == pytest.approx(6.25, rel=1e-6)


def test_tof_aberration_homogeneous_and_slab():
    from tuba.core import hu_acoustics as ha
    dx = 1e-4
    # Homogeneous medium at the reference speed -> zero aberration (the
    # analytic answer for a uniform medium).
    homog = np.full(50, 1500.0)
    assert ha.time_of_flight_aberration_s(homog, dx, c_ref_m_s=1500.0) == \
        pytest.approx(0.0, abs=1e-18)
    # A faster bone slab advances the wavefront -> negative aberration of
    # exactly n * (1/c_bone - 1/c_ref) * dx.
    prof = np.array([1500.0] * 10 + [2900.0] * 5 + [1500.0] * 10)
    tau = ha.time_of_flight_aberration_s(prof, dx, c_ref_m_s=1500.0)
    expect = 5 * (1 / 2900.0 - 1 / 1500.0) * dx
    assert tau == pytest.approx(expect, rel=1e-9)
    assert tau < 0


def test_grid_convergence_tof_smooth_profile():
    from tuba.core import hu_acoustics as ha
    # Smooth Gaussian sound-speed bump (a soft "skull") over 12 mm; the
    # time-of-flight aberration must converge as the pitch halves.
    def c_of_x(x):
        return 1500.0 + 1300.0 * np.exp(-((x - 6e-3) / 2e-3) ** 2)
    dxc = 100e-6
    dxf = 50e-6
    xc = np.arange(0, 12e-3, dxc)
    xf = np.arange(0, 12e-3, dxf)
    res = ha.grid_convergence_tof(c_of_x(xc), dxc, c_of_x(xf), dxf,
                                  verbose=False)
    assert res['tau_coarse_s'] < 0 and res['tau_fine_s'] < 0
    assert res['rel_change'] < 0.05          # well-resolved -> converged
    assert res['abs_change_ns'] >= 0


# ---------------------------------------------------------------------------
# 4. VALiDATe29 resolution + target mapping + affine round-trip (D2)
# ---------------------------------------------------------------------------
def _write_synthetic_atlas(tmp_path):
    """A synthetic VALiDATe29 mirroring the REAL layout: t1/t2/pd channels,
    a shipped brain mask, a region-level (l_/r_ separated) label volume,
    a space-delimited LUT with a section header, a decoy LICENSE.txt, and
    macOS ``__MACOSX``/``._`` cruft the resolver must ignore."""
    nib = pytest.importorskip('nibabel')
    aff = np.eye(4) * 0.3
    aff[3, 3] = 1.0
    aff[:3, 3] = [-15, -15, -15]
    tmpl = np.zeros((30, 30, 30), np.float32)
    tmpl[5:25, 5:25, 5:25] = 1.0
    labels = np.zeros((30, 30, 30), np.int16)
    labels[4:8, 12:18, 12:18] = 3      # l_primary_motor_cortex (M1)
    labels[22:26, 12:18, 12:18] = 4    # r_primary_motor_cortex (M1)
    labels[8:12, 12:18, 12:18] = 11    # l_anterior_parietal_cortex (S1/APC)
    labels[18:22, 12:18, 12:18] = 12   # r_anterior_parietal_cortex (S1/APC)
    mask = (labels > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(tmpl, aff), str(tmp_path / 'VALiDATe22-t2.nii.gz'))
    nib.save(nib.Nifti1Image(tmpl, aff), str(tmp_path / 'VALiDATe12-t1.nii.gz'))
    nib.save(nib.Nifti1Image(labels, aff),
             str(tmp_path / 'VALiDATe3-labels.nii.gz'))
    nib.save(nib.Nifti1Image(mask, aff),
             str(tmp_path / 'VALiDATe-brainmask.nii.gz'))
    with open(tmp_path / 'VALiDATe-labels.txt', 'w') as f:
        f.write('Gray Matter\n')
        f.write('3 l_primary_motor_cortex M1\n')
        f.write('4 r_primary_motor_cortex M1\n')
        f.write('11 l_anterior_parietal_cortex APC\n')
        f.write('12 r_anterior_parietal_cortex APC\n')
    with open(tmp_path / 'LICENSE.txt', 'w') as f:
        f.write('CC BY. This is not a label table.\n')
    junk = tmp_path / '__MACOSX'
    junk.mkdir()
    nib.save(nib.Nifti1Image(labels, aff),
             str(junk / '._VALiDATe3-labels.nii.gz'))   # must be ignored
    return tmp_path


def test_validate29_resolution_and_targets(tmp_path):
    _write_synthetic_atlas(tmp_path)
    from tuba.atlases.validate29 import VALiDATe29
    v = VALiDATe29(atlas_dir=str(tmp_path))            # default channel t2

    assert os.path.basename(v.template_path) == 'VALiDATe22-t2.nii.gz'
    assert os.path.basename(v.annotation_path) == 'VALiDATe3-labels.nii.gz'
    # label table must be the LUT, never LICENSE.txt
    assert os.path.basename(v.label_table_path) == 'VALiDATe-labels.txt'
    # shipped brain mask is preferred (no derive / no ANTs)
    assert os.path.basename(v.shipped_brain_mask_path) == 'VALiDATe-brainmask.nii.gz'
    assert v.derive_brain_mask(verbose=False) == v.shipped_brain_mask_path

    # region-level parcellation, hemisphere-separated ids
    assert v.resolve_label('S1', hemisphere='left') == ('l_anterior_parietal_cortex APC', 11)
    assert v.resolve_label('S1', hemisphere='right') == ('r_anterior_parietal_cortex APC', 12)
    assert v.resolve_label('M1', hemisphere='left') == ('l_primary_motor_cortex M1', 3)
    assert v.resolve_label('M1', hemisphere='right') == ('r_primary_motor_cortex M1', 4)
    # no hemisphere -> lowest matching id (left)
    assert v.resolve_label('M1')[1] == 3

    with pytest.raises(KeyError):
        v.resolve_label('V1')            # not in synthetic LUT -> helpful raise


def test_registration_affine_round_trip():
    """A forward affine composed with its inverse recovers the point cloud
    -- the RAS transform convention the atlas warp + target export rely on
    (synthetic; no ANTs)."""
    rng = np.random.default_rng(0)
    # rigid-ish affine: rotation about z by 20 deg + translation
    th = np.deg2rad(20.0)
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th),  np.cos(th), 0],
                  [0, 0, 1.0]])
    A = np.eye(4)
    A[:3, :3] = R * 1.3
    A[:3, 3] = [4, -7, 2]
    P = rng.normal(size=(50, 3))
    fwd = (A[:3, :3] @ P.T).T + A[:3, 3]
    Ainv = np.linalg.inv(A)
    back = (Ainv[:3, :3] @ fwd.T).T + Ainv[:3, 3]
    assert np.allclose(back, P, atol=1e-9)


# ---------------------------------------------------------------------------
# 5. Data manifest semantics
# ---------------------------------------------------------------------------
def test_manifest_saimiri_entries():
    from tuba.data import get, by_species
    keys = {s.key for s in by_species('saimiri')}
    assert keys == {'saimiri.skull', 'saimiri.atlas_template',
                    'saimiri.atlas_annotation'}
    skull = get('saimiri.skull')
    # Staged USNM 194346 is anisotropic (0.0977 in-plane x 0.1189 slice);
    # the manifest records the in-plane reconstruction pitch.
    assert skull.voxel_size_mm == pytest.approx(0.0977)
    assert 'uncalibrated' in (skull.notes or '').lower()
    assert skull.fetcher_script == 'src/tuba/data/fetch_saimiri.py'
    atlas = get('saimiri.atlas_template')
    assert atlas.license == 'CC BY'
    assert atlas.fetcher_script == 'src/tuba/data/fetch_validate29.py'


def test_saimiri_module_calibration_is_guarded():
    from tuba.species import saimiri as s
    assert s.CT_CALIBRATION.is_calibrated is False
    assert getattr(s.SLAB_RAMP, 'placeholder', False) is True
