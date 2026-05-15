"""Regression test: TUBA human port vs. legacy neuromod_parameters pipeline.

Compares ``tuba.species.human`` against
``neuromod_parameters/registration/halle_placement.py`` +
``skull_slab_3d.py``. Accounts for the coordinate-frame migration:

* Legacy returns coords in NRRD-voxel-mm (+L, +A, +S, origin at NRRD
  corner). TUBA returns coords in Halle-RAS world mm (-i*vox, +j*vox,
  +k*vox). They differ by a sign flip on the first component.
* The slab loader picks t1 via ``cross(beam, helper)``; the cross
  product is a pseudo-vector under the LAS<->RAS reflection, so the
  TUBA slab is the *physical mirror* of the legacy slab along the
  lateral (t1) axis: ``c_map_TUBA[i_lat, ...] == c_map_legacy[
  n_lat-1-i_lat, ...]``. We assert this mirrored bit-equality.

Run from the TUBA project root:
    PYTHONPATH=src python tests/test_human_port_regression.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TUBA_SRC = os.path.join(HERE, '..', 'src')
# LEGACY_REG points at the original neuromod_parameters/registration/
# checkout that TUBA was ported from. Set ``TUBA_LEGACY_HUMAN_REG``
# to its path to run the bit-identity regression; self-skips otherwise.
LEGACY_REG = os.environ.get('TUBA_LEGACY_HUMAN_REG', '')
sys.path.insert(0, TUBA_SRC)

if not LEGACY_REG or not os.path.isdir(LEGACY_REG):
    print('SKIP: set TUBA_LEGACY_HUMAN_REG to the legacy '
          'neuromod_parameters/registration directory to run this regression.')
    sys.exit(0)
sys.path.insert(0, LEGACY_REG)

# Point the TUBA species module at the same legacy directory so the
# warped MNI products and saved ANTs transforms it reads are the ones
# the bit-identity test is comparing against. Set the env vars BEFORE
# importing tuba.species.human so module-level constants resolve to
# the right paths. ``TUBA_HUMAN_NRRD_PATH`` must be exported by the
# caller; the regression is intentionally non-portable and skips here
# rather than hard-coding any one lab's filesystem layout.
os.environ.setdefault('TUBA_HUMAN_REG_DIR', LEGACY_REG)
if not os.environ.get('TUBA_HUMAN_NRRD_PATH'):
    print('SKIP: set TUBA_HUMAN_NRRD_PATH to the Halle skull NRRD '
          '(or run tuba.data.fetch_human to stage it).')
    sys.exit(0)

from tuba.species import human as tuba_human
import halle_placement as legacy_placement
import skull_slab_3d as legacy_slab


def _close(a, b, atol):
    return float(abs(a - b)) <= atol


def _close_vec(a, b, atol):
    a = np.asarray(a); b = np.asarray(b)
    return bool(np.all(np.abs(a - b) <= atol))


def test_coord_helpers():
    print('\n=== test_coord_helpers ===')
    pt_nrrd = (100.0, 50.0, 75.0)
    pt_ras = tuba_human.nrrd_voxel_mm_to_halle_ras(pt_nrrd)
    pt_back = tuba_human.halle_ras_to_nrrd_voxel_mm(pt_ras)
    ok = _close_vec(pt_back, pt_nrrd, 1e-12)
    print(f'  NRRD -> RAS -> NRRD round-trip: {pt_nrrd} -> {tuple(pt_ras)} -> {tuple(pt_back)}: '
          f'{"PASS" if ok else "FAIL"}')
    return ok


def test_placement_S1_left_CP3():
    print('\n=== test_placement_S1_left_CP3 ===')
    legacy_p = legacy_placement.place_on_skull(
        scalp_site='CP3', target_name='S1_left',
        eeg_search_radius_mm=10.0,
        focal_length_mm=30.0, bowl_radius_mm=15.0)
    print(f'  legacy apex (NRRD-vox-mm): '
          f'{tuple(round(v,3) for v in legacy_p["xdc_center_lps"])}')
    print(f'  legacy target (NRRD-vox-mm): '
          f'{tuple(round(v,3) for v in legacy_p["target_lps"])}')
    print(f'  legacy beam_dir (NRRD): '
          f'{tuple(round(v,4) for v in legacy_p["beam_dir_3d"])}')

    tuba_p = tuba_human.place_on_skull(
        scalp_site='CP3', target_name='S1_left',
        eeg_search_radius_mm=10.0,
        focal_length_mm=30.0, bowl_radius_mm=15.0,
        verbose=False)
    print(f'  TUBA apex (Halle RAS): '
          f'{tuple(round(v,3) for v in tuba_p["xdc_center_lps"])}')
    print(f'  TUBA apex (back-compat NRRD): '
          f'{tuple(round(v,3) for v in tuba_p["xdc_center_lps_nrrd_voxel_mm"])}')

    # Compare TUBA's NRRD-back-compat coords vs legacy. Allowed: 1 um
    # rounding from the LPS<->NRRD<->RAS chain (all done as Python
    # floats, identical ops, expected bit-identical).
    atol = 1e-9
    checks = []
    for key in ('xdc_center_lps', 'target_lps', 'focus_lps',
                'scalp_contact_lps'):
        if legacy_p.get(key) is None:
            continue
        tuba_nrrd = tuba_p[f'{key}_nrrd_voxel_mm']
        ok = _close_vec(tuba_nrrd, legacy_p[key], atol)
        diff_max = float(np.max(np.abs(
            np.asarray(tuba_nrrd) - np.asarray(legacy_p[key]))))
        marker = 'OK' if ok else 'DIFF'
        print(f'  [{marker}] {key:25s} max|d|={diff_max:.3e}')
        checks.append(ok)

    # Beam direction in NRRD frame: flip x of TUBA's RAS beam.
    tuba_beam_nrrd = (-tuba_p['beam_dir_3d'][0], tuba_p['beam_dir_3d'][1],
                      tuba_p['beam_dir_3d'][2])
    diff_max = float(np.max(np.abs(
        np.asarray(tuba_beam_nrrd) - np.asarray(legacy_p['beam_dir_3d']))))
    ok = diff_max <= atol
    print(f'  [{"OK" if ok else "DIFF"}] beam_dir_3d              max|d|={diff_max:.3e}')
    checks.append(ok)

    # Scalar fields
    for key in ('scalp_target_dist_mm', 'apex_target_dist_mm',
                'perp_residual_mm', 'bowl_depth_mm',
                'focal_length_mm', 'bowl_radius_mm',
                'eeg_to_placement_mm'):
        if legacy_p.get(key) is None or tuba_p.get(key) is None:
            continue
        ok = _close(tuba_p[key], legacy_p[key], atol)
        marker = 'OK' if ok else 'DIFF'
        print(f'  [{marker}] {key:25s} tuba={tuba_p[key]:.6f}  '
              f'legacy={legacy_p[key]:.6f}')
        checks.append(ok)
    overall = all(checks)
    print(f'  -> placement: {"PASS" if overall else "FAIL"}')
    return overall


def test_slab():
    print('\n=== test_slab (S1_left, CP3) ===')
    legacy_p = legacy_placement.place_on_skull(
        scalp_site='CP3', target_name='S1_left',
        eeg_search_radius_mm=10.0,
        focal_length_mm=30.0, bowl_radius_mm=15.0)
    tuba_p = tuba_human.place_on_skull(
        scalp_site='CP3', target_name='S1_left',
        eeg_search_radius_mm=10.0,
        focal_length_mm=30.0, bowl_radius_mm=15.0,
        verbose=False)

    apex_nrrd = legacy_p['xdc_center_lps']
    beam_nrrd = legacy_p['beam_dir_3d']
    apex_ras = tuba_p['xdc_center_lps']
    beam_ras = tuba_p['beam_dir_3d']

    slab_size_m = (40e-3, 40e-3, 80e-3)
    dx_target_m = 0.5e-3   # 0.5 mm sampling -- coarse to keep the test fast

    print('  running legacy slab loader (NRRD-voxel-mm)...')
    c_l, rho_l, dz_l, frame_l = legacy_slab.load_skull_slab_3d(
        nrrd_path=tuba_human.NRRD_PATH,
        apex_lps_mm=apex_nrrd, beam_3d=beam_nrrd,
        slab_size_m=slab_size_m, dx_target_m=dx_target_m)
    print('  running TUBA slab loader (Halle RAS)...')
    c_t, rho_t, dz_t, frame_t = tuba_human.load_slab(
        apex_world_mm=apex_ras, beam_3d=beam_ras,
        slab_size_m=slab_size_m, dx_target_m=dx_target_m, chatter=False)

    print(f'  shapes: legacy={c_l.shape}, tuba={c_t.shape}')
    print(f'  dz: legacy={dz_l:.6e}, tuba={dz_t:.6e}, '
          f'diff={abs(dz_l-dz_t):.2e}')

    # Direct equality should fail (axes differ): check that.
    direct_eq = np.array_equal(c_l, c_t)
    print(f'  direct c_map equality (expected FAIL): {direct_eq}')

    # Mirrored equality is what we expect: TUBA[i_lat, j, k] ==
    # legacy[n_lat-1-i_lat, j, k] because t1 is the physical mirror.
    c_t_mirrored = c_t[::-1, :, :]
    rho_t_mirrored = rho_t[::-1, :, :]
    diff_c = np.abs(c_l.astype(np.float64) - c_t_mirrored.astype(np.float64))
    diff_rho = np.abs(rho_l.astype(np.float64) - rho_t_mirrored.astype(np.float64))
    print(f'  mirrored c_map:   max|d|={diff_c.max():.3e}, '
          f'mean|d|={diff_c.mean():.6e}')
    print(f'  mirrored rho_map: max|d|={diff_rho.max():.3e}, '
          f'mean|d|={diff_rho.mean():.6e}')

    # Allow tiny floating-point slop from cross-product reordering.
    tol = 1e-3
    ok = (diff_c.max() <= tol and diff_rho.max() <= tol and dz_l == dz_t)
    print(f'  -> slab (mirrored): {"PASS" if ok else "FAIL"}')

    # Beam should match (modulo sign flip on x).
    beam_l = np.asarray(frame_l['beam'])
    beam_t = np.asarray(frame_t['beam'])
    beam_l_to_ras = np.array([-beam_l[0], beam_l[1], beam_l[2]])
    beam_match = _close_vec(beam_l_to_ras, beam_t, 1e-12)
    print(f'  frame beam match (after sign-flip x): {beam_match}')
    return ok and beam_match


if __name__ == '__main__':
    results = [
        test_coord_helpers(),
        test_placement_S1_left_CP3(),
        test_slab(),
    ]
    print('\n========================================')
    print(f'OVERALL: {sum(results)}/{len(results)} tests passed')
    print('========================================')
    sys.exit(0 if all(results) else 1)
