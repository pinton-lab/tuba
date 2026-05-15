"""Regression test: TUBA mouse port vs. legacy mouse_therapy pipeline.

Exercises the read-only paths (cavity-centroid resolution, outer-bone
surface, placement geometry, slab sampler) against the existing cached
files in ``mouse_therapy/registration/``. Compares output dicts and
arrays for bit-identity.

Run from the TUBA project root:
    PYTHONPATH=src python tests/test_mouse_port_regression.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TUBA_SRC = os.path.join(HERE, '..', 'src')
# LEGACY_REG points at the original mouse_therapy/registration/ checkout
# that TUBA was ported from. Set ``TUBA_LEGACY_MOUSE_REG`` to its path
# to run the bit-identity regression; the test self-skips otherwise.
LEGACY_REG = os.environ.get('TUBA_LEGACY_MOUSE_REG', '')
sys.path.insert(0, TUBA_SRC)

if not LEGACY_REG or not os.path.isdir(LEGACY_REG):
    print('SKIP: set TUBA_LEGACY_MOUSE_REG to the legacy '
          'mouse_therapy/registration directory to run this regression.')
    sys.exit(0)
sys.path.insert(0, LEGACY_REG)

# Point the TUBA species module at the same legacy directory so it
# reads the warped Allen products and SyN transforms produced by
# the legacy pipeline. Set BEFORE importing tuba.species.mouse.
os.environ.setdefault('TUBA_MOUSE_REG_DIR', LEGACY_REG)

from tuba.species import mouse as tuba_mouse
import maga_placement as legacy_placement
import maga_skull_slab_3d as legacy_slab


def _compare_placements(label, tuba_p, legacy_p):
    ok = True
    keys = sorted(set(tuba_p) | set(legacy_p))
    for k in keys:
        a, b = tuba_p.get(k), legacy_p.get(k)
        same = (a == b)
        if isinstance(a, tuple) and isinstance(b, tuple):
            same = all(abs(x - y) < 1e-12 for x, y in zip(a, b))
        elif isinstance(a, float) and isinstance(b, float):
            same = abs(a - b) < 1e-12
        marker = 'OK' if same else 'DIFF'
        print(f'  [{marker}] {k!r:35s}  tuba={a}  legacy={b}')
        if not same:
            ok = False
    print(f'  -> {label}: {"PASS" if ok else "FAIL"}')
    return ok


def test_placement_cavity_centre():
    print('\n=== test_placement_cavity_centre ===')
    tuba_p = tuba_mouse.place_on_skull(
        target_name='cavity_centre', apex_to_target_mm=20.0,
        beam_tilt_deg_yz=20.0, verbose=False)
    legacy_p = legacy_placement.place_on_maga_skull(
        target_name='cavity_centre', apex_to_target_mm=20.0,
        beam_tilt_deg_yz=20.0, debug=False)
    return _compare_placements('placement(cavity_centre)', tuba_p, legacy_p)


def test_placement_cc_left_body():
    print('\n=== test_placement_cc_left_body ===')
    tuba_p = tuba_mouse.place_on_skull(
        target_name='CC_left_body', apex_to_target_mm=10.0,
        beam_tilt_deg_yz=0.0, verbose=False)
    legacy_p = legacy_placement.place_on_maga_skull(
        target_name='CC_left_body', apex_to_target_mm=10.0,
        beam_tilt_deg_yz=0.0, debug=False)
    return _compare_placements('placement(CC_left_body)', tuba_p, legacy_p)


def test_slab():
    print('\n=== test_slab (cavity_centre, 20mm, 20deg tilt) ===')
    p = legacy_placement.place_on_maga_skull(
        target_name='cavity_centre', apex_to_target_mm=20.0,
        beam_tilt_deg_yz=20.0, debug=False)

    apex = p['xdc_center_lps']
    beam = p['beam_dir_3d']
    slab_size_m = (12e-3, 12e-3, 15e-3)
    dx_target_m = 1540.0 / 15e6 / 3.5   # mouse_cc default

    print('  running TUBA slab loader...')
    c_t, rho_t, dz_t, frame_t = tuba_mouse.load_slab(
        apex, beam, slab_size_m, dx_target_m, chatter=False)
    print('  running legacy slab loader...')
    c_l, rho_l, dz_l, frame_l = legacy_slab.load_maga_skull_slab_3d(
        apex, beam, slab_size_m, dx_target_m, chatter=False)

    print(f'  c_map shape: tuba={c_t.shape}, legacy={c_l.shape}')
    print(f'  dz: tuba={dz_t:.6e}, legacy={dz_l:.6e}, '
          f'diff={abs(dz_t-dz_l):.2e}')

    same_c = np.array_equal(c_t, c_l)
    same_rho = np.array_equal(rho_t, rho_l)
    if not same_c:
        diff = np.abs(c_t.astype(np.float64) - c_l.astype(np.float64))
        print(f'  c_map NOT bit-identical: max|d|={diff.max():.3f}, '
              f'mean|d|={diff.mean():.6f}')
    else:
        print('  c_map BIT-IDENTICAL')
    if not same_rho:
        diff = np.abs(rho_t.astype(np.float64) - rho_l.astype(np.float64))
        print(f'  rho_map NOT bit-identical: max|d|={diff.max():.3f}, '
              f'mean|d|={diff.mean():.6f}')
    else:
        print('  rho_map BIT-IDENTICAL')

    same_frame = (np.array_equal(frame_t['beam'], frame_l['beam'])
                  and np.array_equal(frame_t['t1'], frame_l['t1'])
                  and np.array_equal(frame_t['t2'], frame_l['t2'])
                  and np.array_equal(frame_t['apex_mm'], frame_l['apex_mm']))
    print(f'  frame bit-identical: {same_frame}')

    ok = same_c and same_rho and same_frame and (dz_t == dz_l)
    print(f'  -> slab: {"PASS" if ok else "FAIL"}')
    return ok


def test_cavity_centroid():
    print('\n=== test_cavity_centroid ===')
    from tuba.core.cavity import cavity_centroid_world_mm
    from maga_placement import _brain_centre_world_mm
    t = cavity_centroid_world_mm(tuba_mouse.CAVITY_NII)
    l = _brain_centre_world_mm()
    diff = np.abs(t - l)
    print(f'  tuba   = {t}')
    print(f'  legacy = {l}')
    print(f'  max|d| = {diff.max():.3e}')
    ok = diff.max() < 1e-12
    print(f'  -> cavity_centroid: {"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    results = [
        test_cavity_centroid(),
        test_placement_cavity_centre(),
        test_placement_cc_left_body(),
        test_slab(),
    ]
    print('\n========================================')
    print(f'OVERALL: {sum(results)}/{len(results)} tests passed')
    print('========================================')
    sys.exit(0 if all(results) else 1)
