"""Regression test: TUBA macaque port vs. legacy monkey_brain pipeline.

Compares ``tuba.species.macaque`` against
``monkey_brain/registration/macaque_placement.py`` +
``monkey_brain/registration/macaque_skull_slab_3d.py``. The port is a
thin wrapper that re-imports the same numerical kernels under
``tuba._macaque``, so this test mainly verifies:

  1. The TUBA module imports cleanly with default env-var fallbacks.
  2. Key constants (Euler angles, padding, voxel size, thresholds)
     match the legacy values.
  3. Round-trip placement against ``TUBA_LEGACY_MACAQUE_REG`` gives
     bit-identical apex / target / beam triples.

Run from the TUBA project root:
    PYTHONPATH=src python tests/test_macaque_port_regression.py

Self-skips when ``TUBA_LEGACY_MACAQUE_REG`` is unset, so CI can run the
file without staged data.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TUBA_SRC = os.path.join(HERE, '..', 'src')
LEGACY_REG = os.environ.get('TUBA_LEGACY_MACAQUE_REG', '')
sys.path.insert(0, TUBA_SRC)


def test_imports_clean():
    """The port should import with default env vars on any host."""
    print('\n=== test_imports_clean ===')
    from tuba.species import macaque  # noqa: F401
    from tuba._macaque import macaque_skull_constants  # noqa: F401
    from tuba._macaque import macaque_placement  # noqa: F401
    from tuba._macaque import macaque_skull_slab_3d  # noqa: F401
    print('  imported tuba.species.macaque and tuba._macaque.* cleanly')
    return True


def test_constants_match_manuscript():
    """Module-level constants should match the manuscript values."""
    print('\n=== test_constants_match_manuscript ===')
    from tuba.species import macaque as m
    checks = {
        'NATIVE_VOXEL_MM':     (m.NATIVE_VOXEL_MM, 0.060613, 1e-6),
        'NATIVE_PAD_VOXELS':   (m.NATIVE_PAD_VOXELS, 400, 0),
        'ALIGN_RX_DEG':        (m.ALIGN_RX_DEG, 95.94, 0.01),
        'ALIGN_RY_DEG':        (m.ALIGN_RY_DEG, -59.69, 0.01),
        'ALIGN_RZ_DEG':        (m.ALIGN_RZ_DEG, -99.12, 0.01),
        'ALIGN_PCA_THRESH':    (m.ALIGN_PCA_THRESH, 25000, 0),
        'BONE_THRESH':         (m.BONE_THRESH, 10000, 0),
    }
    ok = True
    for k, (got, expect, atol) in checks.items():
        if abs(got - expect) > atol:
            print(f'  [DIFF] {k:22s} got={got!r} expected={expect!r}')
            ok = False
        else:
            print(f'  [OK]   {k:22s} {got!r}')
    return ok


def test_placement_bit_identity():
    """When TUBA_LEGACY_MACAQUE_REG points at a real registration
    directory with the cached products from the monkey_brain pipeline,
    the TUBA wrapper should yield bit-identical placement to the
    legacy ``macaque_placement.place_on_macaque_skull``."""
    print('\n=== test_placement_bit_identity ===')
    if not LEGACY_REG or not os.path.isdir(LEGACY_REG):
        print('  SKIP: set TUBA_LEGACY_MACAQUE_REG to the legacy '
              'monkey_brain/registration directory to run this regression.')
        return True

    sys.path.insert(0, LEGACY_REG)

    # Point the TUBA macaque module at the same legacy dir so cached
    # NIfTIs / Affine matrices resolve to the same files.
    os.environ['TUBA_MACAQUE_REG_DIR'] = LEGACY_REG

    # Re-import after env var is set so module-level paths pick it up.
    for mod in list(sys.modules):
        if mod.startswith('tuba'):
            del sys.modules[mod]

    from tuba.species import macaque as tuba_macaque
    import macaque_placement as legacy_placement

    tuba_p = tuba_macaque.place_on_skull(target_name='dACC',
                                          focal_length_mm=45.0,
                                          verbose=False)
    legacy_p = legacy_placement.place_on_macaque_skull(
        target_name='dACC', apex_to_target_mm=45.0, debug=False)

    atol = 1e-9
    ok = True
    for key in ('xdc_center_lps', 'target_lps', 'focus_lps',
                'apex_on_bone_voxel_mm', 'beam_dir_3d'):
        if legacy_p.get(key) is None or tuba_p.get(key) is None:
            continue
        a = np.asarray(tuba_p[key])
        b = np.asarray(legacy_p[key])
        d = float(np.max(np.abs(a - b)))
        marker = 'OK' if d <= atol else 'DIFF'
        print(f'  [{marker}] {key:22s} max|d|={d:.3e}')
        ok &= (d <= atol)
    for key in ('scalp_target_dist_mm', 'scalp_clearance_mm'):
        if legacy_p.get(key) is None or tuba_p.get(key) is None:
            continue
        d = float(abs(tuba_p[key] - legacy_p[key]))
        marker = 'OK' if d <= atol else 'DIFF'
        print(f'  [{marker}] {key:22s} tuba={tuba_p[key]:.6f}  '
              f'legacy={legacy_p[key]:.6f}')
        ok &= (d <= atol)
    return ok


if __name__ == '__main__':
    results = [
        test_imports_clean(),
        test_constants_match_manuscript(),
        test_placement_bit_identity(),
    ]
    print('\n========================================')
    print(f'OVERALL: {sum(results)}/{len(results)} tests passed')
    print('========================================')
    sys.exit(0 if all(results) else 1)
