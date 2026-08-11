"""Offline unit tests for ``tuba.mini`` (no network, no antspyx).

Covers the three things that must hold regardless of whether the heavy
data/deps are present:

1. importing the mini modules must NOT import ``antspyx`` (the ANTs-free
   consumer contract),
2. the STL rasterizer produces the right bone/cavity masks + RAS affine,
3. the QC helpers (point marker, containment metric) are correct.
"""
import os
import subprocess
import sys

import numpy as np
import pytest


def test_import_does_not_pull_in_antspyx():
    """`import tuba.mini.*` must stay ANTs-free (ANTs is imported lazily
    inside tuba.core.warp only when a registration/warp actually runs).
    Checked in a fresh interpreter so a prior import can't mask it."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    code = (
        'import sys; '
        'import tuba.mini, tuba.mini.fetch, tuba.mini.skull, '
        'tuba.mini.register, tuba.mini.itrusst; '
        'assert "ants" not in sys.modules, sorted(m for m in sys.modules '
        'if "ant" in m.lower()); '
        'print("ok")')
    r = subprocess.run([sys.executable, '-c', code], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert 'ok' in r.stdout


def test_rasterize_nested_boxes(tmp_path):
    """Two concentric watertight boxes -> a shell (bone) enclosing a solid
    core (cavity), on a 1 mm RAS grid."""
    trimesh = pytest.importorskip('trimesh')
    from tuba.mini import skull

    outer = trimesh.creation.box(extents=(40.0, 40.0, 40.0))
    inner = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    op = str(tmp_path / 'outer.stl')
    ip = str(tmp_path / 'inner.stl')
    outer.export(op)
    inner.export(ip)

    m = skull.rasterize(ip, op, pitch_mm=1.0, margin_mm=4.0, verbose=False)

    # A box of side L rasterised at 1 mm fills L+1 voxels per side, so the
    # 20 mm core is ~21^3 = 9261 and the shell ~41^3 - 21^3 = 59660. Assert
    # structural ranges rather than over-fitting trimesh's boundary phase.
    assert 8000 <= m['cavity'].sum() <= 9800
    assert 55000 <= m['bone'].sum() <= 62000
    # bone and cavity are disjoint by construction
    assert not np.any(m['bone'] & m['cavity'])
    # RAS diagonal affine, translation at the padded grid corner (-24)
    assert np.allclose(np.diag(m['affine'])[:3], 1.0)
    assert np.allclose(m['affine'][:3, 3], -24.0)
    # cavity centroid back at the origin
    cc = np.argwhere(m['cavity']).mean(0)
    world = m['affine'][:3, :3] @ cc + m['affine'][:3, 3]
    assert np.allclose(world, 0.0, atol=1.0)


def test_point_marker_centroid_roundtrips(tmp_path):
    """A marker written at an RAS-mm coordinate has its centroid back at
    that coordinate."""
    nib = pytest.importorskip('nibabel')
    from tuba.mini import register

    # 1 mm reference volume spanning a generous FOV, RAS affine.
    ref = np.zeros((60, 60, 60), np.float32)
    aff = np.eye(4)
    aff[:3, 3] = [-30, -30, -30]
    ref_path = str(tmp_path / 'ref.nii.gz')
    nib.save(nib.Nifti1Image(ref, aff), ref_path)

    coord = (10.0, -5.0, 3.0)
    out = str(tmp_path / 'mk.nii.gz')
    register.mni_point_marker(coord, ref_path, out, radius_mm=4.0,
                              verbose=False)
    img = nib.load(out)
    vox = np.argwhere(np.asarray(img.dataobj) > 0)
    assert len(vox) > 0
    world = nib.affines.apply_affine(img.affine, vox.mean(0))
    assert np.allclose(world, coord, atol=1.0)


def test_place_perpendicular_on_hemisphere():
    """place() on a synthetic upper-hemisphere skull: the apex sits above
    the dome, the beam points inward (downward) at a sub-apex target, and
    the perpendicular residual is ~0. Exercises the tuba.core.placement
    wiring without fetching the STL."""
    from tuba.mini import itrusst

    # upper hemisphere, radius 70 mm, outward (radial) normals
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0.02, np.pi / 2, 60)
    uu, vv = np.meshgrid(u, v)
    R = 70.0
    verts = np.stack([
        R * np.cos(uu) * np.sin(vv),
        R * np.sin(uu) * np.sin(vv),
        R * np.cos(vv)], axis=-1).reshape(-1, 3)
    normals = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    pl = itrusst.place(target_ras_mm=(0.0, 0.0, 20.0),
                       surface=(verts, normals), focal_length_mm=64.0,
                       bowl_radius_mm=32.0, verbose=False)

    assert pl['beam_dir_3d'][2] < -0.9          # beam points inward/down
    assert pl['perp_residual_mm'] < 2.0         # target under the dome apex
    assert pl['scalp_contact_lps'][2] > 60.0    # contact near the dome top
    # apex sits outside the bone, above the scalp contact
    assert pl['xdc_center_lps'][2] > pl['scalp_contact_lps'][2]


def test_containment_metrics():
    """Perfect containment -> frac 1.0; identical masks -> dice 1.0."""
    import nibabel as nib
    from tuba.mini import demo
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        aff = np.eye(4)
        cavity = np.zeros((20, 20, 20), np.uint8)
        cavity[4:16, 4:16, 4:16] = 1
        brain = np.zeros((20, 20, 20), np.uint8)
        brain[6:14, 6:14, 6:14] = 1
        cp = os.path.join(d, 'c.nii.gz')
        bp = os.path.join(d, 'b.nii.gz')
        nib.save(nib.Nifti1Image(cavity, aff), cp)
        nib.save(nib.Nifti1Image(brain, aff), bp)
        m = demo.containment_metrics(bp, cp)
        assert m['contained_frac'] == pytest.approx(1.0)   # brain fully inside
        m2 = demo.containment_metrics(cp, cp)
        assert m2['dice'] == pytest.approx(1.0)            # identical -> dice 1
