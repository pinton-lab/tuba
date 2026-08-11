"""End-to-end ``tuba.mini`` demo + QC.

Builds the ITRUSST-skull-with-atlas (fetch -> rasterize -> SyN -> warp)
and verifies the atlasing landed correctly:

* the warped MNI brain mask is *contained* in the endocranial cavity,
* named MNI targets (S1, thalamus, ...) fall in anatomically sensible
  skull-space locations,

then writes a three-panel orthoslice QC figure (bone shell + warped brain
mask + target markers) under ``$TUBA_MINI_DIR/qc``.

Run:  ``python -m tuba.mini.demo``
"""
import os
import numpy as np

from tuba.mini import itrusst, register


TARGETS = ('S1_left', 'thalamus_central', 'cerebellum_central')

# Placement demo: focused bowl on M1, apex search seeded at EEG C3.
PLACE_TARGET = 'M1_left'
PLACE_SCALP = 'C3'


def _load(path):
    import nibabel as nib
    img = nib.load(path)
    return np.asarray(img.dataobj), img.affine


def containment_metrics(brain_in_skull_path, cavity_path):
    """How well the warped MNI brain mask sits inside the skull cavity."""
    brain, _ = _load(brain_in_skull_path)
    cavity, _ = _load(cavity_path)
    brain = brain > 0
    cavity = cavity > 0
    inter = np.logical_and(brain, cavity).sum()
    contained = inter / max(brain.sum(), 1)          # brain fraction inside cavity
    dice = 2 * inter / max(brain.sum() + cavity.sum(), 1)
    return {'brain_vox': int(brain.sum()), 'cavity_vox': int(cavity.sum()),
            'contained_frac': float(contained), 'dice': float(dice)}


def target_centroids(paths, mni_mask_path, targets=TARGETS, verbose=True):
    """Warp a marker for each named MNI target into skull space and
    report its skull-grid centroid (RAS mm). Robust to transform
    direction -- rides the same image-warp path as the atlas."""
    import nibabel as nib
    out = {}
    for name in targets:
        coord = itrusst.ANATOMICAL_TARGETS_MNI.get(name)
        if coord is None:
            continue
        mk_mni = os.path.join(paths['reg_dir'], f'_marker_{name}_mni.nii.gz')
        mk_sk = os.path.join(paths['reg_dir'], f'_marker_{name}_skull.nii.gz')
        register.mni_point_marker(coord, mni_mask_path, mk_mni,
                                  radius_mm=5.0, verbose=False)
        register.warp_into_skull(mk_mni, paths['bone'], paths['reg_dir'],
                                 mk_sk, interpolator='nearestNeighbor',
                                 verbose=False)
        arr, aff = _load(mk_sk)
        vox = np.argwhere(arr > 0)
        if len(vox):
            world = nib.affines.apply_affine(aff, vox.mean(0))
        else:
            world = np.array([np.nan] * 3)
        out[name] = {'mni_ras': tuple(float(v) for v in coord),
                     'skull_ras': tuple(round(float(v), 1) for v in world),
                     'n_vox': int((arr > 0).sum())}
        if verbose:
            print(f'    {name:20s} MNI {coord} -> skull '
                  f'{out[name]["skull_ras"]} ({out[name]["n_vox"]} vox)')
    return out


def place_and_slab(target=PLACE_TARGET, scalp_site=PLACE_SCALP, verbose=True):
    """Place a focused bowl perpendicular to the outer skull aimed at
    ``target`` (apex search seeded at EEG ``scalp_site``) and extract the
    beam-aligned acoustic slab. Returns ``(placement, c_map, rho_map, dz)``."""
    pl = itrusst.place(target_name=target, scalp_site=scalp_site,
                       verbose=verbose)
    if verbose:
        print(f"    scalp contact = "
              f"{tuple(round(v, 1) for v in pl['scalp_contact_lps'])}")
        print(f"    apex          = "
              f"{tuple(round(v, 1) for v in pl['xdc_center_lps'])}")
        print(f"    beam (inward) = "
              f"{tuple(round(v, 3) for v in pl['beam_dir_3d'])}")
        print(f"    focus         = "
              f"{tuple(round(v, 1) for v in pl['focus_lps'])}")
        print(f"    perp residual = {pl['perp_residual_mm']:.2f} mm;  "
              f"focus->target = {pl['focus_to_target_mm']:.2f} mm")
    c_map, rho_map, dz, _ = itrusst.load_slab(pl, verbose=verbose)
    return pl, c_map, rho_map, dz


def placement_figure(paths, pl, c_map, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bone, aff = _load(paths['bone'])
    bone = bone > 0
    inv = np.linalg.inv(aff)

    def to_vox(world):
        return inv[:3, :3] @ np.asarray(world) + inv[:3, 3]

    apex = np.asarray(pl['xdc_center_lps'])
    focus = np.asarray(pl['focus_lps'])
    target = np.asarray(pl['target_lps'])
    scalp = np.asarray(pl['scalp_contact_lps'])
    vt = to_vox(target)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    # Panel 1: sagittal skull at the target's x-plane + placement geometry
    ax = axes[0]
    i0 = int(round(vt[0]))
    i0 = min(max(i0, 0), bone.shape[0] - 1)
    ax.imshow(bone[i0].T, origin='lower', cmap='gray')
    pts = {'target': (target, 'yellow'), 'apex': (apex, 'cyan'),
           'scalp': (scalp, 'lime'), 'focus': (focus, 'magenta')}
    for name, (w, col) in pts.items():
        v = to_vox(w)
        ax.plot(v[1], v[2], 'o', ms=8, mfc='none', mec=col, mew=2)
        ax.annotate(name, (v[1], v[2]), color=col, fontsize=8,
                    xytext=(4, 4), textcoords='offset points')
    va, vf = to_vox(apex), to_vox(focus)
    ax.plot([va[1], vf[1]], [va[2], vf[2]], '-', color='cyan', lw=1.2,
            alpha=0.8)
    ax.set_title(f"placement: bowl on {pl['target_name']} "
                 f"(F={pl['focal_length_mm']:.0f} R={pl['bowl_radius_mm']:.0f} mm)")
    ax.set_xlabel('A-P (vox)')
    ax.set_ylabel('S-I (vox)')

    # Panel 2: beam-plane sound-speed slab (central elevation slice)
    ax = axes[1]
    n_lat, n_elev, n_depth = c_map.shape
    c_slice = c_map[:, n_elev // 2, :]           # (lat, depth)
    depth_mm = itrusst.SLAB_SIZE_M[2] * 1e3
    lat_mm = itrusst.SLAB_SIZE_M[0] * 1e3
    im = ax.imshow(c_slice, origin='lower', aspect='auto', cmap='inferno',
                   extent=[0, depth_mm, -lat_mm / 2, lat_mm / 2])
    ax.axvline(pl['focal_length_mm'], color='cyan', ls='--', lw=1,
               label='geometric focus')
    ax.set_title('acoustic slab: sound speed in the beam plane')
    ax.set_xlabel('depth along beam from apex (mm)')
    ax.set_ylabel('lateral (mm)')
    ax.legend(loc='upper right', fontsize=8)
    fig.colorbar(im, ax=ax, label='c (m/s)', fraction=0.046, pad=0.04)

    fig.suptitle('tuba.mini — bowl placement + acoustic slab on the '
                 'ITRUSST skull', fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=100)
    print(f'  wrote {out_png}')
    return out_png


def qc_figure(paths, centroids, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bone, aff = _load(paths['bone'])
    brain, _ = _load(paths['brain_in_skull'])
    bone = bone > 0
    brain = brain > 0

    inv = np.linalg.inv(aff)
    # centre the panels on the warped-brain centroid
    c = np.argwhere(brain).mean(0).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    planes = [
        ('sagittal', lambda v: (v[0], slice(None), slice(None))),
        ('coronal',  lambda v: (slice(None), v[1], slice(None))),
        ('axial',    lambda v: (slice(None), slice(None), v[2])),
    ]
    for ax, (title, sel) in zip(axes, planes):
        b = bone[sel(c)]
        m = brain[sel(c)]
        ax.imshow(b.T, origin='lower', cmap='gray')
        ax.imshow(np.ma.masked_where(~m.T, m.T), origin='lower',
                  cmap='cool', alpha=0.45)
        # target markers projected into this plane
        for name, d in centroids.items():
            if any(np.isnan(d['skull_ras'])):
                continue
            vx = inv[:3, :3] @ np.array(d['skull_ras']) + inv[:3, 3]
            if title == 'sagittal':
                xy = (vx[1], vx[2])
            elif title == 'coronal':
                xy = (vx[0], vx[2])
            else:
                xy = (vx[0], vx[1])
            ax.plot(*xy, 'o', ms=7, mfc='none', mec='yellow', mew=2)
            ax.annotate(name.replace('_', ' '), xy, color='yellow',
                        fontsize=7, xytext=(4, 4),
                        textcoords='offset points')
        ax.set_title(f'{title}  (bone gray, MNI brain cyan)')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('tuba.mini — MNI152 atlas warped into the ITRUSST '
                 'benchmark skull', fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=100)
    print(f'  wrote {out_png}')
    return out_png


def main(parcellation='harvard_oxford_117'):
    print('=== tuba.mini demo ===')
    p = itrusst.build(parcellation=parcellation, verbose=True)
    mp = itrusst.mni_paths()

    print('\n[QC] brain-mask containment in cavity:')
    m = containment_metrics(p['brain_in_skull'], p['cavity'])
    print(f'    brain={m["brain_vox"]} vox  cavity={m["cavity_vox"]} vox')
    print(f'    contained_frac={m["contained_frac"]:.3f}  dice={m["dice"]:.3f}')

    print('\n[QC] named MNI targets in skull space:')
    cents = target_centroids(p, mp['mask'], verbose=True)

    out_png = os.path.join(itrusst.fetch.mini_dir(), 'qc',
                           'mini_atlas_on_itrusst.png')
    qc_figure(p, cents, out_png)

    print(f'\n[place] focused bowl on {PLACE_TARGET} (EEG seed {PLACE_SCALP}) '
          f'+ acoustic slab:')
    pl, c_map, rho_map, dz = place_and_slab(verbose=True)
    place_png = os.path.join(itrusst.fetch.mini_dir(), 'qc',
                             'mini_placement_on_itrusst.png')
    placement_figure(p, pl, c_map, place_png)

    print('\nDone.')
    return {'containment': m, 'targets': cents, 'qc_png': out_png,
            'placement': {k: pl[k] for k in (
                'target_lps', 'xdc_center_lps', 'focus_lps', 'beam_dir_3d',
                'perp_residual_mm', 'focus_to_target_mm',
                'apex_to_scalp_mm')},
            'slab_shape': list(c_map.shape),
            'slab_c_range': [float(c_map.min()), float(c_map.max())],
            'place_png': place_png}


if __name__ == '__main__':
    main()
