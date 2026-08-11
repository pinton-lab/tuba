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
    print('\nDone.')
    return {'containment': m, 'targets': cents, 'qc_png': out_png}


if __name__ == '__main__':
    main()
