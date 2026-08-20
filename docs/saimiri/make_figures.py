"""Generate the VALiDATe29 atlas-overview figure for the squirrel-monkey
pillar manuscript, from the staged atlas (no subject scan needed).

Mirrors the atlas-overview figures of the other pillars (rat
``whs_atlas.png``, macaque ``nmt_atlases.png``): top row = anatomical
template, bottom row = cortical labels, through the brain-mask centroid,
with the S1 (anterior parietal cortex) and M1 (primary motor cortex)
target centroids marked.

Run:  VALIDATE29_DEST=~/.cache/tuba/saimiri/atlas/validate29 \
      python docs/saimiri/make_figures.py
"""
import os

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def _slice(vol, axis, idx):
    sel = [slice(None)] * 3
    sel[axis] = idx
    return vol[tuple(sel)]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nibabel as nib
    from tuba.atlases.validate29 import VALiDATe29

    v = VALiDATe29()
    t2 = nib.load(v.template_path)
    labels_img = nib.load(v.annotation_path)
    mask_img = nib.load(v.brain_mask_path)
    tmpl = np.asarray(t2.dataobj).astype(np.float32)
    labels = np.asarray(labels_img.dataobj).astype(np.int32)
    mask = np.asarray(mask_img.dataobj) > 0

    # brain-mask centroid (voxel) for slice selection
    c = np.argwhere(mask).mean(0).astype(int)

    # cortical (gray-matter) labels only, ids 1..18, for a legible overlay
    cortical = np.where(labels <= 18, labels, 0)
    ncol = 18
    rng = np.linspace(0, 1, ncol + 1)
    palette = plt.get_cmap('tab20')(rng)
    palette[0] = [0, 0, 0, 0]                     # id 0 transparent
    lut = matplotlib.colors.ListedColormap(palette)

    # target centroids (atlas world mm -> voxel)
    targets = {}
    for key, col in (('S1', 'yellow'), ('M1', 'cyan'), ('S2', 'lime')):
        try:
            _name, idx = v.resolve_label(key, hemisphere='left')
        except Exception:
            continue
        ijk = np.argwhere(labels == idx)
        if len(ijk):
            targets[key] = (ijk.mean(0), col)

    planes = [('sagittal', 0), ('coronal', 1), ('axial', 2)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for j, (title, axis) in enumerate(planes):
        t_sl = _slice(tmpl, axis, int(c[axis]))
        l_sl = _slice(cortical, axis, int(c[axis]))
        m_sl = _slice(mask, axis, int(c[axis]))
        # top: template + brain-mask contour
        ax = axes[0, j]
        ax.imshow(t_sl.T, origin='lower', cmap='gray')
        ax.contour(m_sl.T, levels=[0.5], colors='deepskyblue', linewidths=0.8)
        ax.set_title(f'{title}: $T_2$ template + brain mask', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # bottom: cortical labels + target markers
        ax = axes[1, j]
        ax.imshow(t_sl.T, origin='lower', cmap='gray', alpha=0.6)
        ax.imshow(np.ma.masked_where(l_sl.T == 0, l_sl.T), origin='lower',
                  cmap=lut, vmin=0, vmax=ncol, alpha=0.75, interpolation='nearest')
        for key, (vox, col) in targets.items():
            inplane = [a for a in range(3) if a != axis]
            xy = (vox[inplane[0]], vox[inplane[1]])
            if abs(vox[axis] - c[axis]) <= 6:
                ax.plot(*xy, 'o', ms=8, mfc='none', mec=col, mew=2)
                ax.annotate(key, xy, color=col, fontsize=9,
                            xytext=(4, 4), textcoords='offset points')
        ax.set_title(f'{title}: cortical labels + S1/M1/S2', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('VALiDATe29 squirrel-monkey atlas (Schilling et al. 2017): '
                 '$T_2$ template, brain mask, region-level cortical labels',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, 'validate29_atlas.png')
    fig.savefig(out, dpi=110)
    print(f'  wrote {out}')
    return out


def subject_figures():
    """Subject-space figures from the staged-scan build (no-op with a note
    if the registration cache is absent). Produces:

    * ``saimiri_geometry.png`` -- source microCT orthoslices with the solid
      endocranial cavity overlaid (the D2 geometry pipeline);
    * ``saimiri_atlas_in_skull.png`` -- the canonical QC overlay
      (bone shell + warped VALiDATe29 brain + S1/M1 targets).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nibabel as nib
    from tuba.species import saimiri as S

    if not (os.path.exists(S.SAIMIRI_RAS_ALIGNED) and os.path.exists(S.CAVITY_NII)):
        print('  [subject figures] registration cache absent; stage the scan '
              'and run `python -m tuba.species.saimiri build` first.')
        return None
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- geometry figure: source scan + solid cavity ---
    sk = nib.load(S.SAIMIRI_RAS_ALIGNED)
    arr = np.asarray(sk.dataobj).astype(np.float32)
    cav = np.asarray(nib.load(S.CAVITY_NII).dataobj) > 0
    vx = float(abs(sk.affine[0, 0]))
    c = np.argwhere(cav).mean(0).astype(int)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.6))
    for k, (t, a) in enumerate([('sagittal', 0), ('coronal', 1), ('axial', 2)]):
        bg = _slice(arr, a, int(c[a]))
        cm = _slice(cav, a, int(c[a]))
        ax[k].imshow(np.clip(bg.T, 0, 16000), origin='lower', cmap='gray',
                     aspect='equal')
        ax[k].imshow(np.ma.masked_where(~cm.T, cm.T), origin='lower',
                     cmap='autumn', alpha=0.42, aspect='equal')
        ax[k].set_title(f'{t}', fontsize=11)
        ax[k].set_xticks([])
        ax[k].set_yticks([])
    fig.suptitle('Saimiri USNM 194346 microCT + solid endocranial cavity '
                 f'({cav.sum()*vx**3/1000:.1f} mL) — 97.7 um isotropic, '
                 'uncalibrated (geometry only)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = os.path.join(FIG_DIR, 'saimiri_geometry.png')
    fig.savefig(out1, dpi=110)
    plt.close(fig)
    print(f'  wrote {out1}')

    # --- atlas-in-skull QC: reuse the pillar's canonical figure ---
    out2 = os.path.join(FIG_DIR, 'saimiri_atlas_in_skull.png')
    S.qc_figure(out_png=out2, verbose=False)
    print(f'  wrote {out2}')
    return out1, out2


if __name__ == '__main__':
    main()
    subject_figures()
