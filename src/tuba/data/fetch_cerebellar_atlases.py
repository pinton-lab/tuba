"""Fetch the Diedrichsen probabilistic cerebellar atlas (lobules + deep
cerebellar nuclei) in MNI152 space.

Source (open; cite Diedrichsen 2009 for the lobules and Diedrichsen 2011
for the deep cerebellar nuclei):
  https://github.com/DiedrichsenLab/cerebellar_atlases  (Diedrichsen_2009/)
  https://www.diedrichsenlab.org/imaging/propatlas.htm

Files written under ``~/nilearn_data/diedrichsen_2009/`` (overridable via
the ``DIEDRICHSEN_DEST`` env var), matching the ``_user_cache`` layout the
other MNI parcellations in :mod:`tuba.atlases.mni152` use::

  atl-Anatom_space-MNI_dseg.nii      ~1.3 MB  discrete "most probable
                                              compartment" label volume
                                              (34 labels: 28 lobules +
                                              6 deep nuclei, IDs 29-34)
  atl-Anatom_space-MNI_probseg.nii   ~45 MB   4-D per-compartment
                                              probability maps
                                              (153x103x84x34, prob in
                                              [0, 1]; volume v -> label v+1)
  atl-Anatom.tsv                              index<TAB>name<TAB>color
  atl-Anatom.lut                              FSL-style LUT (id R G B name;
                                              RGB floats in [0, 1])

The dentate / interposed / fastigial nuclei are labels 29-34; tuba's
``mni152.dentate_mask()`` thresholds the probseg. GitHub raw serves the
repo files directly (the repo carries ``.noannex``, so the NIfTIs are
real content, not git-annex pointers). Re-run is a no-op for files
already on disk.

Space caveat: the repo also ships SUIT-space variants
(``atl-Anatom_space-SUIT_*``) and an ``MNISym`` variant. tuba binds the
MNI files so the existing MNI -> Halle SyN warp applies. The MNI affine
is a clean 1 mm RAS grid cropped to the infratentorial FOV (origin
[76, -108, -72]).
"""
import os
import sys
import urllib.request

DEST = os.environ.get(
    'DIEDRICHSEN_DEST',
    os.path.expanduser('~/nilearn_data/diedrichsen_2009'),
)
os.makedirs(DEST, exist_ok=True)

# Pinned to the master branch of the DiedrichsenLab/cerebellar_atlases
# repo (current commit 1b62fdc, 2025-10-26). Switch to a commit SHA in
# place of ``master`` if a reproducible pin is needed.
_RAW = ('https://raw.githubusercontent.com/DiedrichsenLab/'
        'cerebellar_atlases/master/Diedrichsen_2009')

URLS = {
    'atl-Anatom_space-MNI_dseg.nii':
        f'{_RAW}/atl-Anatom_space-MNI_dseg.nii',
    'atl-Anatom_space-MNI_probseg.nii':
        f'{_RAW}/atl-Anatom_space-MNI_probseg.nii',
    'atl-Anatom.tsv':
        f'{_RAW}/atl-Anatom.tsv',
    'atl-Anatom.lut':
        f'{_RAW}/atl-Anatom.lut',
}


def _fetch(name, url):
    out = os.path.join(DEST, name)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f'  cached  {out} ({os.path.getsize(out)/1e6:.1f} MB)')
        return out
    print(f'  GET     {url}')
    tmp = out + '.part'
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.rename(tmp, out)
    print(f'  saved   {out} ({os.path.getsize(out)/1e6:.1f} MB)')
    return out


def main():
    print(f'Staging Diedrichsen cerebellar atlas (MNI dseg + probseg) '
          f'into {DEST}')
    for name, url in URLS.items():
        _fetch(name, URLS[name])
    print('\nAll staged.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
