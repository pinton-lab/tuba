"""Fetch the Halle human-skull microCT from Zenodo (Kirchner 2022).

Source (open data, FAIR; doi:10.5281/zenodo.6108435):
  https://zenodo.org/records/6108435

Files written under templates/skull_microCT_zenodo/:
  halle_skull.nrrd            ~ 5.6 GB, 0.125 mm iso int16 ~HU
  halle_skull_segmented.nrrd  ~ 0.4 GB, semi-manual bone segmentation
  halle_skull_2D_projections.zip  ~ 4.4 GB, 2501 cone-beam projections
  etc.zip                     ~ 5 MB,   reconstruction settings + metadata
  README.pdf                  ~ 5 MB,   data descriptor

Zenodo allows unauthenticated direct downloads; this fetcher is fully
automatic. Re-run is a no-op for files already on disk.

Subject context (from README.pdf): ex vivo skull of a 70-year-old male
body donor (Institute of Anatomy, Martin-Luther-Universitat
Halle-Wittenberg). Imaged on Nikon Metrology XT H 225 cone-beam
micro-CT at 150 kV / 400 uA, 1 mm Cu filter, 2501 projections at
360.349 deg rotation. HU is ad-hoc cone-beam calibrated (water samples
+ air) and should not be interpreted as quantitative.
"""
import os
import sys
import urllib.request

DEST = os.environ.get(
    'HALLE_DEST',
    os.path.expanduser('~/.cache/tuba/human/source'),
)
os.makedirs(DEST, exist_ok=True)

URLS = {
    'halle_skull.nrrd':
        'https://zenodo.org/records/6108435/files/halle_skull.nrrd',
    'halle_skull_segmented.nrrd':
        'https://zenodo.org/records/6108435/files/halle_skull_segmented.nrrd',
    'halle_skull_2D_projections.zip':
        'https://zenodo.org/records/6108435/files/halle_skull_2D_projections.zip',
    'etc.zip':
        'https://zenodo.org/records/6108435/files/etc.zip',
    'README.pdf':
        'https://zenodo.org/records/6108435/files/README.pdf',
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
    print(f'Staging Halle human-skull microCT (Kirchner 2022) into {DEST}')
    # Default: only the reconstructed volume + descriptor. Heavy
    # projection / etc archives are opt-in via env var to avoid an
    # unexpected ~10 GB download.
    light = os.environ.get('HALLE_FETCH_ALL', '0') == '0'
    targets = (
        ['halle_skull.nrrd', 'halle_skull_segmented.nrrd', 'README.pdf']
        if light else list(URLS)
    )
    if light:
        print('  (light mode: skipping ~10 GB projections + etc.zip; set '
              'HALLE_FETCH_ALL=1 to grab them too)')
    for name in targets:
        _fetch(name, URLS[name])
    print('\nAll staged.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
