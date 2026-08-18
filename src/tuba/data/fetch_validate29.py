"""Fetch the VALiDATe29 squirrel-monkey (Saimiri sciureus) brain atlas.

Source (NITRC, Vanderbilt University Institute of Imaging Science):
  https://www.nitrc.org/projects/validate29

Schilling et al. (2017), *Neuroinformatics* 15(4):343-364,
doi:10.1007/s12021-017-9334-0. RRID:SCR_015542. License: CC BY.

The atlas ships as a single ``VALiDATe29.zip`` (~30 MB) released through
NITRC's File Release System behind the standard click-through agreement
(handled here by reusing the WHS/NITRC opener). The numeric FRS release
id drifts across versions 1.0/1.1/1.2, so rather than hard-code a
possibly-stale id we **discover** the newest ``download.php/<id>/
VALiDATe29.zip`` link from the project's file-release page. Override the
whole URL with ``$VALIDATE29_URL`` if discovery ever fails.

Re-running is a no-op once the zip is unpacked.
"""
import os
import re
import sys
import urllib.parse
import urllib.request
import zipfile

# Reuse the NITRC click-through downloader (cookie jar + i_agree POST)
# from the WHS rat fetcher so the agreement dance lives in one place.
from tuba.data.fetch_whs import _fetch_nitrc

DEST = os.environ.get(
    'VALIDATE29_DEST',
    os.path.expanduser('~/.cache/tuba/saimiri/atlas/validate29'),
)

PROJECT_URL = 'https://www.nitrc.org/projects/validate29/'
ZIP_NAME = 'VALiDATe29.zip'
ZIP_PATH = os.path.join(DEST, ZIP_NAME)
UNPACKED_MARKER = os.path.join(DEST, 'VALiDATe29')

_UA = {'User-Agent': 'Mozilla/5.0 (tuba fetch_validate29)'}


def _get(url):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req) as r:
        return r.read().decode('utf-8', errors='ignore')


def _discover_zip_url():
    """Find the newest ``download.php/<id>/VALiDATe29.zip`` FRS link.

    Walk: project page -> its ``frs/?group_id=N`` file-release page ->
    the highest-id ``download.php`` link ending in ``VALiDATe29.zip``.
    Returns the absolute URL. Honours ``$VALIDATE29_URL``.
    """
    override = os.environ.get('VALIDATE29_URL')
    if override:
        return override

    proj = _get(PROJECT_URL)
    frs = re.search(r'/frs/\?group_id=(\d+)', proj)
    if not frs:
        raise RuntimeError(
            'Could not find the file-release (frs) link on the VALiDATe29 '
            'project page. Set $VALIDATE29_URL to the direct '
            'download.php/<id>/VALiDATe29.zip URL and re-run.')
    frs_url = urllib.parse.urljoin(PROJECT_URL, frs.group(0))
    files = _get(frs_url)
    # e.g. /frs/download.php/12345/VALiDATe29.zip
    ids = re.findall(r'(/frs/download\.php/(\d+)/VALiDATe29\.zip)', files,
                     flags=re.IGNORECASE)
    if not ids:
        raise RuntimeError(
            f'No VALiDATe29.zip download link found at {frs_url}. '
            'Set $VALIDATE29_URL and re-run.')
    # newest release = highest numeric id
    best = max(ids, key=lambda t: int(t[1]))
    return urllib.parse.urljoin(frs_url, best[0])


def _unpack(zip_path, dest_dir):
    if os.path.isdir(UNPACKED_MARKER) and os.listdir(UNPACKED_MARKER):
        print(f'  cached  {UNPACKED_MARKER}')
        return UNPACKED_MARKER
    os.makedirs(UNPACKED_MARKER, exist_ok=True)
    print(f'  ZIP     extracting {zip_path} -> {UNPACKED_MARKER}')
    with zipfile.ZipFile(zip_path) as zf:
        n = len(zf.namelist())
        zf.extractall(UNPACKED_MARKER)
    print(f'  done    {n} entries')
    return UNPACKED_MARKER


def main():
    os.makedirs(DEST, exist_ok=True)
    print(f'Fetching VALiDATe29 squirrel-monkey atlas into {DEST}')
    if os.path.isdir(UNPACKED_MARKER) and os.listdir(UNPACKED_MARKER):
        print(f'  cached  {UNPACKED_MARKER}')
        print('\nDone.')
        return 0
    try:
        url = _discover_zip_url()
    except Exception as e:  # noqa: BLE001 - surface the reason to the user
        print(f'  ERROR   {e}')
        return 1
    print(f'  URL     {url}')
    _fetch_nitrc(url, ZIP_PATH)
    _unpack(ZIP_PATH, DEST)
    print('\nDone. Cortical labels + templates staged; '
          'tuba.atlases.validate29.VALiDATe29 resolves them by keyword.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
