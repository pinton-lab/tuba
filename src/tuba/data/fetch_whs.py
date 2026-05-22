"""Fetch WHS Sprague-Dawley rat atlas: v4 pack (template + labels) + v4.01 patch.

Source (NITRC, University of Oslo Neural Systems Lab):
  https://www.nitrc.org/projects/whs-sd-atlas

Two-step fetch.

  WHS_SD_rat_atlas_v4_pack.zip  (~1 GB unpacked) bundles:
    - WHS_SD_rat_T2star_v1.01.nii.gz   : 39 um ex vivo T2* MRI template
    - WHS_SD_rat_atlas_v4.nii.gz       : v4 label volume (220 structures)
    - WHS_SD_rat_atlas_v4.label        : ITK-SNAP colormap + name table
    - WHS_SD_rat_atlas_v4.ilf / .ilv   : Mouse BIRN Atlasing Toolkit packs
    - documentation PDF + JSON

  WHS_SD_rat_atlas_v4.01.nii.gz  (~4 MB) is the v4.01 label-only patch
  released 2023-04-20 (Kleven et al. 2023, Nature Methods). Overlays
  the v4 labels with 222 structures (112 new, 57 revised). Companion
  .label file ships alongside.

License: CC BY 4.0. Re-run is a no-op for files already on disk.
"""
import http.cookiejar
import os
import re
import sys
import urllib.parse
import urllib.request
import zipfile

DEST = os.environ.get(
    'WHS_DEST',
    os.path.expanduser('~/.cache/tuba/rat/atlas/whs_sd_v4'),
)
os.makedirs(DEST, exist_ok=True)

PACK_URL = ('https://www.nitrc.org/frs/download.php/12264/'
            'WHS_SD_rat_atlas_v4_pack.zip')
PACK_PATH = os.path.join(DEST, 'WHS_SD_rat_atlas_v4_pack.zip')
UNPACKED_MARKER = os.path.join(DEST, 'WHS_SD_rat_atlas_v4_pack')

V401_URLS = {
    'WHS_SD_rat_atlas_v4.01.nii.gz':
        'https://www.nitrc.org/frs/download.php/13398/'
        'WHS_SD_rat_atlas_v4.01.nii.gz',
    'WHS_SD_rat_atlas_v4.01.label':
        'https://www.nitrc.org/frs/download.php/13399/'
        'WHS_SD_rat_atlas_v4.01.label',
}


_NITRC_AGREEMENT_RE = re.compile(
    r'FORM ACTION="(?P<action>[^"]*i_agree=1[^"]*)"', re.IGNORECASE)


def _build_opener_with_cookies():
    """Build a urllib opener that follows redirects and persists cookies
    across the NITRC click-through dance (PHPSESSID + agreement flag)."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar))
    return opener


def _nitrc_open(opener, url, data=None):
    """GET (or POST when data is given) through the cookie-jar opener,
    handling the NITRC click-through agreement transparently. Returns
    a file-like response whose ``.read()`` yields either the agreement
    HTML or the real payload, depending on session state."""
    if isinstance(data, dict):
        data = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=data)
    return opener.open(req)


def _fetch_nitrc(url, out):
    """NITRC click-through fetch.

    Step 1: GET the file URL. NITRC returns either the payload (200
    binary) or the agreement HTML form (200 HTML containing
    ``<FORM ACTION=".../?i_agree=1..."``).

    Step 2: if HTML, POST i_agree=1 + release_id_<N>=1 to the form's
    action URL. NITRC returns a meta-redirect HTML page pointing at
    ``.../?i_agree=1&download_now=1``.

    Step 3: GET the download_now URL to get the actual payload.

    The cookie jar threads PHPSESSID through all three steps; without
    it NITRC drops the agreement state between requests.
    """
    if os.path.exists(out) and os.path.getsize(out) > 1024:
        print(f'  cached  {out} ({os.path.getsize(out)/1e6:.1f} MB)')
        return out
    print(f'  GET     {url}')
    opener = _build_opener_with_cookies()
    resp = _nitrc_open(opener, url)
    body = resp.read()
    if not body.startswith(b'<?xml') and not body.startswith(b'<!DOC') \
            and not body.startswith(b'<html'):
        # First GET already served the payload (older releases without
        # the agreement wall).
        return _write_payload(body, opener, url, out, prefetched=True)

    # HTML agreement page; extract the form action URL.
    m = _NITRC_AGREEMENT_RE.search(body.decode('utf-8', errors='ignore'))
    if not m:
        raise RuntimeError(
            f'NITRC returned HTML at {url} but no i_agree form was found.\n'
            f'  inspect: first 400 chars: {body[:400]!r}')
    action_path = m.group('action').replace('&amp;', '&')
    action_url = urllib.parse.urljoin(url, action_path)
    # Extract the release_id_<N> hidden-field name from the URL so the
    # POST mimics the browser exactly.
    rel_id = re.search(r'release_id=(\d+)', action_path)
    form = {'i_agree': '1'}
    if rel_id:
        form[f'release_id_{rel_id.group(1)}'] = '1'

    print(f'  POST    {action_url}  (i_agree)')
    resp = _nitrc_open(opener, action_url, data=form)
    body2 = resp.read().decode('utf-8', errors='ignore')
    dl_match = re.search(r'href="([^"]*download_now=1[^"]*)"', body2)
    if not dl_match:
        raise RuntimeError(
            f'NITRC accepted the agreement but did not return a download_now URL.\n'
            f'  first 400 chars: {body2[:400]!r}')
    dl_url = urllib.parse.urljoin(url, dl_match.group(1).replace('&amp;', '&'))
    print(f'  GET     {dl_url}')
    resp = _nitrc_open(opener, dl_url)
    return _write_payload(None, opener, None, out, stream=resp)


def _write_payload(prefetched_bytes, opener, url, out, *,
                    stream=None, prefetched=False):
    """Write the payload to ``out`` (atomically via .part). Handles both
    pre-buffered bytes (small files we already read) and streamed responses
    (large files we want to chunk to disk)."""
    tmp = out + '.part'
    if prefetched:
        with open(tmp, 'wb') as f:
            f.write(prefetched_bytes)
        size = len(prefetched_bytes)
    else:
        size = 0
        with open(tmp, 'wb') as f:
            while True:
                chunk = stream.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                size += len(chunk)
                if size % (32 << 20) < (1 << 20):
                    print(f'    {size/1e6:.0f} MB...')
    os.rename(tmp, out)
    print(f'  saved   {out} ({size/1e6:.1f} MB)')
    return out


# Back-compat alias for the original direct-fetch entry point.
_fetch = _fetch_nitrc


def _unpack(zip_path, dest_dir):
    if os.path.isdir(UNPACKED_MARKER) and os.listdir(UNPACKED_MARKER):
        print(f'  cached  {UNPACKED_MARKER}')
        return
    os.makedirs(UNPACKED_MARKER, exist_ok=True)
    print(f'  ZIP     extracting {zip_path} into {UNPACKED_MARKER}')
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(UNPACKED_MARKER)
    print(f'  done    {len(names)} entries')


def main():
    print(f'Fetching WHS SD rat atlas (v4 pack + v4.01 labels) into {DEST}')
    _fetch(PACK_URL, PACK_PATH)
    _unpack(PACK_PATH, DEST)
    for name, url in V401_URLS.items():
        _fetch(url, os.path.join(DEST, name))
    print('\nDone.')


if __name__ == '__main__':
    main()
