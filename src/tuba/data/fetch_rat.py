"""Stage the DigiMorph rat skull microCT (TMM M-2272).

Source page:
  https://digimorph.org/specimens/Rattus_norvegicus/

Single scan of an adult Rattus norvegicus cranium (Texas Memorial
Museum M-2272), reconstructed at 29.61 um isotropic - 1571 slices of
1024 x 1024, full extent ~28 x 28 x 46 mm, ~0.6 GB on disk. Dry skull
only (no soft tissue, no in-situ brain), so the WHS Sprague-Dawley
brain atlas (39 um T2* template) must be registered to the endocranial
cavity, analogous to the mouse Maga/Allen and macaque AMNH/NMT v2
pipelines.

The host (utexas.box.com) publishes a "shared static" link that
sometimes serves the zip directly to a non-interactive client and
sometimes redirects to an interactive consent page. This fetcher tries
a direct GET first; if that fails (HTTP 4xx, or the payload is not a
real zip), it falls back to the macaque-style "print instructions +
wait for drop" pattern.

Strain is unspecified - TMM M-2272 is a museum specimen, not a
lab-bred Sprague-Dawley. The species module (tuba.species.rat) should
treat any SD-derived atlas (WHS) registration as an across-strain
warp, on the same order of mismatch as the AMNH macaque pillar.

Workflow:
  python3 fetch_rat.py            # tries direct GET, falls back on failure
  python3 fetch_rat.py            # re-run to unpack a manually-saved zip
"""
import os
import sys
import urllib.error
import urllib.request
import zipfile

DEST = os.environ.get(
    'RAT_DEST',
    os.path.expanduser('~/.cache/tuba/rat/source'),
)

URL = 'https://utexas.box.com/shared/static/06k90e1ngabmqkq7ntvdexkcq2od2kpw.zip'
LANDING_URL = 'https://digimorph.org/specimens/Rattus_norvegicus/'
FILENAME = 'Rattus_norvegicus_TMM_M-2272.zip'
VOXEL_UM = 29.61
APPROX_GB = 0.6


def _is_real_zip(path):
    return os.path.exists(path) and os.path.getsize(path) > 1024 \
           and zipfile.is_zipfile(path)


def _try_direct_fetch(url, out):
    print(f'  GET     {url}')
    tmp = out + '.part'
    try:
        with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
            size = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                size += len(chunk)
                if size % (32 << 20) < (1 << 20):
                    print(f'    {size/1e6:.0f} MB...')
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f'  direct fetch failed: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    if not _is_real_zip(tmp):
        print('  direct fetch returned non-zip payload (likely a consent page)')
        os.remove(tmp)
        return False
    os.rename(tmp, out)
    print(f'  saved   {out} ({os.path.getsize(out)/1e6:.1f} MB)')
    return True


def _unpack(zip_path, dest_dir):
    out = os.path.join(dest_dir, 'TMM_M-2272')
    if os.path.isdir(out) and os.listdir(out):
        print(f'  cached     {out}')
        return out
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(out)
    print(f'  unpacked   {out}  ({len(names)} entries)')
    return out


def main():
    os.makedirs(DEST, exist_ok=True)
    print(f'Staging DigiMorph rat skull microCT (TMM M-2272) into {DEST}')
    print(f'  {VOXEL_UM} um iso, ~{APPROX_GB} GB, dry skull only')

    zip_path = os.path.join(DEST, FILENAME)
    if not _is_real_zip(zip_path):
        _try_direct_fetch(URL, zip_path)

    if _is_real_zip(zip_path):
        print(f'\n  found    {zip_path} ({os.path.getsize(zip_path)/1e9:.2f} GB)')
        _unpack(zip_path, DEST)
        print('\nAll staged.')
        return 0

    print(f'\n  MISSING  {zip_path}')
    print(f'           1. open  {LANDING_URL}')
    print(f'           2. follow the "CT Scan data" link (Box.com)')
    print(f'           3. save as {FILENAME} into {DEST}/')
    print(f'           4. re-run this script')
    return 1


if __name__ == '__main__':
    sys.exit(main())
