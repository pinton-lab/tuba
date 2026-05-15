"""Stage the Maga lab (UW) 4K mouse skull microCT scan.

Source page:
  https://faculty.washington.edu/maga/SCRI_MicroCT/

Single scan of an adult mouse dry skull, acquired on a Skyscan 1272:
  6.125 um iso voxels, ~35 GB, 8.5 h scan, 4K detector.

For 15 MHz transcranial (lambda ~ 170 um in cortical bone) this gives
~28 vox/lambda, versus ~1.7 for DigiMouse's 100 um. Dry skull only (no
soft tissue, no in-situ brain), so Allen CCFv3 must be registered to the
endocranial cavity rather than to a soft-tissue volume as in the
DigiMouse pipeline.

The host (Box.com) requires an interactive click to download; there is
no unauthenticated direct-fetch endpoint. This script prints the share
URL and expected filename, then unpacks the zip once it lands in DEST.

Workflow:
  python3 templates/fetch_maga_skull.py   # prints instructions
  # (click through in browser, save zip into DEST)
  python3 templates/fetch_maga_skull.py   # re-run to unpack
"""
import os
import sys
import zipfile

DEST = os.environ.get(
    'MAGA_DEST',
    os.path.expanduser('~/.cache/tuba/mouse/source'),
)

URL = 'https://app.box.com/s/mtvctm7rus8udrav7qkrraq3syatvf5n'
FILENAME = 'dry_skull_4K_6.125micron_Rec.zip'
VOXEL_UM = 6.125
APPROX_GB = 35.0


def _is_real_zip(path):
    return os.path.exists(path) and os.path.getsize(path) > 1024 and zipfile.is_zipfile(path)


def _unpack(zip_path, dest_dir):
    out = os.path.join(dest_dir, 'skull_4k')
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
    print(f'Staging Maga 4K skull microCT into {DEST}')
    print(f'  {VOXEL_UM} um iso, ~{APPROX_GB} GB, dry skull only')

    zip_path = os.path.join(DEST, FILENAME)
    if _is_real_zip(zip_path):
        print(f'\n  found    {zip_path} ({os.path.getsize(zip_path)/1e9:.2f} GB)')
        _unpack(zip_path, DEST)
        print('\nAll staged.')
        return 0

    print(f'\n  MISSING  {zip_path}')
    print(f'           1. open  {URL}')
    print(f'           2. click Download, save as {FILENAME}')
    print(f'           3. drop into {DEST}/  and re-run this script')
    return 1


if __name__ == '__main__':
    sys.exit(main())
