"""Fetch NMT v2.0 symmetric macaque template + bundled CHARM/SARM/D99 atlases.

Source (NIMH/AFNI, Jung et al. 2021 NeuroImage 235:117997):
  https://afni.nimh.nih.gov/pub/dist/atlases/macaque/nmt/NMT_v2.0_sym.tgz

NMT_v2.0_sym.tgz (~400 MB unpacked) bundles:
  - NMT_v2.0_sym/                    : 0.25 mm averaged-T1 template + masks
  - NMT_v2.0_sym_05mm/               : 0.5 mm low-res template
  - NMT_v2.0_sym_fh/                 : full-head FOV variant (the one we
                                       want; preserves skull-relative
                                       geometry for ANTs against an
                                       endocranial cavity)
  - NMT_v2.0_sym_surfaces/           : GIfTI cortical surfaces
  - tables_CHARM/                    : 6-level cortical hierarchy + CSV
  - tables_SARM/                     : 6-level subcortical hierarchy + CSV
  - tables_D99/                      : D99 v2 cortical+subcortical labels

Files cached on disk; re-run is a no-op if already unpacked.
"""
import os
import sys
import tarfile
import urllib.request

DEST = os.environ.get(
    'NMT_DEST',
    os.path.expanduser('~/.cache/tuba/macaque/atlas/nmt_v2'),
)
os.makedirs(DEST, exist_ok=True)

NMT_URL = ('https://afni.nimh.nih.gov/pub/dist/atlases/macaque/nmt/'
           'NMT_v2.0_sym.tgz')
TGZ_PATH = os.path.join(DEST, 'NMT_v2.0_sym.tgz')
UNPACKED_MARKER = os.path.join(DEST, 'NMT_v2.0_sym')


def _fetch(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 1 << 20:
        print(f'  cached  {out} ({os.path.getsize(out)/1e6:.1f} MB)')
        return out
    print(f'  GET     {url}')
    tmp = out + '.part'
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
    os.rename(tmp, out)
    print(f'  saved   {out} ({os.path.getsize(out)/1e6:.1f} MB)')
    return out


def _unpack(tgz, dest_dir):
    if os.path.isdir(UNPACKED_MARKER) and os.listdir(UNPACKED_MARKER):
        print(f'  cached  {UNPACKED_MARKER}')
        return
    print(f'  TAR     extracting {tgz} into {dest_dir}')
    with tarfile.open(tgz, 'r:gz') as tf:
        tf.extractall(dest_dir)
        names = tf.getnames()
    print(f'  done    {len(names)} entries')


def main():
    print(f'Fetching NMT v2.0_sym + CHARM + SARM + D99 into {DEST}')
    _fetch(NMT_URL, TGZ_PATH)
    _unpack(TGZ_PATH, DEST)
    print('\nTop-level contents:')
    for f in sorted(os.listdir(DEST)):
        p = os.path.join(DEST, f)
        if os.path.isdir(p):
            print(f'  d  {f}')
        else:
            print(f'  f  {f}  ({os.path.getsize(p)/1e6:.1f} MB)')
    print('\nDone.')


if __name__ == '__main__':
    main()
