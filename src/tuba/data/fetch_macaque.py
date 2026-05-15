"""Stage an adult Macaca mulatta dry skull microCT from MorphoSource.

Source collection: Copes, Lucas, Thostenson, Hoekstra, Boyer (2016),
``A collection of non-human primate computed tomography scans housed
in MorphoSource''. *Sci Data* 3:160001, doi:10.1038/sdata.2016.1.

Adult M. mulatta vouchered specimens at the Harvard Museum of
Comparative Zoology (MCZ) at 90.75 um isotropic, n=2 in the original
release. License is CC BY-NC; download requires a free MorphoSource
account and per-record approval.

The host (MorphoSource) requires an interactive click + login to
download; there is no unauthenticated direct-fetch endpoint and the
catalogue API is bot-protected (Cloudflare Anubis). This script prints
the search URL and the expected drop location, then unpacks any zip /
tar / tgz archive that lands in DEST.

Workflow
--------
  python3 templates/fetch_morphosource_mmulatta.py    # prints instructions
  # Open the search URL in a browser, log in, find an adult skull-only
  # microCT record (e.g. catalogue MCZ:Mamm:30384 or analogous),
  # request the download, and save the resulting archive into DEST.
  python3 templates/fetch_morphosource_mmulatta.py    # re-run to unpack

If MorphoSource serves the data as a tarball of 16-bit TIFFs, the
downstream pipeline (registration/macaque_skull_constants.py) will
glob the unpacked TIFFs directly. If it serves a single multi-page
TIFF or a NIfTI, the same pipeline picks the format up automatically.
"""
import os
import sys
import tarfile
import zipfile

DEST = os.environ.get(
    'MMULATTA_DEST',
    os.path.expanduser('~/.cache/tuba/macaque/source'),
)

SEARCH_URL = (
    'https://www.morphosource.org/catalog/media?'
    'q=Macaca+mulatta+MCZ&search_field=all_fields'
    '&f%5Bmedia_type_sim%5D%5B%5D=CTImageSeries'
)
PAPER_DOI = 'https://doi.org/10.1038/sdata.2016.1'
APPROX_GB = 2.5            # ~2.3-2.4 GB per Copes 2016 Table 1 for adult Mmu
VOXEL_UM = 90.75


def _is_real_zip(p):
    return os.path.exists(p) and os.path.getsize(p) > 1024 \
           and zipfile.is_zipfile(p)


def _is_real_tar(p):
    return os.path.exists(p) and os.path.getsize(p) > 1024 \
           and tarfile.is_tarfile(p)


def _candidate_archives(dest):
    out = []
    for f in sorted(os.listdir(dest)):
        p = os.path.join(dest, f)
        if not os.path.isfile(p):
            continue
        if _is_real_zip(p) or _is_real_tar(p):
            out.append(p)
    return out


def _unpack(arch_path, dest_dir):
    base = os.path.splitext(os.path.basename(arch_path))[0]
    if base.endswith('.tar'):
        base = base[:-4]
    out = os.path.join(dest_dir, base)
    if os.path.isdir(out) and os.listdir(out):
        print(f'  cached     {out}')
        return out
    os.makedirs(out, exist_ok=True)
    if _is_real_zip(arch_path):
        with zipfile.ZipFile(arch_path) as zf:
            zf.extractall(out)
            print(f'  unpacked   {out}  ({len(zf.namelist())} entries)')
    else:
        with tarfile.open(arch_path) as tf:
            tf.extractall(out)
            print(f'  unpacked   {out}  ({len(tf.getnames())} entries)')
    return out


def main():
    os.makedirs(DEST, exist_ok=True)
    print(f'Staging M. mulatta MCZ microCT into {DEST}')
    print(f'  expected ~{VOXEL_UM} um iso, ~{APPROX_GB} GB per specimen, '
          f'dry skull only (Copes 2016)')

    archs = _candidate_archives(DEST)
    if archs:
        for a in archs:
            print(f'\n  found    {a} ({os.path.getsize(a)/1e9:.2f} GB)')
            _unpack(a, DEST)
        print('\nAll staged.')
        return 0

    print(f'\n  MISSING archive in {DEST}')
    print(f'           1. log in to MorphoSource (free account)')
    print(f'           2. open  {SEARCH_URL}')
    print(f'           3. pick an ADULT M. mulatta cranium, microCT,')
    print(f'              ~90 um voxel, MCZ-vouchered (Copes 2016 release)')
    print(f'              cf. paper {PAPER_DOI}')
    print(f'           4. request the download, save the archive into')
    print(f'                 {DEST}/')
    print(f'           5. re-run this script to unpack')
    return 1


if __name__ == '__main__':
    sys.exit(main())
