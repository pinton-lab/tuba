"""Stage a Saimiri sciureus (common squirrel monkey) dry-skull microCT
for the geometry-only pillar.

Reference specimen
------------------
USNM 338948 (Saimiri sciureus), National Museum of Natural History,
scanned at the University of Texas High-Resolution X-ray CT Facility
(UTCT) and published on DigiMorph (Rossie, J., 2002,
https://digimorph.org/specimens/Saimiri_sciureus/338948/): 540 slices,
0.1189 mm isotropic, 54.0 mm field of reconstruction. The same
openVertebrate / oUTCT specimens are mirrored on MorphoSource (e.g.
media 000039292, "Skull & Mandible [CTImageSeries]"), the collection
this project's macaque pillar also draws from (Copes et al. 2016,
*Sci. Data* 3:160001, doi:10.1038/sdata.2016.1).

**Uncalibrated.** Museum microCT carries no HU phantom, so voxel values
are arbitrary reconstruction counts. This pillar therefore uses the
scan for GEOMETRY ONLY: the quantitative HU->acoustics mapping
(:class:`tuba.core.hu_acoustics.AubryHUMapping`) refuses to run on it,
and the slab loader falls back to the explicitly-flagged placeholder
ramp. See :mod:`tuba.species.saimiri`.

Access
------
Both DigiMorph and MorphoSource gate the raw image stack behind an
interactive request/login (no unauthenticated direct-fetch endpoint;
MorphoSource's catalogue API is Cloudflare-protected). This script,
like the macaque one, prints the source URLs + the expected drop
location, then unpacks any archive that lands in ``$SAIMIRI_SOURCE_DIR``.

Workflow
--------
  python -m tuba.data.fetch_saimiri     # prints instructions
  # request/download the stack from MorphoSource or DigiMorph, drop the
  # archive (zip/tar/tgz of TIFFs) into $SAIMIRI_SOURCE_DIR
  python -m tuba.data.fetch_saimiri     # re-run to unpack
"""
import os
import sys
import tarfile
import zipfile

DEST = os.environ.get(
    'SAIMIRI_SOURCE_DIR',
    os.path.expanduser('~/.cache/tuba/saimiri/source'),
)

DIGIMORPH_URL = 'https://digimorph.org/specimens/Saimiri_sciureus/338948/'
MORPHOSOURCE_SEARCH = (
    'https://www.morphosource.org/catalog/media?q=Saimiri+sciureus'
    '&f%5Bhuman_readable_media_type_sim%5D%5B%5D=CT+Image+Series')
VOXEL_MM = 0.1189           # DigiMorph USNM 338948 reconstructed pitch
APPROX_SLICES = 540


def _is_zip(p):
    return os.path.isfile(p) and os.path.getsize(p) > 1024 and zipfile.is_zipfile(p)


def _is_tar(p):
    return os.path.isfile(p) and os.path.getsize(p) > 1024 and tarfile.is_tarfile(p)


def _candidate_archives(dest):
    return [os.path.join(dest, f) for f in sorted(os.listdir(dest))
            if _is_zip(os.path.join(dest, f)) or _is_tar(os.path.join(dest, f))]


def _unpack(arch, dest_dir):
    base = os.path.splitext(os.path.basename(arch))[0]
    if base.endswith('.tar'):
        base = base[:-4]
    out = os.path.join(dest_dir, base)
    if os.path.isdir(out) and os.listdir(out):
        print(f'  cached     {out}')
        return out
    os.makedirs(out, exist_ok=True)
    if _is_zip(arch):
        with zipfile.ZipFile(arch) as zf:
            zf.extractall(out)
            print(f'  unpacked   {out}  ({len(zf.namelist())} entries)')
    else:
        with tarfile.open(arch) as tf:
            tf.extractall(out)
            print(f'  unpacked   {out}  ({len(tf.getnames())} entries)')
    return out


def main():
    os.makedirs(DEST, exist_ok=True)
    print(f'Staging Saimiri sciureus dry-skull microCT into {DEST}')
    print(f'  expected ~{VOXEL_MM} mm iso, ~{APPROX_SLICES} slices, dry skull '
          f'(GEOMETRY ONLY -- uncalibrated, no HU phantom)')

    archs = _candidate_archives(DEST)
    if archs:
        for a in archs:
            print(f'\n  found    {a} ({os.path.getsize(a)/1e6:.1f} MB)')
            _unpack(a, DEST)
        print('\nAll staged.')
        return 0

    print(f'\n  MISSING archive in {DEST}')
    print('           1. open one of:')
    print(f'                MorphoSource: {MORPHOSOURCE_SEARCH}')
    print(f'                DigiMorph:    {DIGIMORPH_URL}')
    print('           2. request/download the CT image series (TIFF stack)')
    print('              for an adult Saimiri sciureus cranium (~0.12 mm iso)')
    print(f'           3. save the archive into {DEST}/')
    print('           4. re-run this script to unpack')
    return 1


if __name__ == '__main__':
    sys.exit(main())
