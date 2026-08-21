"""Stage a Saimiri sciureus (common squirrel monkey) dry-skull microCT
for the geometry-only pillar.

Reference specimen (STAGED)
---------------------------
NMNH USNM 194346 (squirrel monkey, *Saimiri sp.*), MorphoSource media
000116521 ("Skull [CTImageSeries] [Etc]", ark:/87602/m4/M116521),
scanned at the University of Texas High-Resolution X-ray CT Facility
(UTCT, ACTIS scanner) and also on DigiMorph (Rossie, J.,
https://digimorph.org/specimens/Saimiri_sciureus/194346/): 490 16-bit
TIFF slices, 504x422 px, ANISOTROPIC 0.0977 mm in-plane x 0.1189 mm
slice. The MorphoSource record's taxonomy is internally inconsistent
(physical_object_taxonomy_name 'Saimiri boliviensis'; Rossie's scan
description 'Saimiri sciureus sciureus, USNM 194346, male') -- genus
Saimiri either way. Part of the openVertebrate / oUTCT collection the
macaque pillar also draws from (Copes et al. 2016, *Sci. Data*
3:160001, doi:10.1038/sdata.2016.1).

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

DIGIMORPH_URL = 'https://digimorph.org/specimens/Saimiri_sciureus/194346/'
MORPHOSOURCE_MEDIA = 'https://www.morphosource.org/concern/media/000116521'
MORPHOSOURCE_SEARCH = (
    'https://www.morphosource.org/catalog/media?q=Saimiri'
    '&f%5Bhuman_readable_media_type_sim%5D%5B%5D=CT+Image+Series')
VOXEL_INPLANE_MM = 0.0977   # media 000116521 x/y pixel spacing
VOXEL_SLICE_MM = 0.1189     # media 000116521 z spacing (slice thickness)
VOXEL_MM = VOXEL_INPLANE_MM  # in-plane reconstruction pitch
APPROX_SLICES = 490


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
    print(f'Staging Saimiri dry-skull microCT (USNM 194346, media 000116521) '
          f'into {DEST}')
    print(f'  expected ~{APPROX_SLICES} slices, {VOXEL_INPLANE_MM} mm in-plane '
          f'x {VOXEL_SLICE_MM} mm slice (anisotropic), dry skull '
          f'(GEOMETRY ONLY -- uncalibrated, no HU phantom)')

    archs = _candidate_archives(DEST)
    if archs:
        for a in archs:
            print(f'\n  found    {a} ({os.path.getsize(a)/1e6:.1f} MB)')
            _unpack(a, DEST)
        print('\nAll staged.')
        return 0

    print(f'\n  MISSING archive in {DEST}')
    print('           1. open the reference specimen (free MorphoSource '
          'account, per-record approval):')
    print(f'                MorphoSource media: {MORPHOSOURCE_MEDIA}')
    print(f'                DigiMorph:          {DIGIMORPH_URL}')
    print(f'                (or search:         {MORPHOSOURCE_SEARCH} )')
    print('           2. use the "Download" action to fetch the CT image '
          'series (zip of the 16bit/ TIFF stack)')
    print(f'           3. save the archive into {DEST}/')
    print('           4. re-run this script to unpack')
    return 1


if __name__ == '__main__':
    sys.exit(main())
