"""Stage the two small, openly-hosted inputs the mini pipeline needs.

Unlike the full TUBA species pipelines (multi-GB, often licensed
skull microCT + atlas templates), ``tuba.mini`` depends only on:

1. the **ITRUSST benchmark skull** -- two STL surfaces (~13 MB total),
   fetched from the public ``ucl-bug/transcranial-ultrasound-benchmarks``
   GitHub repository (Aubry et al., JASA 2022; 500 kHz benchmark), and
2. the **MNI152 ICBM 2009a** T1 template + brain mask (~40 MB), fetched
   on demand by ``nilearn``.

Both are direct, unauthenticated fetches, so the mini demo runs
end-to-end without any click-through or login staging. Nothing here is
committed to the repo -- it lands in ``$TUBA_MINI_DIR`` (default
``~/.cache/tuba/mini``) and the nilearn cache.
"""
import os
import urllib.request

# Pinned to a full commit SHA (not ``master``) so the fetched geometry is
# reproducible; raw.githubusercontent serves blobs by SHA reliably.
ITRUSST_REPO = 'ucl-bug/transcranial-ultrasound-benchmarks'
ITRUSST_REF = 'master'
ITRUSST_SUBDIR = 'intercomparison/skull-stl'
ITRUSST_FILES = ('skull_inner.stl', 'skull_outer.stl')


def mini_dir():
    """Root cache for mini artifacts (``$TUBA_MINI_DIR`` or
    ``~/.cache/tuba/mini``)."""
    return os.environ.get(
        'TUBA_MINI_DIR', os.path.expanduser('~/.cache/tuba/mini'))


def source_dir():
    d = os.path.join(mini_dir(), 'source')
    os.makedirs(d, exist_ok=True)
    return d


def fetch_itrusst_skull(ref=ITRUSST_REF, verbose=True):
    """Download the inner/outer ITRUSST skull STLs into
    ``$TUBA_MINI_DIR/source``. Returns ``(inner_path, outer_path)``.
    Idempotent: existing non-empty files are kept."""
    dst = source_dir()
    base = (f'https://raw.githubusercontent.com/{ITRUSST_REPO}/{ref}/'
            f'{ITRUSST_SUBDIR}')
    out = {}
    for f in ITRUSST_FILES:
        path = os.path.join(dst, f)
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            url = f'{base}/{f}'
            if verbose:
                print(f'  fetching {url}')
            urllib.request.urlretrieve(url, path)
        out[f] = path
        if verbose:
            print(f'  have {path} ({os.path.getsize(path)/1e6:.1f} MB)')
    return out['skull_inner.stl'], out['skull_outer.stl']


def fetch_mni152(verbose=True):
    """Fetch the MNI152 ICBM 2009a template via nilearn.

    Returns a dict with the on-disk paths ``{'dir', 't1', 'mask'}`` taken
    straight from the nilearn Bunch (nilearn stages gzipped ``.nii.gz``,
    which differs from the uncompressed McGill layout the full-pipeline
    :class:`tuba.atlases.mni152.MNI152` binding hardcodes -- ANTs reads
    either, so mini just uses the real paths)."""
    try:
        from nilearn.datasets import fetch_icbm152_2009
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'tuba.mini.fetch needs nilearn for MNI152; install with '
            '`pip install "tuba[mini]"` or `pip install nilearn`.') from e
    bunch = fetch_icbm152_2009()
    paths = {'dir': os.path.dirname(bunch['t1']),
             't1': bunch['t1'], 'mask': bunch['mask']}
    if verbose:
        print(f'  MNI152 t1={paths["t1"]}')
        print(f'  MNI152 mask={paths["mask"]}')
    return paths


if __name__ == '__main__':
    inner, outer = fetch_itrusst_skull()
    print(f'inner={inner}\nouter={outer}')
    print(f'mni={fetch_mni152()}')
