"""Fetch Allen CCFv3 atlas: 25 um average template, annotation, structure tree.

Sources (Allen Institute, 2017 CCFv3 release):
  http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/

Files written under templates/allen_ccfv3/:
  average_template_25.nrrd   ~ 50 MB,  Nissl-style average reference
  annotation_25.nrrd         ~ 30 MB,  CCFv3 structure-id volume
  structure_tree.json        ~ 1 MB,   id -> name/acronym map (incl. 'cc' = 776)

All three are cached on disk; re-run is a no-op.
"""
import os
import sys
import urllib.request

DEST = os.environ.get(
    'ALLEN_DEST',
    os.path.expanduser('~/.cache/tuba/mouse/atlas/allen_ccfv3'),
)
os.makedirs(DEST, exist_ok=True)

URLS = {
    'average_template_25.nrrd':
        'http://download.alleninstitute.org/informatics-archive/current-release/'
        'mouse_ccf/average_template/average_template_25.nrrd',
    'annotation_25.nrrd':
        'http://download.alleninstitute.org/informatics-archive/current-release/'
        'mouse_ccf/annotation/ccf_2017/annotation_25.nrrd',
    'structure_tree.json':
        'http://api.brain-map.org/api/v2/structure_graph_download/1.json',
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
    print('Fetching Allen CCFv3 atlas into', DEST)
    for name, url in URLS.items():
        _fetch(name, url)
    print('Done.')


if __name__ == '__main__':
    main()
