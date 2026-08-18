"""VALiDATe29 squirrel-monkey (Saimiri sciureus) brain-atlas binding.

VALiDATe29 is an MRI-based multi-channel atlas of the common squirrel
monkey brain built from 29 animals at the Vanderbilt University
Institute of Imaging Science, combining anatomical templates (proton-
density, T1, T2*), diffusion templates (FA, MD), ex-vivo templates, and
**histologically-defined cortical labels** plus tractography-defined
white-matter labels.

Reference
---------
Schilling, K. G., Gao, Y., Stepniewska, I., Wu, T.-L., Wang, F.,
Landman, B. A., Gore, J. C., Chen, L. M., Anderson, A. W. (2017).
"The VALiDATe29 MRI Based Multi-Channel Atlas of the Squirrel Monkey
Brain." *Neuroinformatics* 15(4):343-364. doi:10.1007/s12021-017-9334-0.
RRID:SCR_015542. Distributed CC BY via NITRC
(https://www.nitrc.org/projects/validate29). See
:mod:`tuba.data.fetch_validate29` for staging.

Registration mode
-----------------
``"cavity_binary"`` -- same convention as the rat (WHS) and mouse
pillars: the subject skull's endocranial cavity (moving) is registered
to the atlas **brain mask** (fixed) with ANTs SyN (affine + deformable),
then the atlas template + cortical labels are pulled back into subject
space through the saved inverse transforms.

Layout tolerance
----------------
The exact filenames inside ``VALiDATe29.zip`` are not hard-coded: the
release bundles several template channels and label volumes and the
naming has drifted across versions 1.0/1.1/1.2. This binding therefore
*resolves* the template + label volume + label table by globbing the
staged directory with keyword heuristics, and every path is overridable
by an explicit constructor argument or environment variable. When a file
cannot be resolved unambiguously the accessor raises with the list of
candidates it actually found, so the user can pin the right one -- it
never guesses silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import glob
import os
import re
from typing import Optional


# Cortical target groups for the somatosensory/motor circuit this pillar
# studies. Each entry is a tuple of case-insensitive substrings; the
# first label whose name contains ANY of them (see :meth:`resolve_label`)
# is used. The squirrel-monkey somatosensory areas follow the Kaas-lab
# Brodmann-style parcellation carried by VALiDATe29's histological labels
# (areas 3a/3b/1/2 in S1, area 4 in M1). Exact spellings vary with the
# staged .label/LUT, so match on the area number substrings, which are
# stable, rather than on a full canonical name.
TARGET_LABEL_KEYWORDS = {
    'S1_3b': ('area 3b', '3b', 'brodmann area 3b'),
    'S1_area1': ('area 1', 'brodmann area 1'),
    'M1': ('area 4', 'primary motor', 'm1', 'brodmann area 4'),
}


@dataclass
class VALiDATe29:
    """Path-binding + label lookup for a staged VALiDATe29 atlas directory.

    Parameters
    ----------
    atlas_dir : str
        Directory holding the unpacked ``VALiDATe29.zip`` contents.
        Defaults to ``$VALIDATE29_DEST`` then
        ``~/.cache/tuba/saimiri/atlas/validate29``.
    template_channel : str
        Preferred anatomical template channel keyword for the fixed
        image / QC background: one of ``'t2star'``, ``'pd'``, ``'t1'``.
        The cavity-binary registration uses the derived brain mask, not
        the template intensity, so this only affects QC + fallbacks.
    template_path_override, annotation_path_override, label_table_override :
        Explicit paths that bypass resolution entirely.
    """
    atlas_dir: str = ''
    template_channel: str = 't2star'
    mode: str = 'cavity_binary'
    template_path_override: Optional[str] = None
    annotation_path_override: Optional[str] = None
    label_table_override: Optional[str] = None
    _structure_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if not self.atlas_dir:
            self.atlas_dir = os.environ.get(
                'VALIDATE29_DEST',
                os.path.expanduser('~/.cache/tuba/saimiri/atlas/validate29'))

    # -- low-level resolution -------------------------------------------------
    def _niftis(self):
        pats = ('*.nii', '*.nii.gz')
        out = []
        for root, _dirs, _files in os.walk(self.atlas_dir):
            for p in pats:
                out.extend(glob.glob(os.path.join(root, p)))
        return sorted(out)

    def _resolve_one(self, keywords, kind):
        """Return the single NIfTI whose basename matches any keyword.
        Raise if zero or ambiguous (>1) after keyword filtering."""
        cands = self._niftis()
        if not cands:
            raise FileNotFoundError(
                f'No NIfTI files under {self.atlas_dir!r}. '
                f'Run `python -m tuba.data.fetch_validate29` first.')
        kw = tuple(k.lower() for k in keywords)
        hits = [c for c in cands
                if any(k in os.path.basename(c).lower() for k in kw)]
        if len(hits) == 1:
            return hits[0]
        raise FileNotFoundError(
            f'Could not uniquely resolve the VALiDATe29 {kind} volume '
            f'(keywords {keywords}). Found {len(hits)} match(es): '
            f'{[os.path.basename(h) for h in hits]}. '
            f'All NIfTIs: {[os.path.basename(c) for c in cands]}. '
            f'Pass an explicit *_override or set the channel.')

    @property
    def template_path(self) -> str:
        if self.template_path_override:
            return self.template_path_override
        kw = {'t2star': ('t2star', 't2*', 't2_star'),
              'pd': ('pd', 'proton'),
              't1': ('t1',)}.get(self.template_channel, ('t2star',))
        return self._resolve_one(kw, f'{self.template_channel} template')

    @property
    def annotation_path(self) -> str:
        if self.annotation_path_override:
            return self.annotation_path_override
        return self._resolve_one(
            ('label', 'labels', 'seg', 'parcel', 'atlas', 'cortic'),
            'cortical-label')

    @property
    def brain_mask_path(self) -> str:
        return os.path.join(self.atlas_dir, 'validate29_brain_mask.nii.gz')

    @property
    def label_table_path(self) -> str:
        if self.label_table_override:
            return self.label_table_override
        for pat in ('*.label', '*.lut', '*.txt', '*.tsv', '*.csv'):
            hits = []
            for root, _d, _f in os.walk(self.atlas_dir):
                hits.extend(glob.glob(os.path.join(root, pat)))
            if hits:
                return sorted(hits)[0]
        raise FileNotFoundError(
            f'No label table (.label/.lut/.txt/.tsv/.csv) under '
            f'{self.atlas_dir!r}. Pass label_table_override.')

    # -- brain mask (fixed image for cavity_binary SyN) ----------------------
    def derive_brain_mask(self, force: bool = False, verbose: bool = True):
        """Binary brain mask from the cortical/white-matter label volume
        (any non-zero label is brain). Written to
        ``<atlas_dir>/validate29_brain_mask.nii.gz``.

        As with the WHS rat binding, we mask by ``label > 0`` rather than
        thresholding a template channel, since the anatomical templates
        carry soft-tissue/background that no fixed threshold cleanly
        separates from brain.
        """
        import ants
        if os.path.exists(self.brain_mask_path) and not force:
            return self.brain_mask_path
        if verbose:
            print(f'Deriving VALiDATe29 brain mask from {self.annotation_path}')
        a = ants.image_read(self.annotation_path)
        mask = (a.numpy() > 0).astype('float32')
        ants.image_write(a.new_image_like(mask), self.brain_mask_path)
        if verbose:
            vox_ml = abs(a.spacing[0]) * abs(a.spacing[1]) * abs(a.spacing[2]) / 1000.0
            print(f'  brain mask: {int(mask.sum())} vox = '
                  f'{int(mask.sum()) * vox_ml:.3f} mL -> {self.brain_mask_path}')
        return self.brain_mask_path

    # -- label table parsing --------------------------------------------------
    def _parse_label_table(self) -> dict:
        """Parse the atlas label table into ``name -> id``. Tolerates the
        two common formats: ITK-SNAP ``.label`` (``id r g b a vis mesh
        "name"``) and delimited ``id,name`` / ``id<tab>name`` tables."""
        if self._structure_cache:
            return self._structure_cache
        path = self.label_table_path
        out: dict = {}
        itk_re = re.compile(r'^\s*(\d+)\s+.*?"([^"]+)"\s*$')
        delim_re = re.compile(r'^\s*(\d+)[\s,;\t]+(.+?)\s*$')
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.rstrip()
                if not s or s.lstrip().startswith('#'):
                    continue
                m = itk_re.match(s) or delim_re.match(s)
                if not m:
                    continue
                idx, name = int(m.group(1)), m.group(2).strip().strip('"')
                if idx == 0:
                    continue
                out.setdefault(name, idx)
        if not out:
            raise ValueError(f'Parsed no label entries from {path!r}.')
        self._structure_cache.update(out)
        return self._structure_cache

    def structure_id(self, name: str) -> int:
        """Exact-name label lookup (case-sensitive to the atlas spelling)."""
        table = self._parse_label_table()
        if name not in table:
            hits = [k for k in table if name.lower() in k.lower()]
            raise KeyError(f'Structure {name!r} not in VALiDATe29 label table. '
                           f'similar: {hits[:6]}')
        return table[name]

    def all_structures(self) -> dict:
        return dict(self._parse_label_table())

    def resolve_label(self, target: str):
        """Map a pillar target key (``'S1_3b'`` / ``'S1_area1'`` / ``'M1'``)
        or a raw substring to an atlas ``(name, id)`` via
        :data:`TARGET_LABEL_KEYWORDS`. Returns the first matching label
        (lowest id on ties). Raises ``KeyError`` with the available cortical
        names if nothing matches -- so the exact atlas spelling can be pinned.
        """
        table = self._parse_label_table()
        keywords = TARGET_LABEL_KEYWORDS.get(target, (target,))
        kw = tuple(k.lower() for k in keywords)
        hits = sorted((idx, name) for name, idx in table.items()
                      if any(k in name.lower() for k in kw))
        if not hits:
            raise KeyError(
                f'No VALiDATe29 label matches target {target!r} '
                f'(keywords {keywords}). Cortical-like names present: '
                f'{[n for n in table if any(t in n.lower() for t in ("area", "cort", "s1", "m1"))][:12]}')
        idx, name = hits[0]
        return name, idx
