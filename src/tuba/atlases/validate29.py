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
# is used.
#
# IMPORTANT -- atlas granularity (confirmed against the staged
# VALiDATe-labels.txt, 2026-08-18): VALiDATe29's histological cortical
# parcellation is *region-level*, not Brodmann-area-level. There is no
# separate ``area 3b`` / ``area 1`` / ``area 4`` label. In the squirrel
# monkey the primary somatosensory hand areas 3a/3b/1/2 are bundled into
# ``anterior_parietal_cortex`` (APC; l/r ids 11/12) and the primary
# motor cortex is ``primary_motor_cortex`` (M1; l/r ids 3/4). Left and
# right hemispheres are SEPARATE label ids (unlike WHS), so
# :meth:`resolve_label` is hemisphere-aware. Separating 3b from area 1
# within APC, or isolating the hand knob, needs a stereotaxic prior on
# the homunculus (not derivable from the distributed label volume).
TARGET_LABEL_KEYWORDS = {
    'S1': ('anterior_parietal', 'anterior parietal'),          # APC = S1 (3a/3b/1/2)
    'M1': ('primary_motor', 'primary motor'),                  # primary motor cortex
    'S2': ('parietal_ventral', 'secondary_somatosensory'),     # PV/S2
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
        Preferred in-vivo anatomical template channel for the QC
        background: one of ``'t2'`` (default), ``'t1'``, ``'pd'``.
        The cavity-binary registration uses the shipped brain mask, not
        the template intensity, so this only affects QC + fallbacks.
    template_path_override, annotation_path_override, label_table_override :
        Explicit paths that bypass resolution entirely.
    """
    atlas_dir: str = ''
    template_channel: str = 't2'
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
    @staticmethod
    def _is_junk(path):
        """macOS archive cruft: ``__MACOSX/`` dirs and ``._`` AppleDouble
        resource forks that ship inside zips made on a Mac (VALiDATe29.zip
        is one). These masquerade as real files and must be excluded."""
        b = os.path.basename(path)
        return '__MACOSX' in path.split(os.sep) or b.startswith('._')

    def _files_matching(self, patterns):
        out = []
        for root, _dirs, _files in os.walk(self.atlas_dir):
            for p in patterns:
                out.extend(glob.glob(os.path.join(root, p)))
        return sorted(f for f in out if not self._is_junk(f))

    def _niftis(self):
        return self._files_matching(('*.nii', '*.nii.gz'))

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
        # In-vivo anatomical channels shipped by VALiDATe29 are t2, t1, pd
        # (there is no t2star). Keywords are plain substrings matched
        # against the basename (e.g. 'VALiDATe22-t2.nii.gz').
        kw = {'t2': ('-t2', '_t2', 't2'),
              't1': ('-t1', '_t1', 't1'),
              'pd': ('-pd', '_pd', 'pd', 'proton')}.get(
                  self.template_channel, ('-t2', 't2'))
        return self._resolve_one(kw, f'{self.template_channel} template')

    @property
    def annotation_path(self) -> str:
        if self.annotation_path_override:
            return self.annotation_path_override
        # Only the discrete cortical/WM label volume, not the brain mask.
        cands = [c for c in self._niftis()
                 if 'label' in os.path.basename(c).lower()
                 and 'mask' not in os.path.basename(c).lower()]
        if len(cands) == 1:
            return cands[0]
        return self._resolve_one(('label', 'seg', 'parcel', 'dseg'),
                                 'cortical-label')

    @property
    def shipped_brain_mask_path(self) -> Optional[str]:
        """The atlas's own brain mask if it ships one (VALiDATe29 does:
        ``VALiDATe-brainmask.nii.gz``), else ``None``."""
        cands = [c for c in self._niftis()
                 if any(k in os.path.basename(c).lower()
                        for k in ('brainmask', 'brain_mask', 'brain-mask', 'mask'))]
        return cands[0] if len(cands) == 1 else (cands[0] if cands else None)

    @property
    def brain_mask_path(self) -> str:
        """Fixed image for the cavity_binary SyN. Prefer the shipped mask;
        otherwise the derive path under ``atlas_dir``."""
        shipped = self.shipped_brain_mask_path
        return shipped or os.path.join(self.atlas_dir,
                                       'validate29_brain_mask.nii.gz')

    @property
    def label_table_path(self) -> str:
        if self.label_table_override:
            return self.label_table_override
        for pat in ('*.label', '*.lut'):
            hits = self._files_matching((pat,))
            if hits:
                return hits[0]
        # Delimited text tables: prefer a name containing 'label', and
        # never pick LICENSE/README/NOTICE.
        txt = [h for h in self._files_matching(('*.txt', '*.tsv', '*.csv'))
               if not any(x in os.path.basename(h).lower()
                          for x in ('license', 'readme', 'notice', 'changelog'))]
        labelled = [h for h in txt if 'label' in os.path.basename(h).lower()]
        if labelled:
            return labelled[0]
        if txt:
            return txt[0]
        raise FileNotFoundError(
            f'No label table (.label/.lut/.txt/.tsv/.csv) under '
            f'{self.atlas_dir!r}. Pass label_table_override.')

    # -- brain mask (fixed image for cavity_binary SyN) ----------------------
    def derive_brain_mask(self, force: bool = False, verbose: bool = True):
        """Return the atlas brain mask (fixed image for the cavity_binary
        SyN). VALiDATe29 ships its own ``VALiDATe-brainmask.nii.gz``, so we
        use it directly (no ANTs needed). Only if no mask ships do we
        derive one from the label volume (``label > 0``), as the WHS rat
        binding does.
        """
        shipped = self.shipped_brain_mask_path
        if shipped:
            if verbose:
                print(f'  using shipped VALiDATe29 brain mask: {shipped}')
            return shipped
        import ants
        out = os.path.join(self.atlas_dir, 'validate29_brain_mask.nii.gz')
        if os.path.exists(out) and not force:
            return out
        if verbose:
            print(f'Deriving VALiDATe29 brain mask from {self.annotation_path}')
        a = ants.image_read(self.annotation_path)
        mask = (a.numpy() > 0).astype('float32')
        ants.image_write(a.new_image_like(mask), out)
        if verbose:
            vox_ml = abs(a.spacing[0]) * abs(a.spacing[1]) * abs(a.spacing[2]) / 1000.0
            print(f'  brain mask: {int(mask.sum())} vox = '
                  f'{int(mask.sum()) * vox_ml:.3f} mL -> {out}')
        return out

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

    def resolve_label(self, target: str, hemisphere: Optional[str] = None):
        """Map a pillar target key (``'S1'`` / ``'M1'`` / ``'S2'``) or a raw
        substring to an atlas ``(name, id)`` via
        :data:`TARGET_LABEL_KEYWORDS`.

        VALiDATe29 lateralises structures into separate ``l_``/``r_`` label
        ids, so pass ``hemisphere='left'|'right'`` to pick the matching
        hemisphere; otherwise the lowest matching id (typically left) wins.
        Raises ``KeyError`` listing the cortical names present if nothing
        matches -- so an unexpected atlas spelling surfaces loudly.
        """
        table = self._parse_label_table()
        keywords = TARGET_LABEL_KEYWORDS.get(target, (target,))
        kw = tuple(k.lower() for k in keywords)
        hits = sorted((idx, name) for name, idx in table.items()
                      if any(k in name.lower() for k in kw))
        if hemisphere in ('left', 'right'):
            pref = 'l_' if hemisphere == 'left' else 'r_'
            hemi = [(i, n) for i, n in hits if n.lower().startswith(pref)]
            if hemi:
                hits = hemi
        if not hits:
            cortical = [n for n in table
                        if any(t in n.lower()
                               for t in ('cortex', 'cort', 'parietal', 'motor',
                                         'somatosensory'))]
            raise KeyError(
                f'No VALiDATe29 label matches target {target!r} '
                f'(keywords {keywords}). Cortical names present: {cortical[:16]}')
        idx, name = hits[0]
        return name, idx
