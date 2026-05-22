"""Waxholm Space Sprague-Dawley rat brain atlas binding (WHS v4 / v4.01).

Atlas files (expected to be pre-cached by :mod:`tuba.data.fetch_whs`).
The v4 pack zip unpacks into ``<atlas_dir>/WHS_SD_rat_atlas_v4_pack/`` and
contains the T2* template + the v4 label volume + .label name table.
The v4.01 patch (Kleven et al. 2023, Nature Methods) lands flat in
``<atlas_dir>/`` as ``WHS_SD_rat_atlas_v4.01.nii.gz`` + ``.label``.

Layout::

  <atlas_dir>/
    WHS_SD_rat_atlas_v4_pack/
      WHS_SD_rat_T2star_v1.01.nii.gz   <- 39 um ex vivo T2* template
      WHS_SD_rat_atlas_v4.nii.gz       <- v4 labels (220 structures)
      WHS_SD_rat_atlas_v4.label        <- ITK-SNAP colormap + name table
      ...
    WHS_SD_rat_atlas_v4.01.nii.gz      <- v4.01 labels (222 structures)
    WHS_SD_rat_atlas_v4.01.label

Registration mode: ``"cavity_binary"``. The brain mask is derived from
the v4.01 annotation (any label > 0). Thresholding the T2* template
directly was rejected: the template carries substantial soft-tissue
and air background so any reasonable threshold (5--30% of max) gives
a 4--7 mL "brain mask" instead of the true ~2.5 mL. The
``brain_mask_threshold_frac`` field is retained for backward
compatibility but unused by ``derive_brain_mask``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Optional


# Structure-name -> integer label ID lookups, populated lazily from the
# v4.01 .label file. A handful of common TUS landmarks are listed here
# as expected name strings (the WHS .label file matches against name on
# parse, so these strings must match the atlas's canonical capitalization).
# Add more entries as needed once a target list is finalized.
_EXPECTED_STRUCTURE_NAMES = (
    'Corpus callosum and associated subcortical white matter',
    'Hippocampal formation',
    'Caudate putamen',
    'Thalamus',
    'Cerebral cortex',
    'Cerebellum',
    'Inferior colliculus',
    'Superior colliculus',
)


@dataclass
class WaxholmSpace:
    """Path-binding for a cached WHS rat atlas directory.

    Parameters
    ----------
    atlas_dir : str
        Directory containing the unpacked v4 pack subdir plus the
        v4.01 patch files. The :mod:`tuba.data.fetch_whs` defaults
        produce exactly this layout under
        ``~/.cache/tuba/rat/atlas/whs_sd_v4/``.
    pack_subdir : str
        Name of the v4 pack subdirectory (after unzip).
    brain_mask_threshold_frac : float
        Fraction of T2* template max used to derive a binary brain
        mask. The cavity-binary SyN uses this as the fixed image.
    """
    atlas_dir: str
    pack_subdir: str = 'WHS_SD_rat_atlas_v4_pack'
    brain_mask_threshold_frac: float = 0.05
    mode: str = 'cavity_binary'
    native_um: int = 39
    working_um: int = 39
    _structure_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def pack_dir(self) -> str:
        return os.path.join(self.atlas_dir, self.pack_subdir)

    @property
    def template_path(self) -> str:
        return os.path.join(self.pack_dir, 'WHS_SD_rat_T2star_v1.01.nii.gz')

    @property
    def annotation_v4_path(self) -> str:
        return os.path.join(self.pack_dir, 'WHS_SD_rat_atlas_v4.nii.gz')

    @property
    def annotation_path(self) -> str:
        """Default annotation = v4.01 patch (222 structures)."""
        return os.path.join(self.atlas_dir, 'WHS_SD_rat_atlas_v4.01.nii.gz')

    @property
    def label_table_path(self) -> str:
        """ITK-SNAP .label name table for the v4.01 annotation."""
        return os.path.join(self.atlas_dir, 'WHS_SD_rat_atlas_v4.01.label')

    @property
    def brain_mask_path(self) -> str:
        return os.path.join(self.atlas_dir, 'whs_brain_mask.nii.gz')

    def derive_brain_mask(self, force: bool = False, verbose: bool = True):
        """Derive a binary brain mask from the v4.01 annotation volume
        (any non-zero label is brain). Saved to
        ``<atlas_dir>/whs_brain_mask.nii.gz``.

        Note: do NOT threshold the T2* template directly -- it carries
        substantial soft-tissue + air background, so any reasonable
        threshold (5--30% of max) gives a 4--7 mL ``brain'' mask
        instead of the real ~2.5 mL volume. The v4.01 annotation
        encodes the curated brain extent (Kleven et al. 2023; 222
        labelled structures plus the 0 background), so masking by
        ``label > 0`` lands at ~2.46 mL -- the correct brain volume.
        """
        import ants
        import numpy as np
        if os.path.exists(self.brain_mask_path) and not force:
            return self.brain_mask_path
        if verbose:
            print(f'Deriving WHS brain mask from annotation '
                  f'{self.annotation_path}...')
        a = ants.image_read(self.annotation_path)
        arr = a.numpy()
        mask = (arr > 0).astype(np.float32)
        mask_img = a.new_image_like(mask)
        ants.image_write(mask_img, self.brain_mask_path)
        if verbose:
            print(f'  WHS brain mask: {int(mask.sum())} voxels '
                  f'= {int(mask.sum()) * abs(a.spacing[0])**3 / 1000:.3f} mL')
            print(f'  wrote {self.brain_mask_path}')
        return self.brain_mask_path

    def _parse_label_table(self) -> dict:
        """Parse the ITK-SNAP .label file into a ``name -> id`` dict.

        Format (one entry per line, # comments)::

            <id> <R> <G> <B> <A> <vis> <mesh> "<name>"
        """
        if self._structure_cache:
            return self._structure_cache
        path = self.label_table_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'WHS label table not found: {path}. '
                f'Run fetch_whs.py first.')
        out: dict = {}
        line_re = re.compile(r'^\s*(\d+)\s+.*?"([^"]+)"\s*$')
        with open(path) as f:
            for line in f:
                line = line.rstrip()
                if not line or line.lstrip().startswith('#'):
                    continue
                m = line_re.match(line)
                if not m:
                    continue
                idx, name = int(m.group(1)), m.group(2).strip()
                if idx == 0:
                    continue
                out[name] = idx
        self._structure_cache.update(out)
        return self._structure_cache

    def structure_id(self, name: str) -> int:
        """Look up a structure label ID by name. Names must match the
        v4.01 .label file canonical spelling (see ``_EXPECTED_STRUCTURE_NAMES``
        for a few common ones).
        """
        table = self._parse_label_table()
        if name not in table:
            hits = [k for k in table if name.lower() in k.lower()]
            hint = f'  similar: {hits[:5]}' if hits else ''
            raise KeyError(
                f'Structure {name!r} not in WHS v4.01 label table.{hint}')
        return table[name]

    def all_structures(self) -> dict:
        """Return the full ``name -> id`` dict from the v4.01 .label
        file (222 entries plus background)."""
        return dict(self._parse_label_table())
