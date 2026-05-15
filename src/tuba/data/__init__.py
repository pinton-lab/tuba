"""Data-source manifest loader.

A single TOML file at :file:`tuba/data/sources.toml` is the source of
truth for every external dataset TUBA depends on. This module exposes
it as Python so downstream code can iterate or resolve a single entry
without re-parsing the TOML by hand.

Usage
-----
>>> from tuba.data import sources, get
>>> sources['mouse.skull']['citation']
'M. Maga and collaborators, ...'
>>> get('macaque.atlas_template').license
'open (NIMH/AFNI public)'

The :func:`get` helper returns a :class:`Source` dataclass with the
manifest fields as attributes; :data:`sources` is the raw nested dict
for cases that need to iterate or pattern-match on keys.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import tomllib                 # Python >= 3.11 stdlib
except ModuleNotFoundError:        # pragma: no cover
    import tomli as tomllib        # type: ignore[no-redef]

_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), 'sources.toml')


@dataclass
class Source:
    """Typed view of one manifest entry. All optional fields default
    to ``None`` so older manifests with fewer columns still parse."""
    key: str
    role: str
    description: str
    citation: str
    license: str
    auth_required: str
    format: str
    n_specimens: int
    doi: Optional[str] = None
    source_url: Optional[str] = None
    blob_urls: list[str] = field(default_factory=list)
    fetcher_script: Optional[str] = None
    voxel_um: Optional[float] = None
    voxel_mm: Optional[float] = None
    approx_size_gb: Optional[float] = None
    sha256: Optional[str] = None
    archive_sha256: Optional[str] = None
    archive_name: Optional[str] = None
    companion_files: list[dict] = field(default_factory=list)
    notes: Optional[str] = None

    @property
    def voxel_size_mm(self) -> Optional[float]:
        """Convenience accessor: returns voxel size in mm regardless of
        whether the manifest stored it as ``voxel_um`` or ``voxel_mm``."""
        if self.voxel_mm is not None:
            return self.voxel_mm
        if self.voxel_um is not None:
            return self.voxel_um / 1000.0
        return None


def _load_manifest() -> dict[str, dict[str, Any]]:
    """Parse the TOML manifest into a flat dict keyed by dotted path
    (e.g. ``"mouse.skull"``, ``"macaque.atlas_template"``)."""
    with open(_MANIFEST_PATH, 'rb') as f:
        raw = tomllib.load(f)
    flat: dict[str, dict[str, Any]] = {}
    for species, group in raw.items():
        for role_key, entry in group.items():
            flat[f'{species}.{role_key}'] = entry
    return flat


sources: dict[str, dict[str, Any]] = _load_manifest()
"""Flat dict of all manifest entries, keyed ``<species>.<role>``."""


_KNOWN_FIELDS = {f for f in Source.__dataclass_fields__ if f != 'key'}


def get(key: str) -> Source:
    """Resolve a manifest key into a :class:`Source` instance.

    Extra TOML keys not declared on :class:`Source` are silently
    dropped (so the manifest can grow new optional fields without
    breaking older library versions). Raises ``KeyError`` if the
    requested key is not in the manifest.
    """
    if key not in sources:
        raise KeyError(
            f'No data source registered as {key!r}. '
            f'Known keys: {sorted(sources)}')
    entry = {k: v for k, v in sources[key].items() if k in _KNOWN_FIELDS}
    return Source(key=key, **entry)


def integrity_lines() -> list[str]:
    """Return ``sha256sum -c`` compatible lines for every entry that
    has a pinned hash. Useful for one-shot verification of the local
    cache."""
    lines = []
    for k, e in sources.items():
        for field_name in ('sha256', 'archive_sha256'):
            h = e.get(field_name)
            if not h:
                continue
            name = (e.get('archive_name') if field_name == 'archive_sha256'
                    else e.get('description', k))
            lines.append(f'{h}  {name}    # {k}')
    return lines


def all_sources() -> list[Source]:
    """Return every manifest entry as a list of :class:`Source`."""
    return [get(k) for k in sources]


def by_species(species: str) -> list[Source]:
    """Return every entry whose dotted key starts with ``<species>.``."""
    return [get(k) for k in sources if k.startswith(f'{species}.')]


def by_role(role: str) -> list[Source]:
    """Return every entry whose ``role`` field equals ``role``."""
    return [get(k) for k in sources if sources[k]['role'] == role]
