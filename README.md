# TUBA — Transcranial Ultrasound Brain Atlas

A unified pipeline for registering skull microCT volumes to brain atlases
for transcranial focused-ultrasound (TUS) simulation, supporting mouse,
macaque, and human species under a single library.

```python
from tuba.species.mouse import place_on_skull, load_slab
placement = place_on_skull(target_name='CC_left_body', apex_to_target_mm=20.0)
c_map, rho_map, dz, frame = load_slab(
    placement['xdc_center_lps'], placement['beam_dir_3d'],
    slab_size_m=(12e-3, 12e-3, 15e-3), dx_target_m=29e-6,
)
```

The same call shape works for `tuba.species.macaque` and
`tuba.species.human`. The library handles orientation, downsampling,
cavity / outer-bone-surface extraction, ANTs SyN registration to the
species-appropriate atlas, intensity-to-acoustic-property mapping, and
beam-aligned slab sampling.

## What TUBA gives you

| Species | Atlas | Registration mode | Native pitch | Cached resolution |
|---|---|---|---|---|
| Mouse (Maga 4K dry skull) | Allen CCFv3 | cavity → brain-mask binary SyN | 6.127 µm | 25 µm |
| Macaque (AMNH dry skull) | NMT v2 + CHARM/SARM/D99 | cavity → brain-mask binary SyN | 60.6 µm | 250 µm |
| Human (Halle dry skull) | MNI152 ICBM 2009a | CT ↔ T1 intensity SyN | 0.125 mm | 1 mm |

Each species ships:
- a `tuba.species.<x>` module with the species-specific numerics
  (voxel size, bone thresholds, closing radii, intensity-ramp endpoints)
  plus thin wrappers over `tuba.core.{cavity, surface, slab,
  placement, warp}`,
- a `tuba.atlases.<atlas>` binding that points at the cached template
  + annotation files and exposes named target / EEG dicts (human: 52
  MNI anatomical targets + 22 EEG 10-20 sites),
- a `tuba.data.sources` manifest entry describing where the raw data
  comes from (DOI, license, fetcher script).

Six volumetric parcellations are bundled for the human pipeline
(Harvard-Oxford, Schaefer 400/1000, AAL3, Pauli 2017 subcortex,
Yeo-7 networks) — see [`docs/human/manuscript.pdf`](docs/human/manuscript.pdf).

## Install

```sh
pip install -e .
```

Optional extras:

```sh
pip install -e ".[viz]"   # nilearn + plotly for richer figures
pip install -e ".[dev]"   # pytest + ruff
```

Python ≥ 3.10. Core deps: numpy, scipy, nibabel, pynrrd, tifffile,
scikit-image, antspyx, matplotlib. `antspyx` is the heaviest install;
on Linux a `pip install antspyx` wheel covers x86_64. For ARM /
macOS use the upstream ANTsPy build instructions.

## Data — `TUBA_*_REG_DIR` env vars

TUBA never bundles raw skull microCT or atlas templates — those are
large, sometimes licensed, and each user is expected to stage them
locally. The library reads paths from environment variables, falling
back to `~/.cache/tuba/<species>/` if unset:

| Env var | Purpose |
|---|---|
| `TUBA_MOUSE_REG_DIR` | Mouse cache + Allen CCFv3 atlas dir |
| `TUBA_MOUSE_SOURCE_DIR` | Raw Maga 4K TIFF stack |
| `TUBA_HUMAN_REG_DIR` | Human cache + MNI152 warp transforms |
| `TUBA_HUMAN_NRRD_PATH` | Halle skull NRRD source |

The full data manifest lives in `src/tuba/data/sources.toml`, with one
entry per dataset (DOI, fetcher script, license, voxel size, expected
size). Load it programmatically:

```python
from tuba.data import all_sources, get
for s in all_sources():
    print(s.key, s.license, s.voxel_size_mm)
src = get('mouse.skull')
print(src.source_url, src.fetcher_script)
```

### Fetchers

Every dataset has a fetcher script in `src/tuba/data/fetch_*.py`.
Direct-fetch sources (Allen, NMT, Zenodo) run end-to-end without
interaction; click-through or login-required sources print
instructions and a drop location, then re-run to unpack.

| Fetcher | Hosts | Auth |
|---|---|---|
| `fetch_mouse.py` | Maga (UW) Box | click-through |
| `fetch_allen.py` | Allen Institute HTTP | none |
| `fetch_macaque.py` | MorphoSource (MCZ) | login + per-record approval |
| `fetch_nmt.py` | AFNI / NIMH HTTP | none |
| `fetch_human.py` | Zenodo (Kirchner 2022) | none (`HALLE_FETCH_ALL=1` for the ~10 GB projection archives) |

### Integrity

Distribution archives have SHA-256 sums pinned in the manifest
(`archive_sha256` for tarballs/zips, `sha256` for single files). Get
the canonical sum list from the manifest:

```python
from tuba.data import integrity_lines
for line in integrity_lines():
    print(line)
```

Pin a new sum when you stage a fresh dataset:

```sh
sha256sum dry_skull_4K_6.125micron_Rec.zip   # compute
# then update sources.toml: archive_sha256 = "..."
```

## Documentation

Per-species validation manuscripts live in [`docs/`](docs/):

- [`docs/mouse/manuscript.pdf`](docs/mouse/manuscript.pdf) —
  Migrating the 15 MHz transcranial mouse pipeline from DigiMouse to
  the Maga (UW) 4K microCT (18 pages, includes focal-prediction
  comparison vs the DigiMouse baseline).
- [`docs/macaque/manuscript.pdf`](docs/macaque/manuscript.pdf) —
  AMNH macaque microCT + NMT v2 registration (9 pages, in progress).
- [`docs/human/manuscript.pdf`](docs/human/manuscript.pdf) —
  MNI152 → Halle dry-skull microCT pipeline (16 pages, six figures,
  six parcellation atlases catalogued and warped).

Each manuscript follows the same section template (source data →
orientation → downsampling → pre-alignment → cavity/surface →
SyN → acoustic-property mapping → placement → slab → lessons).
See [`docs/README.md`](docs/README.md) for the section-parity table
across species.

## Tests

```sh
pytest tests/
```

The regression tests verify that the TUBA port produces outputs
bit-identical (or near-bit-identical, accounting for documented
mirror behaviour in the human case) to the legacy pipelines they were
ported from. They skip automatically if the legacy paths are not
locally present — set `TUBA_LEGACY_MOUSE_REG` and
`TUBA_LEGACY_HUMAN_REG` to run them.

## Citation

If TUBA is useful in your work, please cite the species manuscript
that matches your application (see `docs/<species>/manuscript.pdf`)
and the underlying atlas (Allen CCFv3 / NMT v2 / MNI152 ICBM 2009a).

## License

MIT — see [`LICENSE`](LICENSE).
