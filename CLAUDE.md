# Notes for agents working in TUBA

## The CT / atlas data is NOT in this repo — do not look for it here, do not commit it
TUBA ships **code + a data manifest (`src/tuba/data/sources.toml`) + fetchers
(`src/tuba/data/fetch_*.py`)** — **not** the skull microCT volumes or atlas templates.
Those are multi‑GB and often licensed, so they are fetched/staged to local caches:

- `~/.cache/tuba/<species>/{source,atlas,registration}/` — scans, atlases, warp transforms
- `~/nilearn_data/` — human MNI parcellations (Harvard‑Oxford, Schaefer, AAL3, Pauli, Yeo, Diedrichsen)

**If a data path is missing:** run its fetcher (`python -m tuba.data.fetch_<name>`) or read
`sources.toml` — do **not** conclude the pipeline is broken, and do **not** synthesize or
re‑create the data. The species bindings read paths from `TUBA_*_REG_DIR` /
`TUBA_*_SOURCE_DIR` env vars, falling back to the caches above. The repo's root `data/` is
gitignored entirely (`.gitignore`: `/data/`) — nothing under it is tracked, so it will not
exist in a fresh clone. Create it locally if you want a staging dir there; never commit it.

## Validating registration
The MNI↔subject‑skull registration is validated‑correct; check it against the repo's own QC
(`docs/human/figures/fig3_skull_atlas.png` + the human manuscript), **not** ad‑hoc warp
figures. The Halle skull is a *dry* skull (empty cranial cavity, no native brain), so to *see*
the atlas in the skull you must warp the MNI T1 in and **mask** it.

## General
- Don't add large binaries (scans, `.f32`/`.dat` field dumps) to the repo.
- Per-machine data locations and project history live in the session memory, not here.
