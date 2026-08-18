# TUBA — squirrel monkey (*Saimiri sciureus*) pillar

NIH R01 EB037345, Aim 3. Per-animal squirrel-monkey skull models for
transcranial focused-ultrasound targeting of the somatosensory/motor
circuit (fUS/srfUS imaging + FUS neuromodulation).

**Status: geometry-only (deliverables D1–D2).** This pillar is
bootstrapped on an *uncalibrated museum microCT* (there is no colony CT
yet), so it does the geometry — skull segmentation, endocranial cavity,
VALiDATe29 atlas registration, S1/M1 target export — but **not**
quantitative acoustics: the HU→properties mapping refuses to run on the
uncalibrated scan (see below). The downstream wave physics (D3–D7) is
out of scope for this branch and lives in the `fullwave2-*` solver
siblings.

## Data (not committed — fetched on demand)

| Role | Dataset | Source | License |
|------|---------|--------|---------|
| skull | *Saimiri sciureus* dry-skull microCT, USNM 338948, 0.1189 mm iso | UTCT / DigiMorph (oVert mirror on MorphoSource) | interactive-only (educational/research) |
| atlas | VALiDATe29 multi-channel squirrel-monkey MRI atlas (29 animals) | NITRC ([validate29](https://www.nitrc.org/projects/validate29/)) | CC BY |

```bash
python -m tuba.data.fetch_saimiri      # prints staging instructions (request/login)
python -m tuba.data.fetch_validate29   # CC-BY direct fetch from NITRC
```

Both land under `~/.cache/tuba/saimiri/` (override with `SAIMIRI_SOURCE_DIR`,
`VALIDATE29_DEST`). Manifest rows: `saimiri.{skull,atlas_template,atlas_annotation}`
in `src/tuba/data/sources.toml`.

## Deliverable 1 — CT → acoustic model, with a calibration guard

`tuba.core.hu_acoustics` implements the Aubry (2003) porosity HU→(ρ, c, α)
model. It is **guarded**: `AubryHUMapping` raises `UncalibratedInputError`
unless the bound `CTCalibration.kind == 'hu'`. Because a museum microCT
carries no HU phantom, the pillar flags it `UNCALIBRATED_MICROCT` and the
slab loader falls back to an explicitly-tagged `placeholder_ramp`
(`.placeholder is True`) — geometry is faithful, absolute (c, ρ) are
nominal. Swap in an HU-calibrated colony CT by setting
`saimiri.CT_CALIBRATION` and the quantitative path unlocks automatically.

The report prints the **points-per-wavelength** table on the simulation
grid (60 µm) at the 2/3/4 MHz operating band and — given a placement and a
staged scan — one **grid-convergence run** of the transcranial
time-of-flight aberration (the quantity a time-reversal correction must
undo):

```bash
python -m tuba.species.saimiri report   # PPW table (no data needed)
```

```
2.0 MHz: λ=0.750 mm  PPW=12.50
3.0 MHz: λ=0.500 mm  PPW= 8.33
4.0 MHz: λ=0.375 mm  PPW= 6.25   (≥ 6-PPW floor)
```

## Deliverable 2 — VALiDATe29 registration + S1/M1 targets + QC

Cavity `cavity_binary` SyN (affine + deformable) registers the subject
endocranial cavity to the VALiDATe29 brain mask; the template + cortical
labels are warped back into subject space. `export_targets()` writes the
S1 (areas 3b, 1) and M1 (area 4) hand-representation coordinates in the
subject simulation frame, and `qc_figure()` writes the orthoslice overlay.

```bash
python -m tuba.species.saimiri build    # ingest → cavity → SyN → targets → QC → report
```

Cortical label spellings vary across VALiDATe29 releases, so
`tuba.atlases.validate29.resolve_label` matches stable area-number
substrings (`area 3b`, `area 1`, `area 4`) rather than a full canonical
name; confirm against the staged LUT.

## Provisional constants

Orientation flips, intensity thresholds, and the cavity hull geometry in
`tuba.species.saimiri` are marked `PROVISIONAL`: like every pillar they
are fixed by an orientation/histogram probe on the *staged* scan. They
carry rat/macaque-scaled defaults so the pipeline runs the moment the
scan lands, but must be re-confirmed before any result is trusted.

## Out of scope here (D3–D7, solver siblings)

Time-reversal aberration correction, multifocal (two-foci) beam
synthesis, the 1.5–4.0 mm × 2/3/4 MHz feasibility sweep, and MI/bioheat
safety margins consume the `(c, ρ, dz)` slab this pillar exports and run
in the `fullwave2-*` FDTD siblings.
