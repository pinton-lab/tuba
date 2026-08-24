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
| skull | *Saimiri* sp. dry-skull microCT, NMNH USNM 194346 (MorphoSource media 000116521), 490 slices, 0.0977 mm in-plane x 0.1189 mm slice (anisotropic) | UTCT / DigiMorph (oVert mirror on MorphoSource) | interactive-only (per-record approval) |
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
S1 (anterior parietal cortex) and M1 (primary motor cortex) coordinates in
the subject simulation frame, and `qc_figure()` writes the orthoslice
overlay.

```bash
python -m tuba.species.saimiri build    # ingest → cavity → SyN → targets → QC → report
```

VALiDATe29's parcellation is **region-level, not Brodmann-area-level** —
there is no `area 3b` / `area 1` / `area 4` label. The S1 hand areas
3a/3b/1/2 are bundled into `anterior_parietal_cortex` (APC, l/r ids
11/12) and M1 is `primary_motor_cortex` (l/r ids 3/4); PV/S2 is ids 7/8.
Left and right are separate ids, so
`tuba.atlases.validate29.resolve_label` is hemisphere-aware and matches
region keywords (`anterior_parietal`, `primary_motor`, `parietal_ventral`)
rather than area numbers. Splitting 3b from area 1, or isolating the hand
knob, needs a stereotaxic prior that is not in the distributed volume.

The label volume is also **not a whole-brain parcellation**: 80 distinct
ids fill 45.1% of the shipped brain mask, and the gray matter is nine
bilateral cortical regions and nothing else — no subcortical gray, no
cerebellum, no brainstem. Enough for the S1/M1 aim; see the manuscript
§atlas for the full coverage table.

## Scan-dependent constants (calibrated)

Orientation flips, intensity thresholds, and the cavity hull geometry in
`tuba.species.saimiri` are **calibrated against the staged scan**
(USNM 194346), not provisional: the rat/macaque museum-CT orientation
convention was confirmed clean RAS by an orthoslice probe, and the bone
thresholds are set from the histogram.

One scan-specific note: the field of view ends flush against the occiput
(real bone runs to slice 593 of 596, ~0.3 mm from the caudal wall), which
leaves the cavity extractor's morphology no headroom and truncates the
occipital pole. `downsample_to_working` therefore pads the raw stack axis
by `AP_PAD_SLICES = 30` (~3.6 mm) at each end before isotropisation. No
bone is invented — the padding only restores empty margin.

## Out of scope here (D3–D7, solver siblings)

Time-reversal aberration correction, multifocal (two-foci) beam
synthesis, the 1.5–4.0 mm × 2/3/4 MHz feasibility sweep, and MI/bioheat
safety margins consume the `(c, ρ, dz)` slab this pillar exports and run
in the `fullwave2-*` FDTD siblings.
