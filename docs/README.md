# TUBA Documentation

Validation-report manuscripts for the three species pipelines unified
by TUBA. Each subdirectory mirrors the document structure of the
others (LaTeX source + compiled PDF + figures) so downstream readers
can step between species and compare design choices section by
section.

## Layout

```
docs/
├── compactarticle.cls     shared compact-article class (symlinked from each subdir)
├── mouse/                 Maga 4K microCT + Allen CCFv3 (cavity_binary mode)
│   ├── manuscript.tex     17-page narrative with DigiMouse-vs-Maga focal comparison
│   ├── manuscript.pdf
│   ├── compactarticle.cls -> ../compactarticle.cls
│   ├── figures/           native-resolution validation figures
│   └── tables/            scan_params.tex, source_compare.tex
├── macaque/               AMNH macaque + NMT v2 (cavity_binary mode)
│   ├── manuscript.tex     16-page narrative with 9 figures + cavity-trials appendix
│   ├── manuscript.pdf
│   ├── compactarticle.cls -> ../compactarticle.cls
│   ├── make_atlas_figure.py            regenerates nmt_atlases.png (NMT-native)
│   ├── make_atlas_in_macaque_figure.py regenerates nmt_atlases_in_macaque.png
│   └── figures/           includes cavity_trials_panel.png ablation grid
├── human/                 Halle Zenodo + MNI152 (intensity mode)
│   ├── manuscript.tex     16-page narrative with 6 figures
│   ├── manuscript.pdf
│   ├── compactarticle.cls -> ../compactarticle.cls
│   ├── make_figures.py    regenerable: fig1–fig6 from registered NIfTIs
│   ├── warp_atlases.py    warps the 6 PARCELLATIONS atlases to Halle space
│   └── figures/           fig1_halle_native, fig2_mni_template, …, fig6_extra_parcellations_on_halle
└── rat/                   DigiMorph TMM M-2272 + Waxholm Space (Affine mode)
    ├── manuscript.tex     11-page narrative with 6 figures
    ├── manuscript.pdf
    ├── compactarticle.cls -> ../compactarticle.cls
    ├── make_figures.py    regenerable: intensity_histogram, bone_mass_profile,
    │                      whs_atlas, registration_qc (cavity_qc emitted by pipeline)
    └── figures/           native_orthoslices, world_orthoslices_240um,
                           intensity_histogram, bone_mass_profile, whs_atlas,
                           native_axial_anchors, cavity_qc, registration_qc
```

## Building a manuscript

Each `manuscript.tex` uses the shared `compactarticle.cls`. Build with:

```sh
cd docs/<species>
pdflatex manuscript
pdflatex manuscript    # second pass for cross-references
```

The `compactarticle.cls` symlink in each subdirectory points back to
`docs/compactarticle.cls` so the build is self-contained.

## Regenerating the human figures

```sh
# requires the warped MNI products + atlas caches; see ../README.md
# for the environment variables
cd docs/human
python warp_atlases.py    # produces 6 warped atlases in $TUBA_HUMAN_REG_DIR
python make_figures.py    # produces fig1–fig6 in figures/
pdflatex manuscript
pdflatex manuscript
```

## Section parity across species

| §  | Mouse                       | Macaque                     | Human                  | Rat                                |
|----|-----------------------------|-----------------------------|------------------------|------------------------------------|
| 1  | Source data                 | Source data                 | Source data            | Source data                        |
| 2  | Allen CCFv3 reference       | NMT v2 reference            | MNI152 + 6 parcellations | Waxholm Space v4.01 reference   |
| 3  | Orientation determination   | Orientation                 | Orientation (LAS-vs-LPS) | Orientation (+ `swap_row_col`)   |
| 4  | Downsampling                | Downsampling                | Downsampling           | Downsampling + skipped PCA         |
| 5  | PCA pre-alignment           | PCA pre-alignment           | No pre-alignment       | (folded into §4 — none needed)     |
| 6  | Cavity extraction           | Cavity (Affine NMT warp + bone clip) | Outer-bone surface (raycast + close+fill) | Cavity (Affine WHS warp + bone clip) |
| 7  | ANTs SyN to atlas           | ANTs **Affine** to atlas (SyN deforms soft-tissue features into mask) | ANTs SyN intensity + 6 warped atlases | ANTs **Affine** to atlas (reused §6 matrix) |
| 8  | Acoustic-property mapping   | Acoustic-property mapping   | Acoustic-property mapping | Acoustic-property mapping       |
| 9  | Focal-prediction comparison | Focal-prediction comparison | Placement (perp-to-skull) | Placement (+ slab loader)       |
| 10 | Extension to NHP            | Extension to clinical       | Slab loader            | Calibration trail (3 lessons)      |
| 11 | —                           | —                           | TUBA migration + 4 lessons | Open items (SHA pins, native cavity, focal sim) |

The structural parallelism is deliberate: when porting a fourth
species (marmoset, rat, etc.) one follows the same template,
replacing the species-specific numerical values and explaining any
genuinely new design choice in the final lessons section.

## Human-pipeline figure inventory

| LaTeX # | File | Content |
|---|---|---|
| Fig. 1 | `fig1_halle_native.png` | Halle dry-skull microCT at native 0.125 mm |
| Fig. 2 | `fig2_mni_template.png` | MNI T1 + Harvard-Oxford 117 + Schaefer 400 + Schaefer 1000 (3×4 grid) |
| Fig. 3 | `fig5_extra_parcellations_mni.png` | AAL3 + Pauli 2017 + Yeo 2011 in MNI space (3×3 grid) |
| Fig. 4 | `fig3_skull_atlas.png` | Post-SyN registration QC (Halle CT + warped MNI T1 + brain mask) |
| Fig. 5 | `fig4_warped_atlases_on_halle.png` | Halle bone + warped Harvard-Oxford / Schaefer 400 / Schaefer 1000 |
| Fig. 6 | `fig6_extra_parcellations_on_halle.png` | Halle bone + warped AAL3 / Pauli / Yeo |

LaTeX numbers figures by document order, which differs from the
file-name ordering — every cross-reference inside the PDF resolves
correctly via `\ref{fig:halle_native}` etc.

## Cavity-extraction lessons (carry over to a fourth species)

These lessons emerged from the macaque pipeline (six rejected approaches
documented as the trials table in `docs/macaque/manuscript.tex`,
§Endocranial cavity extraction). They apply to any species whose
cranium has multiple anatomical openings larger than a few mm
(orbits, foramen magnum, temporal fossa, choanae) — which is to say,
any vertebrate larger than a mouse.

1. **Don't deform a smooth atlas mask to fit a noisy cavity.**
   Non-rigid registration (ANTs SyN) propagates any cavity-side
   irregularity (incomplete bone closing, falx-cerebri midline
   indentation, soft-tissue features) into the warped mask. The
   Affine equivalent (12 DOF) cannot deform locally, so the warped
   mask retains its atlas topology.
2. **Bone-shell cavity is a registration target, not the production
   cavity.** A bone-shell with per-axial hull and curved floor/plugs
   is good enough to drive ANTs Affine but will always have
   internal-bone CC fragmentation. Use it to produce the alignment,
   then take the warped atlas mask as the production cavity.
3. **Bone-as-constraint, not bone-as-target.** After warping, clip
   the cavity against the species-specific bone mask
   (`cavity & ~bone_mask`). Sharpens the boundary at the inner
   cortical table without reintroducing CC fragmentation.
4. **Geodesic propagation captures too much.** Pneumatic spaces
   (sinuses, mastoid air cells, ear bullae) share narrow bone-bounded
   paths with the brain space and get included.
5. **Shrink-fit (Halle) is a fallback, not a default.** Halle-style
   uniform shrink + translation grid search is robust when an atlas
   mask does not exist or when bone-as-constraint is too thick. For
   species with a published brain atlas, the Affine warp produces a
   much closer fit.
6. **Affine is resolution-independent.** Fit at the cheapest
   resolution that captures the brain shape (250 µm for macaque),
   then upsample the warped mask to native (60 µm) for high-frequency
   acoustic-property mapping. Never re-fit the registration at
   native resolution; only re-rasterise.

The mouse pipeline (small skull, small foramina) gets away with pure
hysteresis-threshold-plus-closing (~1 mm closing seals all foramina);
the human pipeline (clinical CT, no atlas at hand) uses Halle
shrink-fit. The macaque sits between, and the Affine + atlas-mask +
bone-clip recipe is the one that works for both atlas-rich and
medium-foramen species. A marmoset/rat port should start with this
recipe.

## Citing the manuscripts

Each species manuscript is the canonical write-up for that pipeline.
Cite the one matching the application, plus the underlying atlas
(Allen CCFv3 / NMT v2 / MNI152 ICBM 2009a). Bibliography entries
live at the bottom of each `manuscript.tex`.
