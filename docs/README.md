# TUBA Documentation

Validation-report manuscripts for the three species pipelines unified
by TUBA. Each subdirectory mirrors the document structure of the
others (LaTeX source + compiled PDF + figures) so downstream readers
can step between species and compare design choices section by
section.

## Layout

```
docs/
├── mouse/                 Maga 4K microCT + Allen CCFv3 (cavity_binary mode)
│   ├── manuscript.tex     18-page IOP-style narrative
│   ├── manuscript.pdf
│   ├── iopjournal.cls
│   └── figures/           native-resolution validation figures
├── macaque/               AMNH macaque + NMT v2 (cavity_binary mode, in progress)
│   ├── manuscript.tex     9-page draft
│   ├── manuscript.pdf
│   ├── iopjournal.cls
│   └── figures/
└── human/                 Halle Zenodo + MNI152 (intensity mode)
    ├── manuscript.tex     16-page narrative with 6 figures
    ├── manuscript.pdf
    ├── iopjournal.cls
    ├── make_figures.py    regenerable: fig1–fig6 from registered NIfTIs
    ├── warp_atlases.py    warps the 6 PARCELLATIONS atlases to Halle space
    └── figures/           fig1_halle_native, fig2_mni_template, …, fig6_extra_parcellations_on_halle
```

## Building a manuscript

Each `manuscript.tex` is an IOP-journal LaTeX document. Build with:

```sh
cd docs/<species>
pdflatex manuscript
pdflatex manuscript    # second pass for cross-references
```

The `iopjournal.cls` is co-located in each subdirectory so the build
is self-contained.

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

| §  | Mouse                       | Macaque                     | Human                  |
|----|-----------------------------|-----------------------------|------------------------|
| 1  | Source data                 | Source data                 | Source data            |
| 2  | Allen CCFv3 reference       | NMT v2 reference            | MNI152 reference + 6 parcellations |
| 3  | Orientation determination   | Orientation                 | Orientation (LAS-vs-LPS) |
| 4  | Downsampling                | Downsampling                | Downsampling           |
| 5  | PCA pre-alignment           | PCA pre-alignment           | No pre-alignment       |
| 6  | Cavity extraction           | Cavity extraction           | Outer-bone surface (raycast + close+fill) |
| 7  | ANTs SyN to atlas           | ANTs SyN to atlas           | ANTs SyN intensity + 6 warped atlases |
| 8  | Acoustic-property mapping   | Acoustic-property mapping   | Acoustic-property mapping |
| 9  | Focal-prediction comparison | Focal-prediction comparison | Placement (perpendicular-to-skull) |
| 10 | Extension to NHP            | Extension to clinical       | Slab loader            |
| 11 | —                           | —                           | TUBA migration + 4 lessons |

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

## Citing the manuscripts

Each species manuscript is the canonical write-up for that pipeline.
Cite the one matching the application, plus the underlying atlas
(Allen CCFv3 / NMT v2 / MNI152 ICBM 2009a). Bibliography entries
live at the bottom of each `manuscript.tex`.
