# `tuba.mini` — lightweight ITRUSST-skull demo pillar

A small, self-contained counterpart to the four full TUBA species
pipelines. Instead of a multi-GB subject skull microCT, it uses the
openly-hosted **ITRUSST benchmark skull** (two ~6 MB STL surfaces;
Aubry et al., *JASA* 2022, 500 kHz benchmark) and gives that
acoustics-only skull **brain atlasing** by registering the **MNI152**
template into its endocranial cavity — reusing TUBA's `cavity_binary`
SyN convention (the same one the mouse/macaque bindings use).

Nothing is committed to the repo. Both inputs are direct, unauthenticated
fetches into `$TUBA_MINI_DIR` (default `~/.cache/tuba/mini`) and the
nilearn cache; the total footprint is a few tens of MB.

## Why this skull

The ITRUSST benchmark skull is the one **real** skull that is small,
public, and permissively hosted — every full-TUBA skull is multi-GB
and/or login-gated. Its only gap is that it carries no brain; `tuba.mini`
closes that gap with a one-time MNI→cavity registration.

## Install & run

```sh
pip install -e ".[mini]"          # trimesh + nilearn; antspyx from core deps
python -m tuba.mini.demo          # fetch → rasterize → SyN → warp → QC figure
```

`antspyx` is needed only to **build** the transforms. Once the warped
products are cached (or shipped), consuming them is pure numpy/nibabel.

## Pipeline

```
fetch STLs + MNI152        tuba.mini.fetch
  → rasterize bone+cavity  tuba.mini.skull     1 mm RAS NIfTI
  → SyN cavity ↔ MNI mask   tuba.mini.register  (cavity_binary)
  → warp MNI brain mask + a parcellation onto the skull grid
  → named MNI targets (S1, thalamus, …) via ANATOMICAL_TARGETS_MNI
```

## Programmatic use

```python
from tuba.mini import itrusst
paths = itrusst.build(parcellation="harvard_oxford_117")   # one-time (antspyx)
# paths['bone'], paths['cavity'], paths['brain_in_skull'], paths['parc_in_skull']
```

## Licensing note

The ITRUSST STLs are permissively hosted; the demo warps the **MNI152
brain mask** (redistributable) by default. If you commit a *warped
parcellation* product, pick an openly-redistributable atlas
(AAL3 / Pauli CC0) rather than the non-commercial FSL Harvard-Oxford.
