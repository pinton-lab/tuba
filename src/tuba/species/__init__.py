"""Species modules: thin numerics + atlas-binding wrappers over
:mod:`tuba.core` primitives.

Currently implemented:

* :mod:`tuba.species.mouse` -- Maga (UW) 4K dry-skull microCT
  + Allen CCFv3 atlas. Bit-identical to the legacy
  ``mouse_therapy/registration/`` pipeline (4/4 regression PASS).
* :mod:`tuba.species.human` -- Halle Zenodo skull (Kirchner 2022)
  + MNI152 ICBM 2009a non-linear symmetric T1 atlas. Functionally
  ported (3/3 regression PASS, with a documented mirrored-equality
  assertion along the lateral axis stemming from the LAS->RAS
  reflection at NRRD load time). Skips PCA pre-alignment by design.
* :mod:`tuba.species.macaque` -- AMNH M-87264 dry-skull microCT
  (MorphoSource media 18083/18084, Delson Primate Scans /
  Copes 2016) + NMT v2.0 sym + CHARM / SARM / D99 parcellations.
  Initial port wraps the legacy ``monkey_brain/registration/``
  pipeline under :mod:`tuba._macaque`; constrained-PCA alignment
  (cranium-only, mandible removed via
  :mod:`tuba._macaque.isolate_cranium`) and Affine cavity
  extraction (no SyN). Regression test self-skips until
  ``TUBA_LEGACY_MACAQUE_REG`` is set.
* :mod:`tuba.species.rat` -- DigiMorph TMM M-2272 dry-skull microCT
  (29.6 um, UT Austin) + Waxholm Space SD rat atlas v4.01 (Kleven
  et al. 2023, 39 um T2*, 222 structures). Production pillar -- uses
  the macaque-style bone-shell cavity builder (refactored into
  :func:`tuba.core.cavity.bone_shell_cavity`) with rat-scale
  constants, median-smoothed per-row / per-column plug boundaries,
  and a concave-bowl virtual bone dam at the foramen magnum, then
  ANTs Affine to the WHS brain mask. See ``docs/rat/manuscript.pdf``
  for the calibration trail and ``tests/test_rat_skeleton.py``.
"""
