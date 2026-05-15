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

Planned:

* :mod:`tuba.species.macaque` -- MorphoSource MCZ Macaca mulatta
  microCT + NMT v2.0 sym + CHARM / SARM parcellations
  (in-progress upstream in :file:`monkey_brain/registration/`;
  port deferred until the alignment + cavity-extractor strategy
  settles).
"""
