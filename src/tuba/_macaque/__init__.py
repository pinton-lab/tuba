"""Internal package: macaque pipeline modules ported from the legacy
``monkey_brain/registration/`` layout. Public consumers should import
through :mod:`tuba.species.macaque`, which wraps these modules into the
species API used by the mouse + human ports.

Modules
-------
Constants and helpers
^^^^^^^^^^^^^^^^^^^^^
* :mod:`tuba._macaque.macaque_skull_constants` -- env-var-driven paths,
  PCA / threshold / alignment / pad / Affine output constants.

Orientation determination (manuscript Section 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* :mod:`tuba._macaque.orientation_probe` -- per-slice bone-mass +
  intensity histogram + mid-stack orthoslice renders from the raw
  TIFF stack.
* :mod:`tuba._macaque.orientation_zoom2` -- targeted brain-box
  orthoslice rebuild for D-V disambiguation.

PCA pre-alignment derivation (manuscript Section 5)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* :mod:`tuba._macaque.crop_raw_250um` -- cranium-CC isolation of the
  raw 250 um cache to feed the constrained-PCA derivation.
* :mod:`tuba._macaque.find_cranium_alignment` -- constrained PCA +
  basicranium-vs-vault asymmetry sweep; returns the ZYX intrinsic
  Euler angles used in :mod:`macaque_skull_constants`.

Refinement + isolation
^^^^^^^^^^^^^^^^^^^^^^
* :mod:`tuba._macaque.refine_native` -- world-frame rotation +
  translation refinement of the native 60.6 um cache; pigz-compressed
  output.
* :mod:`tuba._macaque.refine_250um` -- same refinement at 250 um for
  fast iteration; paired with ``refine_native``.
* :mod:`tuba._macaque.isolate_cranium` -- native-resolution mandible
  removal: keep largest bone CC, mask outside its bbox + 5 mm margin.
* :mod:`tuba._macaque.isolate_cranium_250um` -- same operation at
  250 um after refinement, for fast iteration.

Cavity extraction + atlas warp (manuscript Sections 6-7)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* :mod:`tuba._macaque.fast_iter_250um` -- bone-shell intermediate +
  ANTs Affine cavity-to-NMT-mask fit + production-cavity bone-clip
  (the production recipe).
* :mod:`tuba._macaque.warp_atlases_via_affine` -- pushes NMT T1 +
  brain mask + CHARM 5D + SARM 5D + D99 atlases back into macaque
  space via the saved Affine matrix.
* :mod:`tuba._macaque.native_cavity` -- memory-efficient NN upsample
  of the 250 um production cavity to the 60.6 um native grid + QC
  PNG render against the 250 um bone backdrop.
* :mod:`tuba._macaque.register_cavity_to_nmt` -- SyN alternative
  (manuscript Appendix Table 2 row 4); not used in production but
  preserved for reproducibility of the cavity-trials comparison.

Placement + slab loading (manuscript Section 8)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* :mod:`tuba._macaque.macaque_placement` -- apex / target / beam
  triple derivation from a CHARM-defined dACC target.
* :mod:`tuba._macaque.macaque_skull_slab_3d` -- slab loader returning
  (c, rho, dz, frame) along the beam axis.

QC figure generators
^^^^^^^^^^^^^^^^^^^^
* :mod:`tuba._macaque.registration_qc`,
  :mod:`tuba._macaque.cache_qc`,
  :mod:`tuba._macaque.rerender_native_cavity_qc`.
"""
