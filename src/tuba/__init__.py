"""TUBA — Transcranial Ultrasound Brain Atlas.

Unified skull-microCT + brain-atlas registration pipeline for mouse, macaque,
and human. Provides species-agnostic primitives in :mod:`tuba.core` and thin
species modules under :mod:`tuba.species` that bind numerics (voxel sizes,
bone thresholds, closing radii) to one or more atlases declared in
:mod:`tuba.atlases`.
"""

__version__ = "0.0.1"
