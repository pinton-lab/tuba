"""Brain-atlas bindings for TUBA species modules.

Each binding declares:
* atlas paths (template, annotation, brain mask) for one or more
  resolutions;
* registration mode (``"cavity_binary"`` or ``"intensity"``);
* canonical structure-name -> ID lookups.

Fetcher scripts (download the atlas from upstream) are kept external
to TUBA -- each project tends to have its own template-cache layout.
The binding only points to existing files.
"""
