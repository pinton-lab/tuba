"""``tuba.mini`` -- a lightweight, self-contained TUBA demo pillar.

The four full TUBA species pipelines each register a multi-GB skull
microCT to a species brain atlas. ``tuba.mini`` is the small, fully
fetchable counterpart: it takes the openly-hosted **ITRUSST benchmark
skull** (two ~6 MB STL surfaces; Aubry et al., JASA 2022) -- which is
acoustics-only, with no brain -- and gives it TUBA atlasing by
registering the **MNI152** template into its endocranial cavity via the
same ``cavity_binary`` SyN convention the mouse/macaque bindings use.

No scans are committed and nothing here exceeds a few tens of MB: the
STLs and MNI152 template are fetched on demand into ``$TUBA_MINI_DIR``
(default ``~/.cache/tuba/mini``) and the nilearn cache.

Quick start
-----------
    from tuba.mini import itrusst
    itrusst.build()                     # needs antspyx (one-time)
    # -> writes bone/cavity NIfTIs + MNI brain mask & parcellation
    #    warped onto the skull grid, all under $TUBA_MINI_DIR

Run ``python -m tuba.mini.demo`` for an end-to-end build + QC figure.

Submodules
----------
fetch     stage the ITRUSST STLs + MNI152 template (direct, no auth)
skull     rasterize the STLs into aligned bone + cavity voxel masks
register  cavity <-> MNI SyN and warp MNI volumes onto the skull grid
itrusst   the binding tying it together (paths, build, targets)
demo      end-to-end runnable + QC overlay figure
"""

__all__ = ['fetch', 'skull', 'register', 'itrusst', 'demo']
