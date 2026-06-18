# Loading the Mouse Skull into an Acoustic Simulation (recipe)

A solver-agnostic guide for loading the tuba Maga-4K mouse skull into an acoustic
heterogeneous simulation (angular-spectrum, k-Wave, Fullwave, …). It describes the
**data, the CT → acoustic-property conversion, and the geometry conventions** —
you implement the loading in whatever solver you use.

The skull is the **Maga lab (UW) 4K dry-skull microCT** of one adult *Mus
musculus* (Bruker Skyscan 1272, **6.127 µm isotropic**, ~3882 axial uint16
TIFFs, dry skull only — no soft tissue or jaw). Intensities are raw NRecon
counts, **not** Hounsfield units.

---

## 1. Get the skull CT

Use the **tuba** repo's downloader (`github.com/.../tuba`):
1. Open <https://app.box.com/s/mtvctm7rus8udrav7qkrraq3syatvf5n>, click Download,
   save `dry_skull_4K_6.125micron_Rec.zip` into `~/.cache/tuba/mouse/source/`.
2. `python <tuba>/src/tuba/data/fetch_mouse.py` → unpacks the TIFF stack.

(Filenames say "12.5micron" but the scan log confirms 6.127 µm.)

## 2. Pre-process the volume

- **Downsample** to a working resolution (we use ~49 µm) with **block-MAX**, not
  mean — block-max preserves the thin cortical-bone shell through decimation.
- **Anatomically align** so your grid axes are the skull's principal axes. Take
  the cortical-bone cloud (voxels above a high count threshold, ~25 000), do a
  PCA, and rotate so: largest-extent axis → AP (snout↔occiput, ~23.8 mm),
  middle → LR (~15.9 mm), smallest → DV (~11.7 mm). Flip DV so the **dorsal
  calvarium is on the aperture side** (the densest voxels = teeth/palate mark the
  ventral end). (tuba's `core/align.py` does exactly this if you prefer to reuse it.)

## 3. CT intensity → acoustic properties

Piecewise-linear ramp on the raw counts `I` (anchored on the histogram, not HU):

```
t   = clip((I - 5000) / (25000 - 5000), 0, 1)      # 5000=water, 25000=cortical-bone mode
c   = 1540 + t*(2900 - 1540)     [m/s]
rho = 1000 + t*(1900 - 1000)     [kg/m^3]
```

**Attenuation α [dB/cm/MHz] — water-outside / tissue-inside / bone:**
- **Bone** (c above ~2000 m/s): α ≈ 8.
- **Soft tissue** = non-bone voxels *inside* the cranial cavity (between the
  dorsal calvarium and the skull base along the beam): α = 0.5.
- **Water/gel** = non-bone voxels *outside* the skull (the standoff above the
  calvarium, and below the base): α = 0 (lossless — your solver's water
  attenuation handles it). This is what makes the pre-skull field match a
  no-skull water run exactly.

The cranial cavity is "non-bone between the first and last bone along each beam
column"; everything before the first bone or after the last is water.

## 4. Feed into the ASM solver (phase screens, per propagation step dz)

```
phase     = 2*pi*f0 * dz * (1/c - 1/c0)
amplitude = exp(-alpha_Np_per_m * dz),   alpha_Np_per_m = alpha[dB/cm/MHz] * f_MHz * 0.1151/0.01
```
Apply `field *= amplitude .* exp(1i*phase)` at each step (c, alpha are the local
maps; c0 = 1540, water reference). Slices that are uniform water (phase≈0,
amplitude≈1) can be skipped.

## 5. Geometry / placement conventions we used

- **Propagation = DV axis**, beam enters the dorsal calvarium at **normal
  incidence** into the midbrain (no tilt).
- **Transducer**: the proper Vermon **Y-split** (the array is anisotropic —
  ~5.4 mm half-aperture separation along its long axis), **rotated 90°** so the
  two crossing plane-waves lie in the **AP-DV plane**. (If your array is square
  this is just an X-split; ours isn't, so the rotation matters.)
- **Standoff** (aperture→calvarium water gap): ~8.5 mm; the on-axis calvarium
  sits at the standoff.
- **Lateral placement**: centre the beam on the **midbrain** (the endocranial
  cavity centroid), not the bone-bounding-box centre — the long snout pulls the
  bbox centre too far anterior (~5 mm). A quick cavity extraction (close the
  foramina, region-grow, largest component; ~450 mm³ for an adult mouse) gives
  the centroid.

That's the whole setup. Same single-specimen skull every time.
