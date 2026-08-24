"""HU-calibrated acoustic-property mapping for skull FUS, with a hard
calibration guard.

This is the ``CT -> acoustic model`` stage (squirrel-monkey pillar
deliverable 1). It differs from :class:`tuba.core.slab.IntensityToAcousticRamp`
in one essential way: the ramp maps *arbitrary reconstruction intensity*
to ``(c, rho)`` by anchoring on a histogram mode, which is deliberately
robust to detector/reconstruction scale but is therefore **not**
quantitatively calibrated. The mapping here consumes **Hounsfield
units** and produces quantitative ``(rho, c, alpha)`` via a published
porosity model -- and it **refuses to run** on input that is not flagged
HU-calibrated, so an uncalibrated museum microCT can never silently
produce wrong acoustics.

Porosity model (Aubry et al. 2003)
----------------------------------
Aubry, J.-F., Tanter, M., Pernot, M., Thomas, J.-L., Fink, M. (2003).
"Experimental demonstration of noninvasive transskull adaptive focusing
based on prior computed tomography scans." *J. Acoust. Soc. Am.*
113(1):84-93. doi:10.1121/1.1529663.

Bone porosity is estimated from CT radiodensity,

    Phi(HU) = 1 - clip((HU - HU_water) / (HU_bone_max - HU_water), 0, 1)

(``Phi = 1`` water/marrow-filled pore space, ``Phi = 0`` fully dense
cortical bone), and the acoustic properties interpolate linearly in the
bone fraction ``(1 - Phi)``:

    rho(Phi) = rho_water + (rho_bone - rho_water) * (1 - Phi)
    c(Phi)   = c_water   + (c_bone   - c_water)   * (1 - Phi)

Attenuation is taken proportional to bone fraction between a soft-tissue
floor and a cortical-bone value (values: Pinton et al., "Attenuation,
scattering, and absorption of ultrasound in the skull bone," *Med. Phys.*
39(1):299-307, 2012; Pichardo et al., *Phys. Med. Biol.* 56:219-250,
2011). Attenuation is reported in dB/cm/MHz and is applied by the
downstream acoustic solver, not here.

SI units throughout (m, s, kg): density kg/m^3, sound speed m/s.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class UncalibratedInputError(RuntimeError):
    """Raised when a quantitative HU->properties mapping is asked to run
    on input that is not flagged Hounsfield-calibrated.

    The squirrel-monkey pillar is bootstrapped on an uncalibrated museum
    microCT (geometry only). Rather than silently emitting physically
    meaningless densities/sound-speeds, the mapping raises this so the
    caller must either (a) supply an HU-calibrated scan, or (b) opt in to
    the explicitly-flagged placeholder ramp
    (:func:`placeholder_ramp`)."""


@dataclass(frozen=True)
class CTCalibration:
    """Provenance + calibration state of a CT volume's intensity axis.

    Parameters
    ----------
    kind : {'hu', 'uncalibrated'}
        ``'hu'`` means the voxel values are Hounsfield units (a phantom
        or a clinical/energy-calibrated reconstruction). ``'uncalibrated'``
        means arbitrary reconstruction counts (typical of museum microCT:
        no HU phantom in the field of view).
    hu_water, hu_bone_max : float
        HU of water (0 by definition) and of fully-dense cortical bone
        (the porosity-model upper anchor). ``hu_bone_max`` should be set
        from the scan (e.g. its 99.9th-percentile HU) when known; the
        default 2000 HU is a cortical-bone nominal.
    source : str
        Free-text provenance for error messages / logs.
    """
    kind: str = 'uncalibrated'
    hu_water: float = 0.0
    hu_bone_max: float = 2000.0
    source: str = ''

    @property
    def is_calibrated(self) -> bool:
        return self.kind == 'hu'


# The canonical "do not trust these acoustics" calibration: a museum
# microCT staged for geometry only.
UNCALIBRATED_MICROCT = CTCalibration(
    kind='uncalibrated',
    source='museum microCT (arbitrary reconstruction units; no HU phantom)')


@dataclass
class AubryHUMapping:
    """Quantitative HU -> ``(rho, c, alpha)`` via the Aubry 2003 porosity
    model. Guarded: :meth:`properties` / :meth:`__call__` raise
    :class:`UncalibratedInputError` unless ``calibration.is_calibrated``.

    Defaults are cortical-bone / water endpoints consistent with the
    other TUBA pillars (c_bone 2900 m/s, rho_bone 1900 kg/m^3; the rat
    and macaque bindings use the same cortical endpoint).
    """
    calibration: CTCalibration
    rho_water: float = 1000.0     # kg/m^3
    rho_bone: float = 1900.0      # kg/m^3 (cortical; matches rat/macaque)
    c_water: float = 1500.0       # m/s
    c_bone: float = 2900.0        # m/s (cortical)
    alpha_soft_db_cm_mhz: float = 0.5    # soft tissue / water column
    alpha_bone_db_cm_mhz: float = 8.0    # cortical bone (Pinton 2012)

    def _require_calibrated(self):
        if not self.calibration.is_calibrated:
            raise UncalibratedInputError(
                'AubryHUMapping requires Hounsfield-calibrated input, but the '
                f'CT is flagged {self.calibration.kind!r} '
                f'({self.calibration.source!r}). Quantitative acoustics from '
                'uncalibrated reconstruction counts would be physically '
                'meaningless. Supply an HU-calibrated scan, or use the '
                'explicitly-flagged tuba.core.hu_acoustics.placeholder_ramp() '
                'for geometry-only work.')

    def porosity(self, hu):
        """Bone porosity ``Phi in [0, 1]`` from HU (guard applies)."""
        self._require_calibrated()
        hu = np.asarray(hu, dtype=np.float32)
        span = self.calibration.hu_bone_max - self.calibration.hu_water
        norm = np.clip((hu - self.calibration.hu_water) / span, 0.0, 1.0)
        return (1.0 - norm).astype(np.float32)

    def properties(self, hu):
        """Return ``(rho, c, alpha)`` in SI + dB/cm/MHz for HU input.

        Raises :class:`UncalibratedInputError` unless the bound
        calibration is Hounsfield.
        """
        phi = self.porosity(hu)                 # guard runs inside
        bone_frac = (1.0 - phi).astype(np.float32)
        rho = (self.rho_water + (self.rho_bone - self.rho_water) * bone_frac)
        c = (self.c_water + (self.c_bone - self.c_water) * bone_frac)
        alpha = (self.alpha_soft_db_cm_mhz
                 + (self.alpha_bone_db_cm_mhz - self.alpha_soft_db_cm_mhz)
                 * bone_frac)
        return (rho.astype(np.float32), c.astype(np.float32),
                alpha.astype(np.float32))

    def __call__(self, hu):
        """Slab-ramp-compatible adapter: return ``(c, rho)`` so a
        calibrated mapping can drop into
        :func:`tuba.core.slab.sample_slab_world_mm` as its ``ramp``.
        Matches :meth:`tuba.core.slab.IntensityToAcousticRamp.__call__`'s
        ``(c, rho)`` order. Guarded."""
        rho, c, _alpha = self.properties(hu)
        return c, rho

    # IntensityToAcousticRamp exposes ``i_low`` as the bone floor used by
    # the slab cavity-infill test (``hu >= ramp.i_low`` -> bone). Mirror
    # that so a calibrated mapping is a drop-in ramp.
    @property
    def i_low(self) -> float:
        """HU floor above which a voxel counts as bone for cavity infill
        (midpoint between water and cortical bone)."""
        return 0.5 * (self.calibration.hu_water + self.calibration.hu_bone_max)


def placeholder_ramp(i_low, i_high, *, reason='geometry-only (uncalibrated CT)'):
    """An explicitly-flagged, NON-calibrated intensity->(c,rho) ramp for
    uncalibrated scans.

    This is the sanctioned escape hatch when :class:`AubryHUMapping`
    refuses: it returns a :class:`tuba.core.slab.IntensityToAcousticRamp`
    carrying cortical-bone *nominal* endpoints so geometry (skull shell,
    beam path) can be visualized/sampled, while the ``.placeholder`` /
    ``.placeholder_reason`` attributes mark the output as not
    quantitatively calibrated. Downstream code that requires calibrated
    acoustics should check ``getattr(ramp, 'placeholder', False)``.
    """
    from .slab import IntensityToAcousticRamp
    ramp = IntensityToAcousticRamp(
        i_low=float(i_low), i_high=float(i_high),
        c_water=1500.0, rho_water=1000.0,
        c_bone_max=2900.0, rho_bone_max=1900.0)
    # Tag the instance so consumers can detect the uncalibrated origin.
    ramp.placeholder = True
    ramp.placeholder_reason = reason
    return ramp


# ---------------------------------------------------------------------------
# Points-per-wavelength report + grid-convergence (deliverable 1: explicit
# PPW check at the simulation frequencies + one convergence run).
# ---------------------------------------------------------------------------
def points_per_wavelength(dx_m, freqs_hz, c_min_m_s=1500.0):
    """Points per wavelength ``PPW = lambda / dx = c_min / (f * dx)`` at
    each frequency, using the **minimum** medium sound speed (the water
    column at ~1500 m/s), which gives the shortest wavelength and thus
    the most restrictive PPW.

    Returns a list of ``(freq_hz, lambda_m, ppw)`` tuples.
    """
    dx = float(dx_m)
    out = []
    for f in np.atleast_1d(freqs_hz).astype(float):
        lam = c_min_m_s / f
        out.append((float(f), float(lam), float(lam / dx)))
    return out


def report_ppw(dx_m, freqs_hz=(2e6, 3e6, 4e6), c_min_m_s=1500.0,
               ppw_target=6.0, verbose=True):
    """Print + return the PPW table at the simulation frequencies.

    ``ppw_target`` is the nominal FDTD floor (6 PPW is a common pseudo-
    spectral/low-dispersion target; classic 2nd-order FDTD wants more).
    Flags any frequency that falls below it.
    """
    rows = points_per_wavelength(dx_m, freqs_hz, c_min_m_s)
    if verbose:
        print(f'  PPW @ dx={dx_m*1e3:.4f} mm, c_min={c_min_m_s:.0f} m/s '
              f'(target >= {ppw_target:g}):')
        for f, lam, ppw in rows:
            flag = '' if ppw >= ppw_target else '  <-- UNDER-RESOLVED'
            print(f'    {f/1e6:>4.1f} MHz: lambda={lam*1e3:6.3f} mm  '
                  f'PPW={ppw:6.2f}{flag}')
    return rows


def time_of_flight_aberration_s(c_line_m_s, dx_m, c_ref_m_s=1500.0):
    """Skull-insertion time-of-flight aberration along a 1-D ray,

        tau = integral (1/c(x) - 1/c_ref) dx   [seconds],

    the quantity a time-reversal / phase-conjugation correction must
    undo. This is the physically-meaningful scalar whose grid
    convergence matters for transcranial targeting -- computed here on
    a sampled sound-speed profile ``c_line_m_s`` (1-D, m/s) at pitch
    ``dx_m``.
    """
    c = np.asarray(c_line_m_s, dtype=np.float64)
    return float(np.sum(1.0 / c - 1.0 / c_ref_m_s) * dx_m)


def grid_convergence_tof(c_profile_coarse, dx_coarse_m,
                         c_profile_fine, dx_fine_m,
                         c_ref_m_s=1500.0, verbose=True):
    """One grid-convergence run on the model discretization: compare the
    time-of-flight aberration (:func:`time_of_flight_aberration_s`)
    computed on a coarse vs. a refined sampling of the same beam ray.

    Returns ``dict(tau_coarse, tau_fine, abs_change_ns, rel_change)``.

    NOTE: this is the convergence of the **acoustic-property model** with
    voxel pitch, not a full-wave solver convergence (the wave solve lives
    in the fullwave2 siblings and is out of scope for this pillar). It is
    exactly the scalar -- transcranial phase aberration -- that the
    downstream time-reversal step will correct, so it is the right
    model-side convergence to report.
    """
    tau_c = time_of_flight_aberration_s(c_profile_coarse, dx_coarse_m, c_ref_m_s)
    tau_f = time_of_flight_aberration_s(c_profile_fine, dx_fine_m, c_ref_m_s)
    abs_change = abs(tau_f - tau_c)
    rel = abs_change / max(abs(tau_f), 1e-30)
    if verbose:
        print('  grid-convergence (time-of-flight aberration):')
        print(f'    coarse dx={dx_coarse_m*1e3:.4f} mm -> tau={tau_c*1e9:8.3f} ns')
        print(f'    fine   dx={dx_fine_m*1e3:.4f} mm -> tau={tau_f*1e9:8.3f} ns')
        print(f'    |change|={abs_change*1e9:.3f} ns  rel={rel*100:.2f}%')
    return {'tau_coarse_s': tau_c, 'tau_fine_s': tau_f,
            'abs_change_ns': abs_change * 1e9, 'rel_change': rel}
