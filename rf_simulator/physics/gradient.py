"""
Slice-select gradient computation.

Two strategies are used depending on how the RF shape was produced:

  ``from_tbw``   — Analytic formula.  Used when a preset regenerated the shape
                   from the TBW slider.  Fast, exact, O(1).

  ``from_shape`` — Bisection.  Used when the shape was hand-drawn, window-cropped,
                   or smoothed.  Finds G such that Bloch-simulated FWHM = slice_mm.

Both return ``(G_t_per_m, tbw_eff)`` so callers always get the same type.
"""

import numpy as np
from ..constants import GAMMA_HZ_PER_T, GAMMA_RAD
from .bloch import bloch_simulate


def from_tbw(
    tbw:       float,
    rf_dur_ms: float,
    slice_mm:  float,
) -> tuple[float, float]:
    """
    Analytic gradient from the time-bandwidth product.

        G = TBW / (γ · T_RF · Δz)

    Parameters
    ----------
    tbw       : time-bandwidth product (from slider)
    rf_dur_ms : RF pulse duration [ms]
    slice_mm  : target slice thickness [mm]

    Returns
    -------
    (G [T/m], tbw_eff)   — tbw_eff equals tbw for this path
    """
    T_RF = rf_dur_ms * 1e-3
    dz   = slice_mm  * 1e-3
    G    = tbw / (GAMMA_HZ_PER_T * T_RF * dz)
    return G, float(tbw)


def from_shape(
    shape_w:   np.ndarray,
    flip_deg:  float,
    rf_dur_ms: float,
    slice_mm:  float,
    tbw_nominal: float,
) -> tuple[float, float]:
    """
    Bisection gradient: find G such that Bloch FWHM of |Mxy| = slice_mm.

    Uses a fixed FOV = 6 × slice_mm with adaptive point count so narrow
    slices at high G are always resolved.

    Parameters
    ----------
    shape_w      : normalised RF amplitude samples from the active window (N,)
    flip_deg     : target flip angle [degrees]
    rf_dur_ms    : RF pulse duration [ms]
    slice_mm     : target slice thickness [mm]
    tbw_nominal  : slider TBW — used only for the fallback return value

    Returns
    -------
    (G_opt [T/m], tbw_eff)
    """
    N    = len(shape_w)
    dt_s = rf_dur_ms * 1e-3 / N
    dz   = slice_mm  * 1e-3
    T_RF = rf_dur_ms * 1e-3

    shape_integral   = np.abs(shape_w).sum()
    shape_sum_signed = shape_w.sum()

    # Empty pulse — fall back to analytic formula
    if shape_integral < 1e-12:
        return from_tbw(tbw_nominal, rf_dur_ms, slice_mm)

    # Build B1-scaled RF waveform
    flip_rad = np.deg2rad(flip_deg)
    if abs(shape_sum_signed) > 1e-12:
        b1_T = flip_rad / (GAMMA_RAD * dt_s * shape_sum_signed)
    else:
        b1_T = flip_rad / (GAMMA_RAD * dt_s * shape_integral)

    rf_rad = (shape_w * b1_T * GAMMA_RAD * dt_s).astype(np.float64)
    rf_phs = np.zeros(N, dtype=np.float64)

    # Sanity check: pulse must produce non-trivial Mxy on resonance
    Mxy_test, _ = bloch_simulate(rf_rad, rf_phs, np.zeros(1, dtype=np.float64), dt_s)
    if np.clip(np.abs(Mxy_test[0]), 0.0, 1.0) < 0.01:
        return from_tbw(tbw_nominal, rf_dur_ms, slice_mm)

    # ── FWHM helper ────────────────────────────────────────────────────────────
    fov_m = dz * 6   # fixed spatial context for all G values

    def fwhm_for_G(G_tm: float) -> float:
        """Run a coarse Bloch profile and return FWHM [mm]."""
        # Adaptive point count: ensure narrow slices (high G) are resolved
        exp_w_m = max(1.0 / (GAMMA_HZ_PER_T * G_tm * T_RF), dz * 0.001)
        n_pts   = int(np.clip(fov_m / exp_w_m * 20, 128, 2048))
        x_m     = np.linspace(-fov_m / 2, fov_m / 2, n_pts)
        Mxy, _  = bloch_simulate(rf_rad, rf_phs, GAMMA_HZ_PER_T * G_tm * x_m, dt_s)
        mxy     = np.clip(np.abs(Mxy), 0.0, 1.0)
        pk      = mxy.max()
        if pk < 0.005:
            return fov_m * 1e3
        idx = np.where(mxy >= pk * 0.5)[0]
        if len(idx) < 2:
            return fov_m * 1e3
        return (x_m[idx[-1]] - x_m[idx[0]]) * 1e3

    # ── Bisection ──────────────────────────────────────────────────────────────
    G_lo, G_hi = 1e-5, 1.0
    fwhm_lo    = fwhm_for_G(G_lo)
    fwhm_hi    = fwhm_for_G(G_hi)

    # Bracket miss: fall back to analytic formula
    if fwhm_lo < slice_mm:
        return G_lo, G_lo * GAMMA_HZ_PER_T * T_RF * dz
    if fwhm_hi > slice_mm:
        return from_tbw(tbw_nominal, rf_dur_ms, slice_mm)

    for _ in range(24):
        G_mid = (G_lo + G_hi) / 2.0
        if fwhm_for_G(G_mid) > slice_mm:
            G_lo = G_mid
        else:
            G_hi = G_mid
        if abs(G_hi - G_lo) / max(G_hi, 1e-9) < 1e-5:
            break

    G_opt   = (G_lo + G_hi) / 2.0
    tbw_eff = G_opt * GAMMA_HZ_PER_T * T_RF * dz
    return G_opt, tbw_eff
