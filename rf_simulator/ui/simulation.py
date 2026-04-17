"""
Simulation runner and profile plot update logic.

_run_sim is the hot path:
  1. Extract shape from active window
  2. Compute B1 peak and gradient
  3. Run Bloch simulation
  4. Update profile plots and status bar
"""

import numpy as np
from ..constants        import GAMMA_HZ_PER_T, GAMMA_RAD, FAT_WATER_PPM
from ..physics.bloch    import bloch_simulate
from ..physics.gradient import from_tbw, from_shape


def _schedule_sim(self) -> None:
    """Debounce: run simulation 25 ms after the last change."""
    self._sim_timer.start(25)


def _run_sim(self) -> None:
    """
    Core simulation step.

    dt is always rf_dur_ms / N — the window width only selects which
    canvas samples are fed into the sim, not how long they play for.
    """
    dur_s      = self.rf_dur_ms * 1e-3
    dt_s       = dur_s / self.N
    win_dur_ms = max(0.001, self.win_end_ms - self.win_start_ms)

    idx      = self._window_indices()
    shape_w  = self.rf_amp[idx].astype(np.float64)
    rf_phs_w = self.rf_phase[idx].astype(np.float64)

    # B1 peak — signed sum because negative lobes subtract from flip angle
    shape_sum_signed = shape_w.sum()
    shape_integral   = np.abs(shape_w).sum()
    flip_rad         = np.deg2rad(self.flip_deg)

    if abs(shape_sum_signed) > 1e-12:
        b1_peak_T = flip_rad / (GAMMA_RAD * dt_s * shape_sum_signed)
    elif shape_integral > 1e-12:
        b1_peak_T = flip_rad / (GAMMA_RAD * dt_s * shape_integral)
    else:
        b1_peak_T = 0.0

    b1_peak_uT = abs(b1_peak_T) * 1e6
    rf_rad     = shape_w * b1_peak_T * GAMMA_RAD * dt_s

    # Gradient
    if self._grad_from_shape:
        g_t_per_m, tbw_eff = from_shape(
            shape_w, self.flip_deg, self.rf_dur_ms, self.slice_mm, self.tbw
        )
    else:
        g_t_per_m, tbw_eff = from_tbw(self.tbw, self.rf_dur_ms, self.slice_mm)

    # Spatial axis
    fov_m   = self.slice_mm * 1e-3 * 6
    x_m     = np.linspace(-fov_m / 2, fov_m / 2, self.n_slices)
    x_mm    = x_m * 1e3
    offsets = GAMMA_HZ_PER_T * g_t_per_m * x_m

    # Chemical shift (fat-water, B0-dependent)
    chem_shift_hz = FAT_WATER_PPM * GAMMA_HZ_PER_T * self.b0_T
    chem_shift_mm = (
        chem_shift_hz / (GAMMA_HZ_PER_T * g_t_per_m * 1e-3)
        if g_t_per_m > 0 else 0.0
    )

    Mxy, Mz        = bloch_simulate(rf_rad, rf_phs_w, offsets, dt_s)
    mxy_mag        = np.clip(np.abs(Mxy), 0, 1)
    mxy_phs        = np.angle(Mxy)
    noise_floor    = max(mxy_mag.max() * 0.02, 0.01)
    mxy_phs_masked = np.where(mxy_mag >= noise_floor, mxy_phs, np.nan)

    # Update plots
    self.line_mxy.set_data(x_mm, mxy_mag)
    self.ax_mxy.set_xlim(x_mm[0], x_mm[-1])
    self.ax_mxy.set_ylim(-0.02, max(mxy_mag.max() * 1.1, 0.05))

    self.line_mz.set_data(x_mm, Mz)
    self.ax_mz.set_xlim(x_mm[0], x_mm[-1])
    self.ax_mz.set_ylim(min(-1.05, Mz.min() - 0.05), 1.05)

    self.line_phase.set_data(x_mm, mxy_phs_masked)
    self.ax_ph.set_xlim(x_mm[0], x_mm[-1])
    self.ax_ph.set_ylim(-np.pi - 0.2, np.pi + 0.2)

    self._draw_fwhm(x_mm, mxy_mag)
    self._draw_chem_shift(chem_shift_mm)
    self._update_status(
        x_mm, mxy_mag, Mz, rf_rad, dur_s,
        g_t_per_m, b1_peak_uT, win_dur_ms, tbw_eff, chem_shift_mm,
    )
    self.canvas.draw_idle()


# ── profile annotation helpers ─────────────────────────────────────────────────

def _draw_fwhm(self, x_mm: np.ndarray, mxy: np.ndarray) -> None:
    """Draw dashed FWHM marker lines on the |Mxy| plot."""
    peak = mxy.max()
    if peak < 0.01:
        for ln in self._fwhm_lines:
            ln.set_visible(False)
        return
    idx = np.where(mxy >= peak * 0.5)[0]
    if len(idx) >= 2:
        for ln, xv in zip(self._fwhm_lines, [x_mm[idx[0]], x_mm[idx[-1]]]):
            ln.set_xdata([xv, xv])
            ln.set_visible(True)
    else:
        for ln in self._fwhm_lines:
            ln.set_visible(False)


def _draw_chem_shift(self, chem_shift_mm: float) -> None:
    """Draw fat-water chemical shift marker on |Mxy|."""
    if abs(chem_shift_mm) < 0.01:
        self._chem_shift_line.set_visible(False)
        self._chem_shift_text.set_visible(False)
        return
    self._chem_shift_line.set_xdata([chem_shift_mm, chem_shift_mm])
    self._chem_shift_line.set_visible(True)
    self._chem_shift_text.set_x(chem_shift_mm + 0.1)
    self._chem_shift_text.set_text(f"Δfat {chem_shift_mm:.1f}mm")
    self._chem_shift_text.set_visible(True)


def _update_status(
    self, x_mm, mxy, Mz, rf_rad, dur_s,
    g_t_per_m, b1_peak_uT, win_dur_ms, tbw_eff, chem_shift_mm,
) -> None:
    """Update the status bar and computed-value labels."""
    peak       = mxy.max()
    flip_eff   = np.rad2deg(np.arcsin(np.clip(peak, 0, 1)))
    idx        = np.where(mxy >= peak * 0.5)[0]
    fwhm       = (x_mm[idx[-1]] - x_mm[idx[0]]) if len(idx) >= 2 else 0.0
    g_mt_per_m = g_t_per_m * 1e3

    self.lbl_grad_computed.setText(f"{g_mt_per_m:.2f}")
    self.lbl_b1_computed.setText(f"{b1_peak_uT:.2f}")

    tbw_display = (
        f"{tbw_eff:.1f} (from shape)" if self._grad_from_shape
        else f"{self.tbw} (from slider)"
    )
    larmor_mhz = GAMMA_HZ_PER_T * self.b0_T / 1e6

    self.status.showMessage(
        f"  B0={self.b0_T:.1f}T  f₀={larmor_mhz:.1f}MHz"
        f"  Δfat={chem_shift_mm:.1f}mm"
        f"   |   B1peak={b1_peak_uT:.2f}µT  G={g_mt_per_m:.2f}mT/m"
        f"   |   TBW={tbw_display}"
        f"   |   Peak |Mxy|: {peak:.3f}  Eff.flip: {flip_eff:.1f}°"
        f"   |   FWHM: {fwhm:.2f}mm (target: {self.slice_mm:.1f}mm)"
        f"   |   Mz(centre): {Mz[len(Mz) // 2]:.3f}"
    )
