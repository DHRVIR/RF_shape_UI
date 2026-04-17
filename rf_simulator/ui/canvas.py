"""
Canvas coordinate helpers, drawing stroke, window overlay, and commit logic.

All methods here are mixed into RFSimulator via the class definition in app.py.
They deal with the mapping between:
  - display time  (what the x-axis shows, may be offset after a left-edge crop)
  - array time    (raw index 0..canvas_N maps to 0..canvas_dur_ms)
  - canvas index  (integer position in rf_amp / rf_phase arrays)
"""

import numpy as np
from scipy.signal import savgol_filter


# ── coordinate helpers ─────────────────────────────────────────────────────────

def _resize_tol_ms(self) -> float:
    """
    Convert the fixed pixel snap-zone to data units using the current axis.
    Always _RESIZE_TOL_PX pixels on screen regardless of zoom or canvas size.
    """
    fig_w_px = self.canvas.get_width_height()[0]
    ax_frac  = self.ax_rf.get_position().width
    ax_w_px  = fig_w_px * ax_frac
    xl, xr   = self.ax_rf.get_xlim()
    if ax_w_px < 1:
        return 0.5
    return self._RESIZE_TOL_PX / ax_w_px * (xr - xl)


def _ms_to_ix(self, t_ms: float) -> int:
    """Array time (0..canvas_dur_ms) → canvas array index."""
    frac = t_ms / self.canvas_dur_ms
    return int(np.clip(frac * self._canvas_N, 0, self._canvas_N - 1))


def _display_ms_to_ix(self, t_ms_display: float) -> int:
    """Display time (may be offset) → canvas array index."""
    frac = (t_ms_display - self._canvas_x_offset_ms) / self.canvas_dur_ms
    return int(np.clip(frac * self._canvas_N, 0, self._canvas_N - 1))


def _to_display_ms(self, array_ms: float) -> float:
    """Array time → display axis time (adds offset)."""
    return array_ms + self._canvas_x_offset_ms


def _to_array_ms(self, display_ms: float) -> float:
    """Display axis time → array time (subtracts offset)."""
    return display_ms - self._canvas_x_offset_ms


def _window_indices(self) -> np.ndarray:
    """
    Resample canvas region [win_start_ms, win_end_ms] into N indices.
    Uses array-time coordinates (offset-independent).
    """
    i0  = self._ms_to_ix(self.win_start_ms)
    i1  = max(i0 + 1, self._ms_to_ix(self.win_end_ms))
    idx = np.round(np.linspace(i0, i1 - 1, self.N)).astype(int)
    return np.clip(idx, 0, self._canvas_N - 1)


def _event_to_ix(self, event):
    """Convert a matplotlib mouse event to (canvas_index, normalised_y)."""
    if event.inaxes is not self.ax_rf:
        return None, None
    yl, yu = self.ax_rf.get_ylim()
    ix     = self._display_ms_to_ix(event.xdata)
    fy     = float(np.clip((event.ydata - yl) / (yu - yl), 0.0, 1.0))
    return ix, fy


# ── drawing ────────────────────────────────────────────────────────────────────

def _apply_stroke(self, ix: int, fy: float, last_ix: int) -> None:
    """
    Paint a Gaussian brush stroke at canvas index ix.

    Blends the target value into surrounding samples weighted by a Gaussian
    so curves stay smooth even with jerky mouse movement.
    Brush radius is controlled by sl_brush (in canvas samples).
    """
    yl, yu   = self.ax_rf.get_ylim()
    radius   = max(1, self.sl_brush.value())
    is_amp   = (self._draw_mode == 'amp')
    target   = fy * (yu - yl) + yl if is_amp else (fy - 0.5) * 2 * np.pi

    i0     = last_ix if last_ix >= 0 else ix
    steps  = max(1, abs(ix - i0))
    points = np.round(np.linspace(i0, ix, steps + 1)).astype(int)

    r     = radius * 2
    kx    = np.arange(-r, r + 1)
    sigma = max(radius * 0.5, 0.5)
    gauss = np.exp(-0.5 * (kx / sigma) ** 2)
    gauss /= gauss.max()

    arr = self.rf_amp if is_amp else self.rf_phase
    for cx in points:
        for ki, k in enumerate(kx):
            si = int(cx + k)
            if 0 <= si < self._canvas_N:
                w      = gauss[ki]
                arr[si] = arr[si] * (1.0 - w) + target * w
    if is_amp:
        self.rf_amp   = arr
    else:
        self.rf_phase = arr


# ── smooth ─────────────────────────────────────────────────────────────────────

def _on_smooth(self) -> None:
    """Apply Savitzky-Golay smoothing to the whole canvas, preserving peak."""
    radius = max(1, self.sl_brush.value())
    win    = radius * 4 + 1
    if win % 2 == 0:
        win += 1
    win  = max(5, win)
    poly = min(3, win - 1)

    peak_before = np.abs(self.rf_amp).max()
    smoothed    = savgol_filter(self.rf_amp, win, poly)
    if peak_before > 1e-12:
        peak_after = np.abs(smoothed).max()
        if peak_after > 1e-12:
            self.rf_amp = smoothed * (peak_before / peak_after)
        else:
            self.rf_amp = smoothed

    if np.any(self.rf_phase != 0):
        self.rf_phase = savgol_filter(self.rf_phase, win, poly)

    self._grad_from_shape = True
    self._update_rf_plot()
    self._schedule_sim()


# ── window overlay ─────────────────────────────────────────────────────────────

def _update_window_overlay(self) -> None:
    """Redraw the orange window span and edge lines in display coordinates."""
    ws = self._to_display_ms(self.win_start_ms)
    we = self._to_display_ms(self.win_end_ms)
    self._win_span.remove()
    self._win_span = self.ax_rf.axvspan(ws, we, alpha=0.12, color="#ffa657", zorder=0)
    self._win_left.set_xdata([ws, ws])
    self._win_right.set_xdata([we, we])
    win_dur = max(self.win_end_ms - self.win_start_ms, 1e-6)
    self._win_label.set_x(ws + win_dur * 0.02)
    self._win_label.set_text(
        f"window: {self.win_start_ms:.2f} – {self.win_end_ms:.2f} ms"
        f"  |  sim duration = {self.rf_dur_ms:.2f} ms"
    )
    self.canvas.draw_idle()


# ── RF plot ────────────────────────────────────────────────────────────────────

def _update_rf_plot(self) -> None:
    """Redraw the RF waveform lines.  Negative x-offset reveals left-cut content."""
    x0   = self._canvas_x_offset_ms
    t_ms = np.linspace(x0, x0 + self.canvas_dur_ms, self._canvas_N)
    self.line_amp.set_data(t_ms, self.rf_amp)
    self.line_phs.set_data(t_ms, self.rf_phase)
    self.ax_rf.set_xlim(0, self.canvas_dur_ms)
    amax = max(np.abs(self.rf_amp).max(), 0.5)
    pmax = max(np.abs(self.rf_phase).max(), 0.2)
    top  = max(amax, pmax) * 1.15
    self.ax_rf.set_ylim(-top * 0.6, top)
    self._update_window_overlay()


# ── zoom helpers ───────────────────────────────────────────────────────────────

def _all_profile_axes(self):
    return (self.ax_mxy, self.ax_mz, self.ax_ph)


def _reset_zoom_rf(self) -> None:
    """Reset RF axis to show full canvas including any negative-time content."""
    x0 = self._canvas_x_offset_ms
    self.ax_rf.set_xlim(min(x0, 0), self.canvas_dur_ms)
    amax = max(np.abs(self.rf_amp).max(), 0.5)
    pmax = max(np.abs(self.rf_phase).max(), 0.2)
    top  = max(amax, pmax) * 1.15
    self.ax_rf.set_ylim(-top * 0.6, top)
    self.canvas.draw_idle()


def _reset_zoom_profiles(self) -> None:
    for ax in self._all_profile_axes():
        ax.autoscale()
    self.canvas.draw_idle()


# ── window commit ──────────────────────────────────────────────────────────────

def _commit_window_resize(self) -> None:
    """
    On mouse release after a window-edge drag:

    1. Scale the entire canvas so the selected window maps to 0 → rf_dur_ms.
    2. Content before win_start goes to negative time (hidden at normal zoom,
       visible on zoom-out left).
    3. Reset window to 0 → rf_dur_ms and recalculate everything.

    Scale = rf_dur_ms / win_dur_ms
    """
    snap_amp   = self._resize_amp_snap
    snap_phase = self._resize_phase_snap

    if snap_amp is None:
        self.win_start_ms        = 0.0
        self.win_end_ms          = self.rf_dur_ms
        self._canvas_x_offset_ms = 0.0
        self._update_rf_plot()
        self._schedule_sim()
        return

    win_dur_ms = max(0.001, self.win_end_ms - self.win_start_ms)
    scale      = self.rf_dur_ms / win_dur_ms

    t_left    = -self.win_start_ms * scale
    t_right   = t_left + self.canvas_dur_ms
    new_t_ms  = np.linspace(t_left, t_right, self._canvas_N)
    orig_t_ms = self.win_start_ms + new_t_ms / scale

    orig_frac        = orig_t_ms / self.canvas_dur_ms * (self._canvas_N - 1)
    in_bounds        = (orig_frac >= 0) & (orig_frac <= self._canvas_N - 1)
    orig_frac_c      = np.clip(orig_frac, 0, self._canvas_N - 1)

    new_amp   = np.interp(orig_frac_c, np.arange(self._canvas_N), snap_amp)
    new_phase = np.interp(orig_frac_c, np.arange(self._canvas_N), snap_phase)
    new_amp[~in_bounds]   = 0.0
    new_phase[~in_bounds] = 0.0

    self.rf_amp              = new_amp
    self.rf_phase            = new_phase
    self._canvas_x_offset_ms = t_left

    self.win_start_ms     = 0.0
    self.win_end_ms       = self.rf_dur_ms
    self._grad_from_shape = True

    self._update_rf_plot()
    self._schedule_sim()
