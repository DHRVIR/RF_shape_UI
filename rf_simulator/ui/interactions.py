"""
Mouse and scroll event handlers for the RF canvas.

All methods are mixed into RFSimulator.  They handle:
  - Right-click drag  → zoom (X on all axes, Y on RF axis)
  - Right double-click → reset zoom
  - Left-click near window edge → resize window
  - Left-click + drag in draw mode → paint stroke
  - Middle-click / Shift-pulse mode → slide waveform
  - Scroll wheel → x-zoom centred on cursor
"""

import numpy as np
from PyQt5.QtCore import Qt


def _on_press(self, event) -> None:
    if event.inaxes is None:
        return

    # ── right-click ────────────────────────────────────────────────────────────
    if event.button == 3:
        if event.dblclick:
            if event.inaxes is self.ax_rf:
                self._reset_zoom_rf()
            elif event.inaxes in self._all_profile_axes():
                self._reset_zoom_profiles()
            return
        self._zoom_active    = True
        self._zoom_ax        = event.inaxes
        self._zoom_start_px  = (event.x, event.y)
        ax = event.inaxes
        self._zoom_start_lim = (*ax.get_xlim(), *ax.get_ylim())
        return

    # ── left-click near window edge → start resize ────────────────────────────
    if event.button == 1 and event.inaxes is self.ax_rf and event.xdata is not None:
        tol            = self._resize_tol_ms()
        win_start_disp = self._to_display_ms(self.win_start_ms)
        win_end_disp   = self._to_display_ms(self.win_end_ms)
        near_right     = abs(event.xdata - win_end_disp)   <= tol
        near_left      = abs(event.xdata - win_start_disp) <= tol

        if near_right and near_left:
            if abs(event.xdata - win_end_disp) <= abs(event.xdata - win_start_disp):
                self._resize_right = True
            else:
                self._resize_left  = True
        elif near_right:
            self._resize_right = True
        elif near_left:
            self._resize_left  = True

        if self._resize_right or self._resize_left:
            self._resize_amp_snap   = self.rf_amp.copy()
            self._resize_phase_snap = self.rf_phase.copy()
            self._grad_from_shape   = True
            return

    # ── middle-click or slide mode → shift waveform ───────────────────────────
    if event.inaxes is self.ax_rf and event.xdata is not None:
        if event.button == 2 or (event.button == 1 and self._draw_mode == 'slide'):
            self._slide_active     = True
            self._slide_drag_x0    = event.xdata
            self._slide_amp_snap   = self.rf_amp.copy()
            self._slide_phase_snap = self.rf_phase.copy()
            return

    # ── left-click in draw mode → paint ───────────────────────────────────────
    if event.button == 1 and event.inaxes is self.ax_rf and self._draw_mode != 'slide':
        self._drawing         = True
        self._grad_from_shape = True
        ix, fy = self._event_to_ix(event)
        if ix is not None:
            self._last_ix = ix
            self._apply_stroke(ix, fy, -1)
            self._update_rf_plot()
            self._schedule_sim()


def _on_move(self, event) -> None:
    # ── cursor feedback ────────────────────────────────────────────────────────
    if (event.inaxes is self.ax_rf and event.xdata is not None
            and not self._slide_active and not self._zoom_active
            and not self._resize_left  and not self._resize_right):
        tol            = self._resize_tol_ms()
        win_start_disp = self._to_display_ms(self.win_start_ms)
        win_end_disp   = self._to_display_ms(self.win_end_ms)
        if (abs(event.xdata - win_end_disp)   <= tol or
                abs(event.xdata - win_start_disp) <= tol):
            self.canvas.setCursor(Qt.SplitHCursor)
        elif self._draw_mode == 'slide':
            self.canvas.setCursor(Qt.SizeHorCursor)
        else:
            self.canvas.setCursor(Qt.CrossCursor)

    # ── right-edge resize drag ─────────────────────────────────────────────────
    if self._resize_right and event.xdata is not None:
        min_win = self.canvas_dur_ms * 0.005
        new_end = np.clip(
            self._to_array_ms(event.xdata),
            self.win_start_ms + min_win,
            self.canvas_dur_ms,
        )
        self.win_end_ms = new_end
        self._update_window_overlay()
        self._schedule_sim()
        return

    # ── left-edge resize drag ──────────────────────────────────────────────────
    if self._resize_left and event.xdata is not None:
        min_win   = self.canvas_dur_ms * 0.005
        new_start = np.clip(
            self._to_array_ms(event.xdata),
            0.0,
            self.win_end_ms - min_win,
        )
        self.win_start_ms = new_start
        self._update_window_overlay()
        self._schedule_sim()
        return

    # ── waveform slide drag ────────────────────────────────────────────────────
    if self._slide_active and event.xdata is not None:
        delta_ms  = event.xdata - self._slide_drag_x0
        delta_smp = int(round(delta_ms / self.canvas_dur_ms * self._canvas_N))
        self.rf_amp   = np.roll(self._slide_amp_snap,   delta_smp)
        self.rf_phase = np.roll(self._slide_phase_snap, delta_smp)
        if delta_smp > 0:
            self.rf_amp[:delta_smp]   = 0.0
            self.rf_phase[:delta_smp] = 0.0
        elif delta_smp < 0:
            self.rf_amp[delta_smp:]   = 0.0
            self.rf_phase[delta_smp:] = 0.0
        self._update_rf_plot()
        self._schedule_sim()
        return

    # ── zoom drag ──────────────────────────────────────────────────────────────
    if self._zoom_active and self._zoom_ax is not None:
        ax              = self._zoom_ax
        xl0, xr0, yl0, yu0 = self._zoom_start_lim
        dx_px = event.x - self._zoom_start_px[0]
        dy_px = event.y - self._zoom_start_px[1]

        SENS     = 200.0
        x_factor = 2.0 ** (-dx_px / SENS)
        x_center = (xl0 + xr0) / 2.0
        x_half   = (xr0 - xl0) / 2.0 * x_factor
        if x_half > 1e-9:
            ax.set_xlim(x_center - x_half, x_center + x_half)

        if ax is self.ax_rf:
            y_factor = 2.0 ** (dy_px / SENS)
            y_center = (yl0 + yu0) / 2.0
            y_half   = (yu0 - yl0) / 2.0 * y_factor
            if y_half > 1e-9:
                ax.set_ylim(y_center - y_half, y_center + y_half)

        self.canvas.draw_idle()
        return

    # ── draw stroke ────────────────────────────────────────────────────────────
    if not self._drawing or event.inaxes is not self.ax_rf:
        return
    ix, fy = self._event_to_ix(event)
    if ix is not None:
        self._apply_stroke(ix, fy, self._last_ix)
        self._last_ix = ix
        self._update_rf_plot()
        self._schedule_sim()


def _on_release(self, event) -> None:
    if event.button == 3:
        self._zoom_active = False
        self._zoom_ax     = None

    if event.button in (1, 2):
        was_resize           = self._resize_left or self._resize_right
        self._slide_active   = False
        self._resize_left    = False
        self._resize_right   = False
        self._slide_amp_snap   = None
        self._slide_phase_snap = None
        if was_resize:
            self._commit_window_resize()
            self._resize_amp_snap   = None
            self._resize_phase_snap = None

    if event.button == 1:
        self._drawing = False
        self._last_ix = -1


def _on_scroll(self, event) -> None:
    """Scroll wheel: zoom x-axis centred on cursor."""
    ax = event.inaxes
    if ax is None or event.xdata is None:
        return
    FACTOR = 1.15
    scale  = 1.0 / FACTOR if event.button == "up" else FACTOR
    cx     = event.xdata
    xl, xr = ax.get_xlim()
    new_xl = cx + (xl - cx) * scale
    new_xr = cx + (xr - cx) * scale
    if new_xr - new_xl > 1e-9:
        ax.set_xlim(new_xl, new_xr)
    self.canvas.draw_idle()
