"""
RF Pulse Bloch Equation Simulator
==================================
Hand-draw RF pulse shapes and see real-time slice profiles.

Parameters
----------
  - RF duration (ms)  – typed manually
  - Slice thickness (mm)
  - TBW (time-bandwidth product)
  - B1 max (µT)
  - Gradient max (mT/m)

Requirements:
    pip install numpy scipy matplotlib PyQt5

Run:
    python rf_bloch_simulator.py
"""

import sys
import numpy as np
from scipy.signal import windows
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QGroupBox, QComboBox,
    QRadioButton, QStatusBar, QLineEdit,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor, QDoubleValidator, QCursor


# ── physical constants ─────────────────────────────────────────────────────────
GAMMA_HZ_PER_T = 42.577e6   # Hz/T  (proton gyromagnetic ratio)
GAMMA_RAD      = 2 * np.pi * GAMMA_HZ_PER_T


# ── Bloch solver (vectorised, no relaxation) ───────────────────────────────────

# ── Bloch solver ───────────────────────────────────────────────────────────────
# We use Numba for the Bloch simulation — it is the only real hotspot.
# The time-step loop (N=512) cannot be vectorised because each step depends on
# the previous Mx/My/Mz state.  Numba JIT-compiles it to native code and runs
# the S=300 slice positions in parallel via prange.
#
# First import attempt: if Numba is not installed we fall back to a pure-numpy
# vectorised version that is still reasonably fast.

try:
    import numba
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=True)
    def bloch_simulate(rf_amp_rad, rf_phase_rad, offsets_hz, dt_s):
        """
        Hard-pulse Bloch simulation — Numba parallel JIT version.
        Outer loop (slice positions) runs in parallel; inner loop (RF steps)
        is sequential because each step depends on the previous M state.
        """
        N = rf_amp_rad.shape[0]
        S = offsets_hz.shape[0]

        Mx_out  = np.empty(S)
        My_out  = np.empty(S)
        Mz_out  = np.empty(S)

        for s in prange(S):                      # parallel over slice positions
            dw = 2.0 * np.pi * offsets_hz[s] * dt_s

            mx = 0.0; my = 0.0; mz = 1.0        # initial equilibrium

            for i in range(N):                   # sequential over RF time steps
                a   = rf_amp_rad[i]
                phi = rf_phase_rad[i]

                wx = a * np.cos(phi)
                wy = a * np.sin(phi)
                wz = dw

                w2 = wx*wx + wy*wy + wz*wz
                w  = np.sqrt(w2) + 1e-30
                nx = wx / w;  ny = wy / w;  nz = wz / w

                c  = np.cos(w)
                s_ = np.sin(w)
                oc = 1.0 - c

                nmx = mx*(c + nx*nx*oc) + my*(nx*ny*oc - nz*s_) + mz*(nx*nz*oc + ny*s_)
                nmy = mx*(ny*nx*oc + nz*s_) + my*(c + ny*ny*oc) + mz*(ny*nz*oc - nx*s_)
                nmz = mx*(nz*nx*oc - ny*s_) + my*(nz*ny*oc + nx*s_) + mz*(c + nz*nz*oc)

                mx = nmx;  my = nmy;  mz = nmz

            Mx_out[s] = mx
            My_out[s] = my
            Mz_out[s] = mz

        return Mx_out + 1j * My_out, Mz_out

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False

    def bloch_simulate(rf_amp_rad, rf_phase_rad, offsets_hz, dt_s):
        """
        Hard-pulse Bloch simulation — pure-numpy vectorised fallback.
        All S slice positions are computed simultaneously per RF step.
        Install numba for ~10-20x speedup.
        """
        N = len(rf_amp_rad)
        S = len(offsets_hz)

        dw = 2.0 * np.pi * offsets_hz * dt_s    # (S,) off-res rotation per step

        Mx = np.zeros(S)
        My = np.zeros(S)
        Mz = np.ones(S)

        for i in range(N):
            a   = rf_amp_rad[i]
            phi = rf_phase_rad[i]

            wx = a * np.cos(phi)
            wy = a * np.sin(phi)
            wz = dw

            w  = np.sqrt(wx**2 + wy**2 + wz**2) + 1e-30
            nx = wx / w;  ny = wy / w;  nz = wz / w

            c  = np.cos(w);  s_ = np.sin(w);  oc = 1.0 - c

            nMx = Mx*(c + nx*nx*oc)      + My*(nx*ny*oc - nz*s_) + Mz*(nx*nz*oc + ny*s_)
            nMy = Mx*(ny*nx*oc + nz*s_)  + My*(c + ny*ny*oc)     + Mz*(ny*nz*oc - nx*s_)
            nMz = Mx*(nz*nx*oc - ny*s_)  + My*(nz*ny*oc + nx*s_) + Mz*(c + nz*nz*oc)

            Mx, My, Mz = nMx, nMy, nMz

        return Mx + 1j * My, Mz


# ── preset pulses ──────────────────────────────────────────────────────────────
# All presets receive (n, flip_deg, tbw) and return (amp, phase).
# Amplitude can be negative — negative lobes are real and go below zero.
# _norm scales so ∑amp = flip_deg in rad (correct hard-pulse area).

def _norm(env, flip_deg):
    """Normalise so the signed area equals the requested flip angle in rad."""
    s = env.sum()
    if abs(s) < 1e-12:
        # fall back to peak normalisation if signed sum is near zero
        pk = np.abs(env).max()
        return env / pk * np.deg2rad(flip_deg) if pk > 1e-12 else env
    return env / s * np.deg2rad(flip_deg)

def preset_sinc_hann(n, flip_deg, tbw):
    """Sinc × Hann — standard workhorse, negative sidelobes intact."""
    t   = np.linspace(-tbw / 2, tbw / 2, n)
    env = np.sinc(t) * windows.hann(n)
    return _norm(env, flip_deg), np.zeros(n)

def preset_sinc_hamming(n, flip_deg, tbw):
    """Sinc × Hamming — slightly wider transition band, negative lobes intact."""
    t   = np.linspace(-tbw / 2, tbw / 2, n)
    env = np.sinc(t) * windows.hamming(n)
    return _norm(env, flip_deg), np.zeros(n)

def preset_sinc_infinite(n, flip_deg, tbw):
    """
    Unwindowed sinc — maximum lobes for given TBW, no tapering.
    Negative lobes appear as real negative amplitude values.
    """
    t   = np.linspace(-tbw / 2, tbw / 2, n)
    env = np.sinc(t)
    return _norm(env, flip_deg), np.zeros(n)

def preset_sinc_gauss(n, flip_deg, tbw):
    """
    Sinc × Gaussian — better sidelobe suppression than Hann while
    preserving more lobes.  Negative sidelobes intact.
    Gaussian sigma = tbw/4 so width scales with TBW.
    """
    t     = np.linspace(-tbw / 2, tbw / 2, n)
    sigma = tbw / 4.0
    gauss = np.exp(-t**2 / (2 * sigma**2))
    env   = np.sinc(t) * gauss
    return _norm(env, flip_deg), np.zeros(n)

def preset_sinc_gauss_causal(n, flip_deg, tbw):
    """
    Asymmetric (minimum-phase / causal) sinc-Gauss:
    The main lobe sits at the RIGHT end of the pulse.
    The leading sidelobes (negative) ramp up from the left.
    Achieved by mirroring the time axis of a standard sinc-gauss
    so t=0 is at the far right (index n-1).
    """
    t     = np.linspace(-tbw, 0, n)          # t runs from -tbw → 0 (peak at right)
    sigma = tbw / 4.0
    gauss = np.exp(-t**2 / (2 * sigma**2))
    env   = np.sinc(t) * gauss               # sinc(0)=1 at right end
    return _norm(env, flip_deg), np.zeros(n)

def preset_gauss(n, flip_deg, tbw):
    """Pure Gaussian — no lobes, smooth, poor slice selectivity."""
    t   = np.linspace(-0.5, 0.5, n)
    sig = 0.5 / max(tbw, 1)
    env = np.exp(-t**2 / (2 * sig**2))
    return _norm(env, flip_deg), np.zeros(n)

def preset_rect(n, flip_deg, tbw):
    """Rectangular (hard pulse) — worst slice profile, easiest to implement."""
    env = np.zeros(n)
    env[n//4 : 3*n//4] = 1.0
    return _norm(env, flip_deg), np.zeros(n)

def preset_chirp(n, flip_deg, tbw):
    """Linear-phase chirp (adiabatic-like sweep) — TBW sets sweep bandwidth."""
    t     = np.linspace(0, 1, n)
    phase = np.pi * tbw * (t - 0.5)**2
    env   = windows.hann(n)
    return _norm(env, flip_deg), phase

PRESETS = {
    "Sinc + Hann":           preset_sinc_hann,
    "Sinc + Hamming":        preset_sinc_hamming,
    "Sinc infinite (no win)":preset_sinc_infinite,
    "Sinc × Gauss":          preset_sinc_gauss,
    "Sinc × Gauss (causal)": preset_sinc_gauss_causal,
    "Gaussian":              preset_gauss,
    "Rectangular":           preset_rect,
    "Chirp":                 preset_chirp,
}


# ── Numba warm-up ──────────────────────────────────────────────────────────────
# Trigger JIT compilation before the UI starts so the first interaction is fast.
def _warmup_bloch():
    if not _NUMBA_AVAILABLE:
        return
    _dummy_amp   = np.zeros(8,  dtype=np.float64)
    _dummy_phase = np.zeros(8,  dtype=np.float64)
    _dummy_off   = np.zeros(16, dtype=np.float64)
    bloch_simulate(_dummy_amp, _dummy_phase, _dummy_off, 1e-6)


class RFSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Pulse Bloch Simulator")
        self.setMinimumSize(1200, 760)

        # internal state
        self.N          = 512
        self.n_slices   = 300
        self.CANVAS_MULT    = 4
        self.canvas_dur_ms  = 4.0 * self.CANVAS_MULT   # updated with rf_dur_ms
        # arrays are canvas-sized (CANVAS_MULT × N)
        self.rf_amp   = np.zeros(self.N * self.CANVAS_MULT)
        self.rf_phase = np.zeros(self.N * self.CANVAS_MULT)
        self._drawing   = False
        self._draw_mode = "amp"   # "amp" | "phase" | "slide"
        self._last_ix   = -1

        # window: both edges are draggable
        # win_start_ms = left edge,  win_end_ms = right edge
        self.win_start_ms = 0.0
        self.win_end_ms   = 4.0      # initialised equal to rf_dur_ms

        # waveform-slide state
        self._slide_active     = False
        self._slide_drag_x0    = 0.0
        self._slide_amp_snap   = None
        self._slide_phase_snap = None

        # window-resize drag state — separate flag for left/right edge
        self._resize_left        = False
        self._resize_right       = False
        self._resize_amp_snap    = None   # full canvas snapshot at drag start
        self._resize_phase_snap  = None
        self._RESIZE_TOL_MS      = 0.0

        # zoom state — one entry per axis, stores (xl, xr, yl, yu) at drag start
        # and pixel position at drag start
        self._zoom_active  = False
        self._zoom_ax      = None
        self._zoom_start_px   = (0, 0)   # (x_px, y_px) when right-button pressed
        self._zoom_start_lim  = None     # (xl, xr, yl, yu) at that moment

        # physical params (defaults)
        self.rf_dur_ms  = 4.0
        self.flip_deg   = 90.0
        self.tbw        = 4.0
        self.b0_T       = 3.0
        self.slice_mm   = 5.0
        self.canvas_dur_ms = self.rf_dur_ms * self.CANVAS_MULT
        self.win_start_ms   = 0.0
        self.win_end_ms     = self.rf_dur_ms
        self._RESIZE_TOL_MS = self.canvas_dur_ms * 0.02

        self._build_ui()
        self._connect_signals()
        self._on_b0_changed(1)          # trigger 3T defaults (Larmor title + B1 suggestion)
        self._apply_preset("Sinc + Hann")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 4)

        root.addLayout(self._build_toolbar())

        self.fig = plt.Figure(figsize=(14, 8))
        self._build_axes()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(480)
        # suppress the default right-click context menu so right-drag zoom works cleanly
        self.canvas.setContextMenuPolicy(Qt.PreventContextMenu)
        root.addWidget(self.canvas)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            "Left-drag = draw  |  '↔ Shift pulse' mode: left-drag shifts the whole waveform along time axis  |  "
            "Right-drag = zoom  |  Right dbl-click = reset zoom  |  Scroll = x-zoom"
        )

        self.canvas.mpl_connect("button_press_event",   self._on_press)
        self.canvas.mpl_connect("motion_notify_event",  self._on_move)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event",         self._on_scroll)

        self._sim_timer = QTimer()
        self._sim_timer.setSingleShot(True)
        self._sim_timer.timeout.connect(self._run_sim)

    def _lbl(self, text):
        l = QLabel(text)
        l.setStyleSheet("color:#8b949e; font-size:11px;")
        return l

    def _build_toolbar(self):
        row = QHBoxLayout()
        row.setSpacing(8)

        # preset
        g = QGroupBox("Preset")
        gl = QHBoxLayout(g); gl.setContentsMargins(6, 2, 6, 4)
        self.cb_preset = QComboBox()
        self.cb_preset.addItems(list(PRESETS.keys()))
        self.cb_preset.setMinimumWidth(175)
        gl.addWidget(self.cb_preset)
        row.addWidget(g)

        # draw mode  (Amplitude | Phase | Shift pulse)
        g2 = QGroupBox("Draw mode")
        g2l = QHBoxLayout(g2); g2l.setContentsMargins(6, 2, 6, 4)
        self.rb_amp   = QRadioButton("Amplitude")
        self.rb_phase = QRadioButton("Phase")
        self.rb_slide = QRadioButton("↔ Shift pulse")
        self.rb_slide.setStyleSheet("color:#ffa657;")
        self.rb_amp.setChecked(True)
        g2l.addWidget(self.rb_amp)
        g2l.addWidget(self.rb_phase)
        g2l.addWidget(self.rb_slide)
        row.addWidget(g2)

        # RF duration — typed manually
        g3 = QGroupBox("RF duration")
        g3l = QHBoxLayout(g3); g3l.setContentsMargins(6, 2, 6, 4); g3l.setSpacing(4)
        self.edit_dur = QLineEdit("4.00")
        self.edit_dur.setFixedWidth(56)
        self.edit_dur.setValidator(QDoubleValidator(0.1, 200.0, 2))
        self.edit_dur.setAlignment(Qt.AlignRight)
        self.edit_dur.setStyleSheet(
            "background:#0d1117; color:#c9d1d9; border:1px solid #30363d;"
            "border-radius:4px; padding:1px 4px; font-size:12px;"
        )
        g3l.addWidget(self.edit_dur); g3l.addWidget(self._lbl("ms"))
        row.addWidget(g3)

        # flip angle
        g4 = QGroupBox("Flip angle")
        g4l = QHBoxLayout(g4); g4l.setContentsMargins(6, 2, 6, 4); g4l.setSpacing(4)
        self.sl_flip = QSlider(Qt.Horizontal)
        self.sl_flip.setRange(1, 360); self.sl_flip.setValue(90)
        self.sl_flip.setFixedWidth(90)
        self.lbl_flip = QLabel("90°"); self.lbl_flip.setMinimumWidth(34)
        g4l.addWidget(self.sl_flip); g4l.addWidget(self.lbl_flip)
        row.addWidget(g4)

        # TBW
        g5 = QGroupBox("TBW")
        g5l = QHBoxLayout(g5); g5l.setContentsMargins(6, 2, 6, 4); g5l.setSpacing(4)
        self.sl_tbw = QSlider(Qt.Horizontal)
        self.sl_tbw.setRange(1, 16); self.sl_tbw.setValue(4)
        self.sl_tbw.setFixedWidth(80)
        self.lbl_tbw = QLabel("4"); self.lbl_tbw.setMinimumWidth(18)
        g5l.addWidget(self.sl_tbw); g5l.addWidget(self.lbl_tbw)
        row.addWidget(g5)

        # Slice thickness
        g6 = QGroupBox("Slice thickness")
        g6l = QHBoxLayout(g6); g6l.setContentsMargins(6, 2, 6, 4); g6l.setSpacing(4)
        self.sl_slice = QSlider(Qt.Horizontal)
        self.sl_slice.setRange(1, 50); self.sl_slice.setValue(5)
        self.sl_slice.setFixedWidth(80)
        self.lbl_slice = QLabel("5 mm"); self.lbl_slice.setMinimumWidth(36)
        g6l.addWidget(self.sl_slice); g6l.addWidget(self.lbl_slice)
        row.addWidget(g6)

        # B0 field strength selector
        g_b0 = QGroupBox("B0 field")
        g_b0l = QHBoxLayout(g_b0); g_b0l.setContentsMargins(6, 2, 6, 4); g_b0l.setSpacing(4)
        self.cb_b0 = QComboBox()
        self.cb_b0.addItems(["1.5 T", "3.0 T", "7.0 T"])
        self.cb_b0.setCurrentIndex(1)   # default 3T
        self.cb_b0.setMinimumWidth(68)
        g_b0l.addWidget(self.cb_b0)
        row.addWidget(g_b0)

        # B1 peak — computed from flip angle, pulse shape, and duration (read-only)
        g7 = QGroupBox("B1 peak (computed)")
        g7l = QHBoxLayout(g7); g7l.setContentsMargins(6, 2, 6, 4); g7l.setSpacing(4)
        self.lbl_b1_computed = QLabel("—")
        self.lbl_b1_computed.setStyleSheet(
            "color:#ffa657; font-size:12px; font-weight:500;"
        )
        self.lbl_b1_computed.setMinimumWidth(52)
        g7l.addWidget(self.lbl_b1_computed); g7l.addWidget(self._lbl("µT"))
        row.addWidget(g7)

        # Gradient max — computed from TBW / (γ · T_RF · Δz), read-only display
        g8 = QGroupBox("G_slice (computed)")
        g8l = QHBoxLayout(g8); g8l.setContentsMargins(6, 2, 6, 4); g8l.setSpacing(4)
        self.lbl_grad_computed = QLabel("—")
        self.lbl_grad_computed.setStyleSheet(
            "color:#ffa657; font-size:12px; font-weight:500;"
        )
        self.lbl_grad_computed.setMinimumWidth(70)
        g8l.addWidget(self.lbl_grad_computed); g8l.addWidget(self._lbl("mT/m"))
        row.addWidget(g8)

        # Clear
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedWidth(52)
        row.addWidget(self.btn_clear)

        # Brush smoothness
        g_brush = QGroupBox("Brush smooth")
        g_brushl = QHBoxLayout(g_brush); g_brushl.setContentsMargins(6, 2, 6, 4); g_brushl.setSpacing(4)
        self.sl_brush = QSlider(Qt.Horizontal)
        self.sl_brush.setRange(1, 40); self.sl_brush.setValue(8)
        self.sl_brush.setFixedWidth(70)
        self.lbl_brush = QLabel("8"); self.lbl_brush.setMinimumWidth(16)
        g_brushl.addWidget(self.sl_brush); g_brushl.addWidget(self.lbl_brush)
        row.addWidget(g_brush)

        # Smooth button — post-process the whole canvas
        self.btn_smooth = QPushButton("Smooth")
        self.btn_smooth.setFixedWidth(60)
        self.btn_smooth.setToolTip("Apply Savitzky-Golay smoothing to entire canvas")
        row.addWidget(self.btn_smooth)

        row.addStretch()
        return row

    def _build_axes(self):
        gs = gridspec.GridSpec(
            2, 3, figure=self.fig,
            hspace=0.50, wspace=0.35,
            left=0.06, right=0.98, top=0.95, bottom=0.08
        )
        self.ax_rf  = self.fig.add_subplot(gs[0, :])
        self.ax_mxy = self.fig.add_subplot(gs[1, 0])
        self.ax_mz  = self.fig.add_subplot(gs[1, 1])
        self.ax_ph  = self.fig.add_subplot(gs[1, 2])

        BG  = "#0d1117"
        FIG = "#161b22"
        TC  = "#8b949e"
        SP  = "#30363d"
        self.fig.patch.set_facecolor(FIG)

        for ax in (self.ax_rf, self.ax_mxy, self.ax_mz, self.ax_ph):
            ax.set_facecolor(BG)
            ax.tick_params(labelsize=8, colors=TC)
            for sp in ax.spines.values():
                sp.set_color(SP)

        for ax, title in [
            (self.ax_rf,  "RF pulse  —  left-drag: draw   right-drag: zoom   dbl-right-click: reset zoom"),
            (self.ax_mxy, "|Mxy|  (transverse magnetisation)"),
            (self.ax_mz,  "Mz  (longitudinal magnetisation)"),
            (self.ax_ph,  "Phase(Mxy)  across slice"),
        ]:
            ax.set_title(title, fontsize=9, color="#c9d1d9", pad=5)

        t_ms = np.linspace(0, self.canvas_dur_ms, self.N * self.CANVAS_MULT)
        x_mm = np.linspace(-self.slice_mm * 3, self.slice_mm * 3, self.n_slices)

        # RF — x-axis spans the full canvas (4× rf_dur_ms)
        self.line_amp, = self.ax_rf.plot(t_ms, np.zeros(len(t_ms)), color="#58a6ff", lw=1.8, label="shape (norm.)")
        self.line_phs, = self.ax_rf.plot(t_ms, np.zeros(len(t_ms)), color="#ff7b72", lw=1.2, ls="--", alpha=0.85, label="phase (rad)")
        self.ax_rf.axhline(0, color=SP, lw=0.5)
        self.ax_rf.set_xlim(0, self.canvas_dur_ms)
        self.ax_rf.set_xlabel("time (ms)  [full drawing canvas]", fontsize=8, color=TC)
        self.ax_rf.set_ylabel("normalised amplitude  /  phase (rad)", fontsize=8, color=TC)
        self.ax_rf.legend(loc="upper right", fontsize=8, facecolor=FIG, edgecolor=SP, labelcolor="#c9d1d9")

        # Active window — both edges draggable
        self._win_span = self.ax_rf.axvspan(
            self.win_start_ms, self.win_end_ms, alpha=0.12, color="#ffa657", zorder=0
        )
        self._win_left  = self.ax_rf.axvline(self.win_start_ms, color="#ffa657", lw=2.0, ls="-", zorder=4)
        self._win_right = self.ax_rf.axvline(self.win_end_ms,   color="#ffa657", lw=2.0, ls="-", zorder=4)
        self._win_label = self.ax_rf.text(
            self.win_start_ms + (self.win_end_ms - self.win_start_ms) * 0.02, 0.97,
            f"window: {self.win_start_ms:.2f} – {self.win_end_ms:.2f} ms  |  drag either edge to resize",
            transform=self.ax_rf.get_xaxis_transform(),
            fontsize=7, color="#ffa657", va="top"
        )

        # |Mxy|
        self.line_mxy, = self.ax_mxy.plot(x_mm, np.zeros(self.n_slices), color="#58a6ff", lw=1.8)
        self.ax_mxy.set_xlim(x_mm[0], x_mm[-1])
        self.ax_mxy.set_ylim(-0.02, 1.05)
        self.ax_mxy.set_xlabel("position (mm)", fontsize=8, color=TC)
        self.ax_mxy.set_ylabel("|Mxy|", fontsize=8, color=TC)
        self.ax_mxy.axhline(0, color=SP, lw=0.5)
        self.ax_mxy.axhline(0.5, color="#444c56", lw=0.5, ls=":")

        # Mz
        self.line_mz, = self.ax_mz.plot(x_mm, np.ones(self.n_slices), color="#3fb950", lw=1.8)
        self.ax_mz.set_xlim(x_mm[0], x_mm[-1])
        self.ax_mz.set_ylim(-1.05, 1.05)
        self.ax_mz.set_xlabel("position (mm)", fontsize=8, color=TC)
        self.ax_mz.set_ylabel("Mz", fontsize=8, color=TC)
        self.ax_mz.axhline(0, color=SP, lw=0.5)

        # Phase
        self.line_phase, = self.ax_ph.plot(x_mm, np.zeros(self.n_slices), color="#f78166", lw=1.8)
        self.ax_ph.set_xlim(x_mm[0], x_mm[-1])
        self.ax_ph.set_ylim(-np.pi - 0.2, np.pi + 0.2)
        self.ax_ph.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        self.ax_ph.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"], fontsize=8, color=TC)
        self.ax_ph.set_xlabel("position (mm)", fontsize=8, color=TC)
        self.ax_ph.set_ylabel("phase (rad)", fontsize=8, color=TC)
        self.ax_ph.axhline(0, color=SP, lw=0.5)

        # FWHM markers
        self._fwhm_lines = [
            self.ax_mxy.axvline(0, color="#ffa657", lw=0.8, ls="--", visible=False),
            self.ax_mxy.axvline(0, color="#ffa657", lw=0.8, ls="--", visible=False),
        ]
        # Chemical shift marker on Mxy (shows fat-water shift from B0)
        self._chem_shift_line = self.ax_mxy.axvline(
            0, color="#f78166", lw=1.0, ls=":", visible=False, label="fat shift"
        )
        self._chem_shift_text = self.ax_mxy.text(
            0, 0.5, "", fontsize=7, color="#f78166", va="bottom", ha="left",
            transform=self.ax_mxy.get_xaxis_transform(), visible=False
        )

    # ── signal wiring ──────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.cb_preset.currentTextChanged.connect(
            lambda _: self._apply_preset(self.cb_preset.currentText())
        )
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_smooth.clicked.connect(self._on_smooth)
        self.sl_brush.valueChanged.connect(
            lambda v: self.lbl_brush.setText(str(v))
        )

        from PyQt5.QtWidgets import QButtonGroup
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.rb_amp,   0)
        self._mode_group.addButton(self.rb_phase, 1)
        self._mode_group.addButton(self.rb_slide, 2)
        self._mode_group.idClicked.connect(self._on_mode_changed)
        self.cb_b0.currentIndexChanged.connect(self._on_b0_changed)
        self.edit_dur.editingFinished.connect(self._on_duration_changed)

        def _sl(sl, lbl, suffix, attr):
            def _f(v):
                lbl.setText(f"{v}{suffix}")
                setattr(self, attr, float(v))
                # re-apply preset so shape scales with new TBW/flip
                self._apply_preset(self.cb_preset.currentText())
            sl.valueChanged.connect(_f)

        _sl(self.sl_flip,  self.lbl_flip,  "°",   "flip_deg")
        _sl(self.sl_tbw,   self.lbl_tbw,   "",    "tbw")

        def _sl_sim(sl, lbl, suffix, attr):
            def _f(v):
                lbl.setText(f"{v}{suffix}")
                setattr(self, attr, float(v))
                self._schedule_sim()
            sl.valueChanged.connect(_f)

        _sl_sim(self.sl_slice, self.lbl_slice, " mm", "slice_mm")

    def _on_mode_changed(self, btn_id):
        modes  = {0: 'amp', 1: 'phase', 2: 'slide'}
        self._draw_mode = modes.get(btn_id, 'amp')
        if self._draw_mode == 'slide':
            self.canvas.setCursor(Qt.SizeHorCursor)
        else:
            self.canvas.setCursor(Qt.CrossCursor)

    def _on_duration_changed(self):
        try:
            v = float(self.edit_dur.text().replace(",", "."))
            v = max(0.1, min(200.0, v))
            self.rf_dur_ms = v
            self.edit_dur.setText(f"{v:.2f}")
        except ValueError:
            self.edit_dur.setText(f"{self.rf_dur_ms:.2f}")
        # resize canvas arrays preserving existing content
        new_canvas = self.rf_dur_ms * self.CANVAS_MULT
        old_canvas = self.canvas_dur_ms
        if abs(new_canvas - old_canvas) > 1e-9:
            new_N = self.N * self.CANVAS_MULT
            old_N = len(self.rf_amp)
            new_amp   = np.zeros(new_N)
            new_phase = np.zeros(new_N)
            copy_n = min(old_N, new_N)
            new_amp[:copy_n]   = self.rf_amp[:copy_n]
            new_phase[:copy_n] = self.rf_phase[:copy_n]
            self.rf_amp   = new_amp
            self.rf_phase = new_phase
            self.canvas_dur_ms  = new_canvas
            self.win_start_ms   = 0.0
            self.win_end_ms     = self.rf_dur_ms
            self._RESIZE_TOL_MS = self.canvas_dur_ms * 0.02
        self._update_rf_plot()
        self._schedule_sim()

    # ── preset & clear ─────────────────────────────────────────────────────────

    def _apply_preset(self, name):
        fn = PRESETS[name]
        amp_rad, phase = fn(self.N, self.flip_deg, self.tbw)
        # Store waveform as normalised shape: peak absolute value = 1.0
        # B1 peak is computed from this shape + flip angle + duration in _run_sim
        peak = np.abs(amp_rad).max()
        amp_norm = amp_rad / peak if peak > 1e-12 else np.zeros(self.N)
        idx = self._window_indices()
        self.rf_amp[idx]   = amp_norm
        self.rf_phase[idx] = phase
        self._update_rf_plot()
        self._schedule_sim()

    def _on_b0_changed(self, idx):
        """B0 field selection — updates Larmor frequency and window title."""
        b0_map = {0: 1.5, 1: 3.0, 2: 7.0}
        self.b0_T = b0_map.get(idx, 3.0)
        larmor_mhz = GAMMA_HZ_PER_T * self.b0_T / 1e6
        self.setWindowTitle(
            f"RF Pulse Bloch Simulator  —  B0={self.b0_T:.1f} T  "
            f"(Larmor {larmor_mhz:.1f} MHz)"
        )
        self._schedule_sim()

    def _on_clear(self):
        """Clear entire canvas and reset window to default (rf_dur_ms)."""
        self.rf_amp[:]   = 0.0
        self.rf_phase[:] = 0.0
        # reset window right edge back to nominal rf duration
        self.win_start_ms   = 0.0
        self.win_end_ms     = self.rf_dur_ms
        self._update_window_overlay()
        self._schedule_sim()
        self._update_rf_plot()

    # ── zoom helpers ───────────────────────────────────────────────────────────

    def _all_profile_axes(self):
        return (self.ax_mxy, self.ax_mz, self.ax_ph)

    def _reset_zoom_rf(self):
        """Snap RF x-axis back to full canvas width, y to data extent."""
        self.ax_rf.set_xlim(0, self.canvas_dur_ms)
        amax = max(np.abs(self.rf_amp).max(), 0.5)
        pmax = max(np.abs(self.rf_phase).max(), 0.2)
        top  = max(amax, pmax) * 1.15
        self.ax_rf.set_ylim(-top * 0.6, top)
        self.canvas.draw_idle()

    def _reset_zoom_profiles(self):
        """Snap profile axes back to full data x range."""
        fov_m = self.slice_mm * 1e-3 * 6
        x_mm  = np.array([-fov_m / 2, fov_m / 2]) * 1e3
        for ax in self._all_profile_axes():
            ax.set_xlim(x_mm[0], x_mm[1])
        self.canvas.draw_idle()

    # ── mouse drawing & zoom ───────────────────────────────────────────────────
    #
    # Button 1 (left)  on ax_rf  → draw amplitude or phase
    # Button 3 (right) anywhere  → drag zoom:
    #     RF panel   : horizontal drag → x zoom,  vertical drag → y zoom
    #     Profile axes: horizontal drag → x zoom  (y is fixed / auto)
    # Double-click right          → reset zoom on that panel
    # Scroll wheel                → x-zoom centred on cursor (all axes same rule)

    # ── canvas ↔ sample index helpers ─────────────────────────────────────────

    @property
    def _canvas_N(self):
        return self.N * self.CANVAS_MULT

    def _ms_to_ix(self, t_ms):
        return int(np.clip(t_ms / self.canvas_dur_ms * self._canvas_N, 0, self._canvas_N - 1))

    def _window_indices(self):
        """
        Resample the canvas region [win_start_ms, win_end_ms] into N indices.
        Both edges are user-adjustable. The window width determines effective dt.
        """
        i0  = self._ms_to_ix(self.win_start_ms)
        i1  = max(i0 + 1, self._ms_to_ix(self.win_end_ms))
        idx = np.round(np.linspace(i0, i1 - 1, self.N)).astype(int)
        return np.clip(idx, 0, self._canvas_N - 1)

    def _event_to_ix(self, event):
        if event.inaxes is not self.ax_rf:
            return None, None
        xl, xr = self.ax_rf.get_xlim()
        yl, yu = self.ax_rf.get_ylim()
        # map x from axis data coords → canvas sample index
        frac = (event.xdata - 0.0) / self.canvas_dur_ms   # 0 to 1 across full canvas
        ix   = int(np.clip(frac * self._canvas_N, 0, self._canvas_N - 1))
        fy   = np.clip((event.ydata - yl) / (yu - yl), 0.0, 1.0)
        return ix, fy

    def _apply_stroke(self, ix, fy, last_ix):
        """
        Paint with a Gaussian brush instead of a hard pixel write.
        Brush radius comes from sl_brush (canvas samples).
        Blends the target value into surrounding samples weighted by a
        Gaussian — gives smooth, arc-like curves even with jerky mouse moves.
        """
        yl, yu   = self.ax_rf.get_ylim()
        radius   = max(1, self.sl_brush.value())
        is_amp   = (self._draw_mode == 'amp')
        target   = fy * (yu - yl) + yl if is_amp else (fy - 0.5) * 2 * np.pi

        # fill gaps between last position and current position
        i0     = last_ix if last_ix >= 0 else ix
        i1     = ix
        steps  = max(1, abs(i1 - i0))
        points = np.round(np.linspace(i0, i1, steps + 1)).astype(int)

        # Gaussian kernel: sigma = radius/2, extent = ±2*radius
        r     = radius * 2
        kx    = np.arange(-r, r + 1)
        sigma = max(radius * 0.5, 0.5)
        gauss = np.exp(-0.5 * (kx / sigma) ** 2)
        gauss /= gauss.max()   # peak weight = 1 → centre hits target exactly

        arr = self.rf_amp if is_amp else self.rf_phase
        for cx in points:
            for ki, k in enumerate(kx):
                si = int(cx + k)
                if 0 <= si < self._canvas_N:
                    w       = gauss[ki]
                    arr[si] = arr[si] * (1.0 - w) + target * w
        if is_amp:
            self.rf_amp   = arr
        else:
            self.rf_phase = arr

    def _commit_window_resize(self):
        """
        On window-edge release — rescale the ENTIRE canvas proportionally.

        The window defines which region is "active", but the whole canvas
        is resampled so that:
          - The window region maps to [0 → rf_dur_ms] (first N positions)
          - Everything outside the window is scaled by the same factor
            and placed after rf_dur_ms on the canvas

        This keeps the whole pulse proportionally correct so subsequent
        window adjustments always see a consistent, properly scaled canvas.

        Scale factor = rf_dur_ms / win_dur_ms
          - Narrow window (crop) → factor > 1 → canvas content stretched
          - Wide window (expand) → factor < 1 → canvas content compressed
        """
        snap_amp   = self._resize_amp_snap
        snap_phase = self._resize_phase_snap
        if snap_amp is None:
            self.win_start_ms = 0.0
            self.win_end_ms   = self.rf_dur_ms
            self._update_rf_plot()
            self._schedule_sim()
            return

        win_dur_ms = max(0.001, self.win_end_ms - self.win_start_ms)

        # Scale factor: how much does the canvas content need to stretch/compress
        # so that the selected window fills rf_dur_ms exactly.
        scale = self.rf_dur_ms / win_dur_ms   # >1 = stretch, <1 = compress

        # The new canvas represents: original_canvas_dur * scale in time
        # We resample the entire snapshot into canvas_N points, but shifted
        # so that win_start_ms maps to t=0.

        # Original canvas time axis (ms):  0 → canvas_dur_ms
        # After scaling, win_start → 0, and everything shifts and scales.
        # New time axis: (orig_t - win_start_ms) * scale → fills canvas

        # Resample: for each new canvas position i, find the original position
        #   orig_t = win_start_ms + new_t / scale
        # where new_t goes from 0 → canvas_dur_ms

        new_t_ms  = np.linspace(0, self.canvas_dur_ms, self._canvas_N)
        orig_t_ms = self.win_start_ms + new_t_ms / scale   # map back to snapshot

        # Convert original time to fractional index in snapshot
        orig_frac = orig_t_ms / self.canvas_dur_ms * (self._canvas_N - 1)
        orig_frac = np.clip(orig_frac, 0, self._canvas_N - 1)

        self.rf_amp   = np.interp(orig_frac,
                                  np.arange(self._canvas_N), snap_amp)
        self.rf_phase = np.interp(orig_frac,
                                  np.arange(self._canvas_N), snap_phase)

        # Reset window to 0 → rf_dur_ms
        self.win_start_ms = 0.0
        self.win_end_ms   = self.rf_dur_ms

        # Redraw and recalculate everything
        self._update_rf_plot()
        self._schedule_sim()

    def _on_smooth(self):
        """
        Post-process: Savitzky-Golay smooth over the whole canvas.
        Window size scales with brush radius. Peak amplitude is preserved.
        """
        from scipy.signal import savgol_filter
        radius = max(1, self.sl_brush.value())
        win    = radius * 4 + 1
        if win % 2 == 0:
            win += 1
        win  = max(5, win)
        poly = min(3, win - 1)

        peak_before = np.abs(self.rf_amp).max()
        if peak_before > 1e-12:
            smoothed = savgol_filter(self.rf_amp, win, poly)
            peak_after = np.abs(smoothed).max()
            if peak_after > 1e-12:
                self.rf_amp = smoothed / peak_after * peak_before
            else:
                self.rf_amp = smoothed

        if np.any(self.rf_phase != 0):
            self.rf_phase = savgol_filter(self.rf_phase, win, poly)

        self._update_rf_plot()
        self._schedule_sim()

    def _on_press(self, event):
        if event.inaxes is None:
            return

        # ── right-click: start zoom drag ──────────────────────────────────────
        if event.button == 3:
            # double-click right → reset
            if event.dblclick:
                if event.inaxes is self.ax_rf:
                    self._reset_zoom_rf()
                elif event.inaxes in self._all_profile_axes():
                    self._reset_zoom_profiles()
                return
            self._zoom_active = True
            self._zoom_ax     = event.inaxes
            self._zoom_start_px  = (event.x, event.y)
            ax = event.inaxes
            self._zoom_start_lim = (*ax.get_xlim(), *ax.get_ylim())
            return

        # ── left-click near window edge: resize left or right edge ──────────
        if event.button == 1 and event.inaxes is self.ax_rf and event.xdata is not None:
            tol = self._RESIZE_TOL_MS
            near_right = abs(event.xdata - self.win_end_ms)   <= tol
            near_left  = abs(event.xdata - self.win_start_ms) <= tol
            if near_right and near_left:
                if near_right <= near_left:
                    self._resize_right = True
                else:
                    self._resize_left  = True
            elif near_right:
                self._resize_right = True
            elif near_left:
                self._resize_left  = True
            if self._resize_right or self._resize_left:
                # snapshot full canvas at drag start so commit can restore outside-window content
                self._resize_amp_snap   = self.rf_amp.copy()
                self._resize_phase_snap = self.rf_phase.copy()
                return

        # ── middle-click OR left-click in slide mode: shift the waveform ────────
        if event.inaxes is self.ax_rf and event.xdata is not None:
            if event.button == 2 or (event.button == 1 and self._draw_mode == 'slide'):
                self._slide_active     = True
                self._slide_drag_x0    = event.xdata
                self._slide_amp_snap   = self.rf_amp.copy()
                self._slide_phase_snap = self.rf_phase.copy()
                return

        # ── left-click on RF: draw amplitude or phase ─────────────────────────
        if event.button == 1 and event.inaxes is self.ax_rf and self._draw_mode != 'slide':
            self._drawing = True
            ix, fy = self._event_to_ix(event)
            if ix is not None:
                self._last_ix = ix
                self._apply_stroke(ix, fy, -1)
                self._update_rf_plot()
                self._schedule_sim()

    def _on_move(self, event):
        # ── cursor feedback near window edges ────────────────────────────────
        if (event.inaxes is self.ax_rf and event.xdata is not None
                and not self._slide_active and not self._zoom_active
                and not self._resize_left and not self._resize_right):
            tol = self._RESIZE_TOL_MS
            if (abs(event.xdata - self.win_end_ms)   <= tol or
                    abs(event.xdata - self.win_start_ms) <= tol):
                self.canvas.setCursor(Qt.SplitHCursor)
            elif self._draw_mode == 'slide':
                self.canvas.setCursor(Qt.SizeHorCursor)
            else:
                self.canvas.setCursor(Qt.CrossCursor)

        # ── right edge resize drag ────────────────────────────────────────────
        if self._resize_right and event.xdata is not None:
            min_win = self.canvas_dur_ms * 0.005
            new_end = np.clip(event.xdata,
                              self.win_start_ms + min_win,
                              self.canvas_dur_ms)
            self.win_end_ms = new_end
            self._update_window_overlay()
            self._schedule_sim()
            return

        # ── left edge resize drag ─────────────────────────────────────────────
        if self._resize_left and event.xdata is not None:
            min_win = self.canvas_dur_ms * 0.005
            new_start = np.clip(event.xdata,
                                0.0,
                                self.win_end_ms - min_win)
            self.win_start_ms = new_start
            self._update_window_overlay()
            self._schedule_sim()
            return

        # ── waveform shift drag ────────────────────────────────────────────────
        if self._slide_active and event.xdata is not None:
            delta_ms  = event.xdata - self._slide_drag_x0
            delta_smp = int(round(delta_ms / self.canvas_dur_ms * self._canvas_N))
            # roll the snapshot by delta_smp samples
            self.rf_amp   = np.roll(self._slide_amp_snap,   delta_smp)
            self.rf_phase = np.roll(self._slide_phase_snap, delta_smp)
            # zero-pad the end that wrapped around (no periodic extension)
            if delta_smp > 0:
                self.rf_amp[:delta_smp]   = 0.0
                self.rf_phase[:delta_smp] = 0.0
            elif delta_smp < 0:
                self.rf_amp[delta_smp:]   = 0.0
                self.rf_phase[delta_smp:] = 0.0
            self._update_rf_plot()
            self._schedule_sim()
            return

        # ── zoom drag ─────────────────────────────────────────────────────────
        if self._zoom_active and self._zoom_ax is not None:
            ax   = self._zoom_ax
            xl0, xr0, yl0, yu0 = self._zoom_start_lim
            px0, py0 = self._zoom_start_px
            dx_px = event.x  - px0
            dy_px = event.y  - py0

            SENS = 200.0
            x_factor = 2.0 ** (-dx_px / SENS)
            x_center = (xl0 + xr0) / 2.0
            x_half   = (xr0 - xl0) / 2.0 * x_factor
            new_xl   = x_center - x_half
            new_xr   = x_center + x_half
            if new_xr - new_xl > 1e-9:
                ax.set_xlim(new_xl, new_xr)

            if ax is self.ax_rf:
                y_factor = 2.0 ** (dy_px / SENS)
                y_center = (yl0 + yu0) / 2.0
                y_half   = (yu0 - yl0) / 2.0 * y_factor
                new_yl   = y_center - y_half
                new_yu   = y_center + y_half
                if new_yu - new_yl > 1e-9:
                    ax.set_ylim(new_yl, new_yu)

            self.canvas.draw_idle()
            return

        # ── draw stroke ───────────────────────────────────────────────────────
        if not self._drawing or event.inaxes is not self.ax_rf:
            return
        ix, fy = self._event_to_ix(event)
        if ix is not None:
            self._apply_stroke(ix, fy, self._last_ix)
            self._last_ix = ix
            self._update_rf_plot()
            self._schedule_sim()

    def _on_release(self, event):
        if event.button == 3:
            self._zoom_active = False
            self._zoom_ax     = None
        if event.button in (1, 2):
            was_resize = self._resize_left or self._resize_right
            self._slide_active       = False
            self._resize_left        = False
            self._resize_right       = False
            self._slide_amp_snap     = None
            self._slide_phase_snap   = None
            if was_resize:
                self._commit_window_resize()
                self._resize_amp_snap   = None
                self._resize_phase_snap = None
        if event.button == 1:
            self._drawing = False
            self._last_ix = -1

    def _on_scroll(self, event):
        """Scroll wheel: zoom x-axis centred on cursor position."""
        ax = event.inaxes
        if ax is None:
            return
        FACTOR = 1.15
        scale  = 1.0 / FACTOR if event.button == "up" else FACTOR
        if event.xdata is None:
            return
        cx = event.xdata
        xl, xr = ax.get_xlim()
        new_xl = cx + (xl - cx) * scale
        new_xr = cx + (xr - cx) * scale
        if new_xr - new_xl > 1e-9:
            ax.set_xlim(new_xl, new_xr)
        self.canvas.draw_idle()

    # ── simulation ─────────────────────────────────────────────────────────────

    def _schedule_sim(self):
        self._sim_timer.start(25)

    def _required_gradient(self, shape_w):
        """
        Find the slice-select gradient (T/m) such that the Bloch-simulated
        FWHM of |Mxy| equals slice_mm exactly, for the given shape.

        Uses bisection on a coarse profile (64 slices) for speed.
        This is the only physically correct method — no correction factors,
        works for any pulse shape including truncated, hand-drawn, etc.
        """
        dt_s   = self.rf_dur_ms * 1e-3 / self.N
        dz     = self.slice_mm * 1e-3
        flip_rad = np.deg2rad(self.flip_deg)

        shape_integral = np.abs(shape_w).sum()
        if shape_integral < 1e-12:
            # empty pulse — return nominal gradient
            G_nom = self.tbw / (GAMMA_HZ_PER_T * self.rf_dur_ms * 1e-3 * dz)
            return G_nom, self.tbw

        b1_T   = flip_rad / (GAMMA_RAD * dt_s * shape_integral)
        rf_rad = (shape_w * b1_T * GAMMA_RAD * dt_s).astype(np.float64)
        rf_phs = np.zeros(self.N, dtype=np.float64)

        # Quick sanity check: does this pulse actually produce meaningful Mxy?
        # Test at a very coarse level (on-resonance only) to guard against
        # near-zero pulses returning garbage gradients.
        test_offsets = np.zeros(1, dtype=np.float64)
        Mxy_test, _ = bloch_simulate(rf_rad, rf_phs, test_offsets, dt_s)
        if np.abs(Mxy_test[0]) < 0.01:
            G_nom = self.tbw / (GAMMA_HZ_PER_T * self.rf_dur_ms * 1e-3 * dz)
            return G_nom, self.tbw

        def fwhm_for_G(G_tm):
            """Run a coarse 128-point Bloch sim and return FWHM in mm."""
            S      = 128
            # FOV wide enough to always capture the profile — use 20× slice
            fov_m  = dz * 20
            x_m    = np.linspace(-fov_m / 2, fov_m / 2, S)
            offsets = GAMMA_HZ_PER_T * G_tm * x_m
            Mxy, _ = bloch_simulate(rf_rad, rf_phs, offsets, dt_s)
            mxy    = np.abs(Mxy)
            pk     = mxy.max()
            if pk < 0.01:
                return fov_m * 1e3
            idx = np.where(mxy >= pk * 0.5)[0]
            if len(idx) < 2:
                return fov_m * 1e3
            return (x_m[idx[-1]] - x_m[idx[0]]) * 1e3   # mm

        # Bisection: find G such that fwhm_for_G(G) = slice_mm
        G_lo, G_hi = 1e-5, 1.0   # 0.01 mT/m to 1000 mT/m
        fwhm_lo = fwhm_for_G(G_lo)
        fwhm_hi = fwhm_for_G(G_hi)

        # If bracket doesn't straddle the target, return the closer bound
        # rather than an extreme value
        if fwhm_lo < self.slice_mm:
            return G_lo, G_lo * GAMMA_HZ_PER_T * (self.rf_dur_ms * 1e-3) * dz
        if fwhm_hi > self.slice_mm:
            # Profile still too wide even at maximum G — pulse is very weak/wide
            # Use nominal formula as a safe fallback
            G_nom = self.tbw / (GAMMA_HZ_PER_T * self.rf_dur_ms * 1e-3 * dz)
            return G_nom, self.tbw

        for _ in range(24):
            G_mid    = (G_lo + G_hi) / 2.0
            fwhm_mid = fwhm_for_G(G_mid)
            if fwhm_mid > self.slice_mm:
                G_lo = G_mid
            else:
                G_hi = G_mid
            if abs(G_hi - G_lo) / max(G_hi, 1e-9) < 1e-5:
                break

        G_opt   = (G_lo + G_hi) / 2.0
        tbw_eff = G_opt * GAMMA_HZ_PER_T * (self.rf_dur_ms * 1e-3) * dz
        return G_opt, tbw_eff

    def _draw_fwhm(self, x_mm, mxy):
        peak = mxy.max()
        if peak < 0.01:
            for ln in self._fwhm_lines: ln.set_visible(False)
            return
        idx = np.where(mxy >= peak * 0.5)[0]
        if len(idx) >= 2:
            for ln, xv in zip(self._fwhm_lines, [x_mm[idx[0]], x_mm[idx[-1]]]):
                ln.set_xdata([xv, xv]); ln.set_visible(True)
        else:
            for ln in self._fwhm_lines: ln.set_visible(False)

    def _update_window_overlay(self):
        """Redraw overlay. Both edges draggable."""
        ws, we = self.win_start_ms, self.win_end_ms
        self._win_span.remove()
        self._win_span = self.ax_rf.axvspan(
            ws, we, alpha=0.12, color="#ffa657", zorder=0
        )
        self._win_left.set_xdata([ws, ws])
        self._win_right.set_xdata([we, we])
        win_dur = max(we - ws, 1e-6)
        self._win_label.set_x(ws + win_dur * 0.02)
        self._win_label.set_text(
            f"window: {ws:.2f} – {we:.2f} ms  (selects shape region)"
            f"  |  sim duration = rf_dur = {self.rf_dur_ms:.2f} ms  (fixed)"
        )
        self.canvas.draw_idle()

    def _draw_chem_shift(self, chem_shift_mm):
        """Draw fat-water chemical shift marker on the Mxy plot."""
        if abs(chem_shift_mm) < 0.01:
            self._chem_shift_line.set_visible(False)
            self._chem_shift_text.set_visible(False)
            return
        self._chem_shift_line.set_xdata([chem_shift_mm, chem_shift_mm])
        self._chem_shift_line.set_visible(True)
        self._chem_shift_text.set_x(chem_shift_mm + 0.1)
        self._chem_shift_text.set_text(f"Δfat {chem_shift_mm:.1f}mm")
        self._chem_shift_text.set_visible(True)


        """Redraw overlay. Both edges draggable."""
        ws, we = self.win_start_ms, self.win_end_ms
        self._win_span.remove()
        self._win_span = self.ax_rf.axvspan(
            ws, we, alpha=0.12, color="#ffa657", zorder=0
        )
        self._win_left.set_xdata([ws, ws])
        self._win_right.set_xdata([we, we])
        win_dur = we - ws
        eff_dt_us = win_dur * 1e3 / self.N
        self._win_label.set_x(ws + win_dur * 0.02)
        self._win_label.set_text(
            f"window: {ws:.2f} – {we:.2f} ms  (selects shape region)"
            f"  |  sim duration = rf_dur = {self.rf_dur_ms:.2f} ms  (fixed)"
        )
        self.canvas.draw_idle()

    def _update_rf_plot(self):
        """Redraw the full canvas waveform and the window overlay."""
        t_ms = np.linspace(0, self.canvas_dur_ms, self._canvas_N)
        self.line_amp.set_data(t_ms, self.rf_amp)
        self.line_phs.set_data(t_ms, self.rf_phase)
        self.ax_rf.set_xlim(0, self.canvas_dur_ms)
        amax = max(np.abs(self.rf_amp).max(), 0.5)
        pmax = max(np.abs(self.rf_phase).max(), 0.2)
        top  = max(amax, pmax) * 1.15
        bot  = -top * 0.6
        self.ax_rf.set_ylim(bot, top)
        self._update_window_overlay()

    def _run_sim(self):
        # The window selects WHICH samples to feed into the sim (shape selector).
        # The RF duration typed by the user is ALWAYS the time those N samples
        # play out over — dt is fixed to rf_dur_ms regardless of window width.
        #
        #   dt = rf_dur_ms / N   ← always, invariant
        #
        # Window width only affects which part of the drawn canvas is sampled.
        # Narrow window = zoom into a short region of the drawing and play it
        # over the full rf_dur_ms. Wide window = use more of the drawing.
        # Either way the Bloch sim sees exactly rf_dur_ms worth of pulse.

        dur_s = self.rf_dur_ms * 1e-3     # always the typed RF duration
        dt_s  = dur_s / self.N             # fixed dwell time

        win_dur_ms = max(0.001, self.win_end_ms - self.win_start_ms)

        # Extract N resampled samples from the window region
        idx      = self._window_indices()
        shape_w  = self.rf_amp[idx].astype(np.float64)
        rf_phs_w = self.rf_phase[idx].astype(np.float64)

        # Compute B1 peak from the desired flip angle and pulse shape integral:
        #   flip_rad = γ · B1_peak · ∫|shape(t)| dt
        #            = γ · B1_peak · dt · Σ|shape[i]|
        #   → B1_peak (T) = flip_rad / (γ · dt · Σ|shape[i]|)
        shape_integral = np.abs(shape_w).sum()
        flip_rad = np.deg2rad(self.flip_deg)
        if shape_integral > 1e-12:
            b1_peak_T  = flip_rad / (GAMMA_RAD * dt_s * shape_integral)
        else:
            b1_peak_T  = 0.0
        b1_peak_uT = b1_peak_T * 1e6

        # Scale shape to physical B1 amplitude for the Bloch sim
        rf_rad = shape_w * b1_peak_T * GAMMA_RAD * dt_s   # flip-angle per step (rad)

        g_t_per_m, tbw_eff = self._required_gradient(shape_w)

        fov_m   = self.slice_mm * 1e-3 * 6
        x_m     = np.linspace(-fov_m / 2, fov_m / 2, self.n_slices)
        x_mm    = x_m * 1e3
        offsets = GAMMA_HZ_PER_T * g_t_per_m * x_m

        # Chemical shift offset due to B0: fat-water = 3.5 ppm × γ × B0
        # This shifts the fat slice profile by this many Hz → mm
        chem_shift_hz  = 3.5e-6 * GAMMA_HZ_PER_T * self.b0_T   # Hz
        chem_shift_mm  = chem_shift_hz / (GAMMA_HZ_PER_T * g_t_per_m * 1e-3) if g_t_per_m > 0 else 0.0

        Mxy, Mz = bloch_simulate(rf_rad, rf_phs_w, offsets, dt_s)

        mxy_mag = np.abs(Mxy)
        mxy_phs = np.angle(Mxy)

        noise_floor    = max(mxy_mag.max() * 0.02, 0.01)
        mxy_phs_masked = np.where(mxy_mag >= noise_floor, mxy_phs, np.nan)

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
        self._update_status(x_mm, mxy_mag, Mz, rf_rad, dur_s, g_t_per_m, b1_peak_uT, win_dur_ms, tbw_eff, chem_shift_mm)
        self.canvas.draw_idle()

    def _update_status(self, x_mm, mxy, Mz, rf_rad, dur_s, g_t_per_m, b1_peak_uT, win_dur_ms, tbw_eff, chem_shift_mm):
        peak     = mxy.max()
        flip_eff = np.rad2deg(np.arcsin(np.clip(peak, 0, 1)))
        sum_flip = np.rad2deg(rf_rad.sum())

        idx  = np.where(mxy >= peak * 0.5)[0]
        fwhm = (x_mm[idx[-1]] - x_mm[idx[0]]) if len(idx) >= 2 else 0.0

        g_mt_per_m = g_t_per_m * 1e3
        self.lbl_grad_computed.setText(f"{g_mt_per_m:.2f}")
        self.lbl_b1_computed.setText(f"{b1_peak_uT:.2f}")

        larmor_mhz = GAMMA_HZ_PER_T * self.b0_T / 1e6
        self.status.showMessage(
            f"  B0={self.b0_T:.1f}T  f₀={larmor_mhz:.1f}MHz"
            f"  Δfat={chem_shift_mm:.1f}mm"
            f"   |   B1peak={b1_peak_uT:.2f}µT  G={g_mt_per_m:.2f}mT/m"
            f"   |   TBW_eff={tbw_eff:.2f} (nominal={self.tbw})"
            f"   |   Peak |Mxy|: {peak:.3f}  Eff.flip: {flip_eff:.1f}°"
            f"   |   FWHM: {fwhm:.2f}mm (target: {self.slice_mm:.1f}mm)"
            f"   |   Mz(centre): {Mz[len(Mz)//2]:.3f}"
        )


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor("#161b22"))
    pal.setColor(QPalette.WindowText,      QColor("#c9d1d9"))
    pal.setColor(QPalette.Base,            QColor("#0d1117"))
    pal.setColor(QPalette.AlternateBase,   QColor("#161b22"))
    pal.setColor(QPalette.ToolTipBase,     QColor("#c9d1d9"))
    pal.setColor(QPalette.ToolTipText,     QColor("#c9d1d9"))
    pal.setColor(QPalette.Text,            QColor("#c9d1d9"))
    pal.setColor(QPalette.Button,          QColor("#21262d"))
    pal.setColor(QPalette.ButtonText,      QColor("#c9d1d9"))
    pal.setColor(QPalette.Highlight,       QColor("#1f6feb"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.Mid,             QColor("#30363d"))
    app.setPalette(pal)

    # warm up Numba JIT before window appears (compiles on tiny dummy arrays)
    _warmup_bloch()

    win = RFSimulator()
    numba_str = f"numba {numba.__version__} ✓" if _NUMBA_AVAILABLE else "numba not installed — using numpy fallback"
    win.setWindowTitle(f"RF Pulse Bloch Simulator  [{numba_str}]")
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


# First solve fwhm change from change in rf shape
# then see problem of start window adjustment
# then see for ampplitude, should we increase amplitude