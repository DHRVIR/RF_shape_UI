"""
RFSimulator — main application window.

Assembles the UI (toolbar + axes) and wires all event handlers.
Business logic lives in the other ui/ modules; physics in physics/.

Design rule: this file only contains __init__, _build_ui, _build_toolbar,
_build_axes, _connect_signals, and the three simple event responses that
modify top-level state (_on_duration_changed, _on_b0_changed, _on_clear,
_apply_preset, _on_mode_changed).  Everything else is imported as plain
functions and bound to the class below.
"""

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QGroupBox, QComboBox,
    QRadioButton, QStatusBar, QLineEdit, QButtonGroup,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui  import QDoubleValidator

from ..constants        import GAMMA_HZ_PER_T
from ..physics.presets  import PRESETS

# Import free functions from the other ui modules and bind them as methods
from .canvas       import (
    _resize_tol_ms, _ms_to_ix, _display_ms_to_ix,
    _to_display_ms, _to_array_ms,
    _window_indices, _event_to_ix,
    _apply_stroke, _on_smooth,
    _update_window_overlay, _update_rf_plot,
    _all_profile_axes, _reset_zoom_rf, _reset_zoom_profiles,
    _commit_window_resize,
)
from .interactions import _on_press, _on_move, _on_release, _on_scroll
from .simulation   import (
    _schedule_sim, _run_sim,
    _draw_fwhm, _draw_chem_shift, _update_status,
)


class RFSimulator(QMainWindow):
    """Interactive RF pulse designer with real-time Bloch simulation."""

    # ── canvas geometry ────────────────────────────────────────────────────────
    CANVAS_MULT: int = 4   # canvas is this many × rf_dur_ms wide

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RF Pulse Bloch Simulator")
        self.setMinimumSize(1200, 760)

        # ── simulation params ──────────────────────────────────────────────────
        self.N          = 512     # samples in the active window (= RF sim points)
        self.n_slices   = 300     # spatial points in the slice profile

        self.rf_dur_ms  = 4.0
        self.flip_deg   = 90.0
        self.tbw        = 4.0
        self.b0_T       = 3.0
        self.slice_mm   = 5.0

        # ── canvas arrays ──────────────────────────────────────────────────────
        self.canvas_dur_ms = self.rf_dur_ms * self.CANVAS_MULT
        self.rf_amp        = np.zeros(self.N * self.CANVAS_MULT)
        self.rf_phase      = np.zeros(self.N * self.CANVAS_MULT)

        # ── window (orange overlay) ────────────────────────────────────────────
        self.win_start_ms = 0.0
        self.win_end_ms   = self.rf_dur_ms

        # ── interaction state ──────────────────────────────────────────────────
        self._drawing    = False
        self._draw_mode  = "amp"    # "amp" | "phase" | "slide"
        self._last_ix    = -1

        self._slide_active     = False
        self._slide_drag_x0    = 0.0
        self._slide_amp_snap   = None
        self._slide_phase_snap = None

        self._resize_left       = False
        self._resize_right      = False
        self._resize_amp_snap   = None
        self._resize_phase_snap = None
        self._RESIZE_TOL_PX     = 6

        self._zoom_active    = False
        self._zoom_ax        = None
        self._zoom_start_px  = (0, 0)
        self._zoom_start_lim = None

        # ── gradient calculation mode ──────────────────────────────────────────
        # False → analytic TBW formula  (preset / slider driven)
        # True  → bisection from shape  (window crop / drawing / smooth)
        self._grad_from_shape    = False
        self._canvas_x_offset_ms = 0.0

        # ── build and connect ──────────────────────────────────────────────────
        self._build_ui()
        self._connect_signals()
        self._on_b0_changed(1)               # set 3T defaults
        self._apply_preset("Sinc + Hann")

    # ── canvas size property ───────────────────────────────────────────────────

    @property
    def _canvas_N(self) -> int:
        return self.N * self.CANVAS_MULT

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
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
        self.canvas.setContextMenuPolicy(Qt.PreventContextMenu)
        root.addWidget(self.canvas)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            "Left-drag = draw  |  '↔ Shift pulse' mode: left-drag shifts waveform  |  "
            "Right-drag = zoom  |  Right dbl-click = reset zoom  |  Scroll = x-zoom"
        )

        self.canvas.mpl_connect("button_press_event",   self._on_press)
        self.canvas.mpl_connect("motion_notify_event",  self._on_move)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event",         self._on_scroll)

        self._sim_timer = QTimer()
        self._sim_timer.setSingleShot(True)
        self._sim_timer.timeout.connect(self._run_sim)

    def _lbl(self, text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet("color:#8b949e; font-size:11px;")
        return l

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        # Preset
        g = QGroupBox("Preset")
        gl = QHBoxLayout(g); gl.setContentsMargins(6, 2, 6, 4)
        self.cb_preset = QComboBox()
        self.cb_preset.addItems(list(PRESETS.keys()))
        self.cb_preset.setMinimumWidth(175)
        gl.addWidget(self.cb_preset)
        row.addWidget(g)

        # Draw mode
        g2 = QGroupBox("Draw mode")
        g2l = QHBoxLayout(g2); g2l.setContentsMargins(6, 2, 6, 4)
        self.rb_amp   = QRadioButton("Amplitude")
        self.rb_phase = QRadioButton("Phase")
        self.rb_slide = QRadioButton("↔ Shift pulse")
        self.rb_slide.setStyleSheet("color:#ffa657;")
        self.rb_amp.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.rb_amp,   0)
        self._mode_group.addButton(self.rb_phase, 1)
        self._mode_group.addButton(self.rb_slide, 2)
        g2l.addWidget(self.rb_amp)
        g2l.addWidget(self.rb_phase)
        g2l.addWidget(self.rb_slide)
        row.addWidget(g2)

        # RF duration
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

        # Flip angle
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

        # B0 field
        g_b0 = QGroupBox("B0 field")
        g_b0l = QHBoxLayout(g_b0); g_b0l.setContentsMargins(6, 2, 6, 4)
        self.cb_b0 = QComboBox()
        self.cb_b0.addItems(["1.5 T", "3.0 T", "7.0 T"])
        self.cb_b0.setCurrentIndex(1)
        self.cb_b0.setMinimumWidth(68)
        g_b0l.addWidget(self.cb_b0)
        row.addWidget(g_b0)

        # B1 peak (computed, read-only)
        g7 = QGroupBox("B1 peak (computed)")
        g7l = QHBoxLayout(g7); g7l.setContentsMargins(6, 2, 6, 4); g7l.setSpacing(4)
        self.lbl_b1_computed = QLabel("—")
        self.lbl_b1_computed.setStyleSheet("color:#ffa657; font-size:12px; font-weight:500;")
        self.lbl_b1_computed.setMinimumWidth(52)
        g7l.addWidget(self.lbl_b1_computed); g7l.addWidget(self._lbl("µT"))
        row.addWidget(g7)

        # G_slice (computed, read-only)
        g8 = QGroupBox("G_slice (computed)")
        g8l = QHBoxLayout(g8); g8l.setContentsMargins(6, 2, 6, 4); g8l.setSpacing(4)
        self.lbl_grad_computed = QLabel("—")
        self.lbl_grad_computed.setStyleSheet("color:#ffa657; font-size:12px; font-weight:500;")
        self.lbl_grad_computed.setMinimumWidth(70)
        g8l.addWidget(self.lbl_grad_computed); g8l.addWidget(self._lbl("mT/m"))
        row.addWidget(g8)

        # Clear
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedWidth(52)
        row.addWidget(self.btn_clear)

        # Brush smooth
        g_brush = QGroupBox("Brush smooth")
        g_brushl = QHBoxLayout(g_brush); g_brushl.setContentsMargins(6, 2, 6, 4)
        self.sl_brush = QSlider(Qt.Horizontal)
        self.sl_brush.setRange(1, 40); self.sl_brush.setValue(8)
        self.sl_brush.setFixedWidth(70)
        self.lbl_brush = QLabel("8"); self.lbl_brush.setMinimumWidth(16)
        g_brushl.addWidget(self.sl_brush); g_brushl.addWidget(self.lbl_brush)
        row.addWidget(g_brush)

        # Smooth button
        self.btn_smooth = QPushButton("Smooth")
        self.btn_smooth.setFixedWidth(60)
        self.btn_smooth.setToolTip("Apply Savitzky-Golay smoothing to entire canvas")
        row.addWidget(self.btn_smooth)

        row.addStretch()
        return row

    def _build_axes(self) -> None:
        gs = gridspec.GridSpec(
            2, 3, figure=self.fig,
            hspace=0.50, wspace=0.35,
            left=0.06, right=0.98, top=0.95, bottom=0.08,
        )
        self.ax_rf  = self.fig.add_subplot(gs[0, :])
        self.ax_mxy = self.fig.add_subplot(gs[1, 0])
        self.ax_mz  = self.fig.add_subplot(gs[1, 1])
        self.ax_ph  = self.fig.add_subplot(gs[1, 2])

        BG  = "#0d1117"; FIG = "#161b22"; TC = "#8b949e"; SP = "#30363d"
        self.fig.patch.set_facecolor(FIG)

        for ax in (self.ax_rf, self.ax_mxy, self.ax_mz, self.ax_ph):
            ax.set_facecolor(BG)
            ax.tick_params(labelsize=8, colors=TC)
            for sp in ax.spines.values():
                sp.set_color(SP)

        titles = {
            self.ax_rf:  "RF pulse  —  left-drag: draw   right-drag: zoom   dbl-right-click: reset zoom",
            self.ax_mxy: "|Mxy|  (transverse magnetisation)",
            self.ax_mz:  "Mz  (longitudinal magnetisation)",
            self.ax_ph:  "Phase(Mxy)  across slice",
        }
        for ax, title in titles.items():
            ax.set_title(title, fontsize=9, color="#c9d1d9", pad=5)

        t_ms = np.linspace(0, self.canvas_dur_ms, self._canvas_N)
        x_mm = np.linspace(-self.slice_mm * 3, self.slice_mm * 3, self.n_slices)

        # RF panel
        self.line_amp, = self.ax_rf.plot(t_ms, np.zeros_like(t_ms), color="#58a6ff", lw=1.8, label="shape (norm.)")
        self.line_phs, = self.ax_rf.plot(t_ms, np.zeros_like(t_ms), color="#ff7b72", lw=1.2, ls="--", alpha=0.85, label="phase (rad)")
        self.ax_rf.axhline(0, color=SP, lw=0.5)
        self.ax_rf.set_xlim(0, self.canvas_dur_ms)
        self.ax_rf.set_xlabel("time (ms)  [full drawing canvas]", fontsize=8, color=TC)
        self.ax_rf.set_ylabel("normalised amplitude  /  phase (rad)", fontsize=8, color=TC)
        self.ax_rf.legend(loc="upper right", fontsize=8, facecolor=FIG, edgecolor=SP, labelcolor="#c9d1d9")

        # Window overlay
        self._win_span  = self.ax_rf.axvspan(self.win_start_ms, self.win_end_ms, alpha=0.12, color="#ffa657", zorder=0)
        self._win_left  = self.ax_rf.axvline(self.win_start_ms, color="#ffa657", lw=2.0, zorder=4)
        self._win_right = self.ax_rf.axvline(self.win_end_ms,   color="#ffa657", lw=2.0, zorder=4)
        self._win_label = self.ax_rf.text(
            self.win_start_ms + (self.win_end_ms - self.win_start_ms) * 0.02, 0.97,
            f"window: {self.win_start_ms:.2f} – {self.win_end_ms:.2f} ms  |  drag either edge to resize",
            transform=self.ax_rf.get_xaxis_transform(),
            fontsize=7, color="#ffa657", va="top",
        )

        # |Mxy|
        self.line_mxy, = self.ax_mxy.plot(x_mm, np.zeros(self.n_slices), color="#58a6ff", lw=1.8)
        self.ax_mxy.set_xlim(x_mm[0], x_mm[-1]); self.ax_mxy.set_ylim(-0.02, 1.05)
        self.ax_mxy.set_xlabel("position (mm)", fontsize=8, color=TC)
        self.ax_mxy.set_ylabel("|Mxy|", fontsize=8, color=TC)
        self.ax_mxy.axhline(0,   color=SP,        lw=0.5)
        self.ax_mxy.axhline(0.5, color="#444c56", lw=0.5, ls=":")

        # Mz
        self.line_mz, = self.ax_mz.plot(x_mm, np.ones(self.n_slices), color="#3fb950", lw=1.8)
        self.ax_mz.set_xlim(x_mm[0], x_mm[-1]); self.ax_mz.set_ylim(-1.05, 1.05)
        self.ax_mz.set_xlabel("position (mm)", fontsize=8, color=TC)
        self.ax_mz.set_ylabel("Mz", fontsize=8, color=TC)
        self.ax_mz.axhline(0, color=SP, lw=0.5)

        # Phase
        self.line_phase, = self.ax_ph.plot(x_mm, np.zeros(self.n_slices), color="#f78166", lw=1.8)
        self.ax_ph.set_xlim(x_mm[0], x_mm[-1]); self.ax_ph.set_ylim(-np.pi - 0.2, np.pi + 0.2)
        self.ax_ph.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        self.ax_ph.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"], fontsize=8, color=TC)
        self.ax_ph.set_xlabel("position (mm)", fontsize=8, color=TC)
        self.ax_ph.set_ylabel("phase (rad)", fontsize=8, color=TC)
        self.ax_ph.axhline(0, color=SP, lw=0.5)

        # FWHM and chem-shift markers
        self._fwhm_lines = [
            self.ax_mxy.axvline(0, color="#ffa657", lw=0.8, ls="--", visible=False),
            self.ax_mxy.axvline(0, color="#ffa657", lw=0.8, ls="--", visible=False),
        ]
        self._chem_shift_line = self.ax_mxy.axvline(0, color="#f78166", lw=1.0, ls=":", visible=False)
        self._chem_shift_text = self.ax_mxy.text(
            0, 0.5, "", fontsize=7, color="#f78166", va="bottom", ha="left",
            transform=self.ax_mxy.get_xaxis_transform(), visible=False,
        )

    # ── signal wiring ──────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.cb_preset.currentTextChanged.connect(self._apply_preset)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_smooth.clicked.connect(self._on_smooth)
        self._mode_group.idClicked.connect(self._on_mode_changed)
        self.cb_b0.currentIndexChanged.connect(self._on_b0_changed)
        self.edit_dur.editingFinished.connect(self._on_duration_changed)
        self.sl_brush.valueChanged.connect(
            lambda v: self.lbl_brush.setText(str(v))
        )

        def _preset_slider(sl, lbl, suffix, attr):
            """Slider that re-applies the current preset when moved."""
            def _f(v):
                lbl.setText(f"{v}{suffix}")
                setattr(self, attr, float(v))
                self._apply_preset(self.cb_preset.currentText())
            sl.valueChanged.connect(_f)

        def _sim_slider(sl, lbl, suffix, attr):
            """Slider that schedules a re-simulation when moved."""
            def _f(v):
                lbl.setText(f"{v}{suffix}")
                setattr(self, attr, float(v))
                self._schedule_sim()
            sl.valueChanged.connect(_f)

        _preset_slider(self.sl_flip,  self.lbl_flip,  "°",    "flip_deg")
        _preset_slider(self.sl_tbw,   self.lbl_tbw,   "",     "tbw")
        _sim_slider   (self.sl_slice, self.lbl_slice,  " mm", "slice_mm")

    # ── simple top-level event responses ──────────────────────────────────────

    def _on_mode_changed(self, btn_id: int) -> None:
        self._draw_mode = {0: "amp", 1: "phase", 2: "slide"}.get(btn_id, "amp")

    def _on_duration_changed(self) -> None:
        try:
            val = float(self.edit_dur.text())
        except ValueError:
            return
        if val <= 0:
            return
        old_N   = self.N * self.CANVAS_MULT
        new_N   = old_N   # sample count stays fixed; only timing changes
        new_canvas = val * self.CANVAS_MULT
        self.rf_dur_ms     = val
        self.canvas_dur_ms = new_canvas
        self.win_start_ms  = 0.0
        self.win_end_ms    = self.rf_dur_ms
        self._apply_preset(self.cb_preset.currentText())

    def _apply_preset(self, name: str) -> None:
        fn = PRESETS[name]
        amp_rad, phase = fn(self.N, self.flip_deg, self.tbw)
        peak           = np.abs(amp_rad).max()
        amp_norm       = amp_rad / peak if peak > 1e-12 else np.zeros(self.N)
        idx            = self._window_indices()
        self.rf_amp[idx]         = amp_norm
        self.rf_phase[idx]       = phase
        self._grad_from_shape    = False
        self._canvas_x_offset_ms = 0.0
        self._update_rf_plot()
        self._schedule_sim()

    def _on_b0_changed(self, idx: int) -> None:
        self.b0_T = {0: 1.5, 1: 3.0, 2: 7.0}.get(idx, 3.0)
        larmor_mhz = GAMMA_HZ_PER_T * self.b0_T / 1e6
        self.setWindowTitle(
            f"RF Pulse Bloch Simulator  —  B0={self.b0_T:.1f}T  (Larmor {larmor_mhz:.1f} MHz)"
        )
        self._schedule_sim()

    def _on_clear(self) -> None:
        self.rf_amp[:]           = 0.0
        self.rf_phase[:]         = 0.0
        self.win_start_ms        = 0.0
        self.win_end_ms          = self.rf_dur_ms
        self._canvas_x_offset_ms = 0.0
        self._update_window_overlay()
        self._update_rf_plot()
        self._schedule_sim()


# ── bind free functions as methods ────────────────────────────────────────────
# This keeps each module a plain Python file (no class inheritance needed)
# while still letting each function access `self`.

_CANVAS_METHODS = [
    _resize_tol_ms, _ms_to_ix, _display_ms_to_ix,
    _to_display_ms, _to_array_ms,
    _window_indices, _event_to_ix,
    _apply_stroke, _on_smooth,
    _update_window_overlay, _update_rf_plot,
    _all_profile_axes, _reset_zoom_rf, _reset_zoom_profiles,
    _commit_window_resize,
]
_INTERACTION_METHODS = [_on_press, _on_move, _on_release, _on_scroll]
_SIM_METHODS         = [_schedule_sim, _run_sim, _draw_fwhm, _draw_chem_shift, _update_status]

for _fn in _CANVAS_METHODS + _INTERACTION_METHODS + _SIM_METHODS:
    setattr(RFSimulator, _fn.__name__, _fn)
