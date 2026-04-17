"""
RF Pulse Bloch Simulator  v0.1.0
=================================
Package structure:

  rf_simulator/
  ├── constants.py         physical constants (γ, ppm)
  ├── physics/
  │   ├── bloch.py         Bloch solver (Numba JIT + NumPy fallback)
  │   ├── presets.py       preset pulse shapes + PRESETS registry
  │   └── gradient.py      slice-select gradient computation
  └── ui/
      ├── app.py           RFSimulator(QMainWindow) + mixin binding
      ├── canvas.py        coordinate helpers, drawing, window overlay
      ├── interactions.py  mouse + scroll event handlers
      └── simulation.py    _run_sim, profile plots, status bar

Import physics without Qt::

    from rf_simulator.physics.presets import PRESETS

Import full app (requires PyQt5 + matplotlib)::

    from rf_simulator.ui.app import RFSimulator
"""

__version__ = "0.1.0"

