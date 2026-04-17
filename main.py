"""
Entry point — run with:   python main.py
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor

from rf_simulator.physics.bloch import warmup
from rf_simulator.ui.app import RFSimulator


def _dark_palette() -> QPalette:
    """GitHub-dark colour palette applied to the whole Qt application."""
    p = QPalette()
    roles = {
        QPalette.Window:          "#161b22",
        QPalette.WindowText:      "#c9d1d9",
        QPalette.Base:            "#0d1117",
        QPalette.AlternateBase:   "#161b22",
        QPalette.ToolTipBase:     "#161b22",
        QPalette.ToolTipText:     "#c9d1d9",
        QPalette.Text:            "#c9d1d9",
        QPalette.Button:          "#21262d",
        QPalette.ButtonText:      "#c9d1d9",
        QPalette.BrightText:      "#ffffff",
        QPalette.Highlight:       "#1f6feb",
        QPalette.HighlightedText: "#ffffff",
    }
    for role, hex_colour in roles.items():
        p.setColor(role, QColor(hex_colour))
    return p


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())

    # Trigger Numba JIT compilation before the window opens
    warmup()

    win = RFSimulator()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
