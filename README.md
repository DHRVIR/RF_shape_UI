# RF Pulse Bloch Simulator

> 🚧 **Under Construction** — actively developed, expect changes and new features.

An interactive desktop application for designing and simulating MRI RF pulse shapes with real-time Bloch equation simulation of slice profiles.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-under%20construction-orange)

---

## Features

- **Hand-draw RF pulses** directly on the canvas with a smooth Gaussian brush
- **Real-time Bloch simulation** (hard-pulse approximation, no T1/T2) powered by Numba JIT + parallel CPU cores
- **Live slice profiles**: |Mxy|, Mz, and Phase(Mxy) update as you draw
- **Physically computed parameters**: B1 peak (µT) and slice-select gradient (mT/m) are derived from the pulse shape — never entered manually
- **Window crop tool**: drag either edge of the active window to truncate the pulse; the canvas rescales automatically and TBW/gradient recompute from the new shape
- **Shift pulse mode**: slide the entire waveform along the time axis without changing its shape
- **B0 field selector** (1.5 T / 3.0 T / 7.0 T) with fat-water chemical shift marker
- **Preset pulses**: Sinc+Hann, Sinc+Hamming, Sinc infinite, Sinc×Gauss, Sinc×Gauss (causal), Gaussian, Rectangular, Chirp
- **Zoom**: right-drag to zoom X/Y on the RF panel and X on profiles; scroll wheel zoom; double-right-click to reset
- **Smooth button**: Savitzky-Golay post-processing for hand-drawn curves

---

## Requirements

- Python 3.9 or higher
- See `requirements.txt` for full dependency list

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. (Recommended) Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python rf_bloch_simulator.py
```

### Controls

| Action | How |
|---|---|
| Draw RF amplitude | Left-drag on RF panel (Amplitude mode) |
| Draw RF phase | Left-drag on RF panel (Phase mode) |
| Shift pulse along time axis | Select **↔ Shift pulse** mode, then left-drag |
| Resize active window (left/right edge) | Hover near orange edge line → cursor changes → left-drag |
| Zoom X-axis | Right-drag left/right on any panel |
| Zoom Y-axis (RF only) | Right-drag up/down on RF panel |
| Scroll zoom | Mouse wheel on any panel |
| Reset zoom | Double right-click on any panel |
| Post-process smooth | Click **Smooth** button |
| Clear everything | Click **Clear** button |

---

## Physics

The simulation uses the **hard-pulse Bloch equation** (rotation matrix formulation):

- Each RF sample is treated as an instantaneous rotation about the effective field axis
- Off-resonance frequencies are determined by `offset_hz = γ · G · x` where G is the slice-select gradient
- **Gradient** is computed by bisection: finds G such that Bloch-simulated FWHM of |Mxy| equals the target slice thickness
- **B1 peak** is computed from: `B1_peak = flip_rad / (γ · dt · Σshape)` using the signed integral of the pulse shape
- **TBW** is back-calculated from the found gradient: `TBW_eff = G · γ · T_RF · Δz`
- **Chemical shift** (fat-water offset) scales with B0: `Δx = 3.5 ppm · B0 / G`

---

## Project Structure

```
rf_bloch_simulator.py    # Main application (single file)
requirements.txt         # Python dependencies
README.md                # This file
.gitignore               # Git ignore rules
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Developed as part of MRI RF pulse design research at FAU (Friedrich-Alexander-Universität Erlangen-Nürnberg).
