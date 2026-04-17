"""
RF pulse preset shapes.

Each preset function has the signature::

    preset_xyz(n: int, flip_deg: float, tbw: float)
        -> tuple[np.ndarray, np.ndarray]   # (amplitude, phase)

Amplitude values are normalised so their signed sum equals flip_deg in rad
(correct hard-pulse area formula).  Negative values represent negative lobes.

The ``PRESETS`` dict maps display names to functions and is the single source
of truth used by both the UI combo-box and the simulation engine.
"""

import numpy as np
from scipy.signal import windows as sig_windows


# ── normalisation ──────────────────────────────────────────────────────────────

def _norm(envelope: np.ndarray, flip_deg: float) -> np.ndarray:
    """
    Scale envelope so its signed sum equals flip_deg converted to radians.

    Uses signed sum (not absolute sum) because negative lobes subtract from
    the net rotation — using |sum| would over-estimate the flip angle.
    Falls back to peak normalisation when the signed sum is near zero
    (e.g. pure imaginary or antisymmetric pulses).
    """
    signed_sum = envelope.sum()
    if abs(signed_sum) < 1e-12:
        peak = np.abs(envelope).max()
        return envelope / peak * np.deg2rad(flip_deg) if peak > 1e-12 else envelope
    return envelope / signed_sum * np.deg2rad(flip_deg)


# ── preset functions ───────────────────────────────────────────────────────────

def preset_sinc_hann(n: int, flip_deg: float, tbw: float):
    """Sinc × Hann window — standard workhorse, negative sidelobes intact."""
    t = np.linspace(-tbw / 2, tbw / 2, n)
    return _norm(np.sinc(t) * sig_windows.hann(n), flip_deg), np.zeros(n)


def preset_sinc_hamming(n: int, flip_deg: float, tbw: float):
    """Sinc × Hamming — slightly wider transition band, negative lobes intact."""
    t = np.linspace(-tbw / 2, tbw / 2, n)
    return _norm(np.sinc(t) * sig_windows.hamming(n), flip_deg), np.zeros(n)


def preset_sinc_infinite(n: int, flip_deg: float, tbw: float):
    """
    Unwindowed sinc — maximum lobes for given TBW, no tapering.
    Negative lobes appear as real negative amplitude values.
    """
    t = np.linspace(-tbw / 2, tbw / 2, n)
    return _norm(np.sinc(t), flip_deg), np.zeros(n)


def preset_sinc_gauss(n: int, flip_deg: float, tbw: float):
    """
    Sinc × Gaussian — better sidelobe suppression than Hann while
    preserving more lobes.  sigma = tbw/4 so width scales with TBW.
    """
    t     = np.linspace(-tbw / 2, tbw / 2, n)
    sigma = tbw / 4.0
    env   = np.sinc(t) * np.exp(-t**2 / (2 * sigma**2))
    return _norm(env, flip_deg), np.zeros(n)


def preset_sinc_gauss_causal(n: int, flip_deg: float, tbw: float):
    """
    Asymmetric (causal) sinc × Gauss — sidelobes ramp up from the left,
    main lobe fully completed on the right.

    t-axis: -(tbw-1) → +1  (starts and ends at sinc zero-crossings)
    Total span = tbw units, identical density to the symmetric version.
    Gaussian sigma = tbw/4 centred at t=0 (the peak) — same as symmetric.
    """
    t     = np.linspace(-(tbw - 1.0), 1.0, n)
    sigma = tbw / 4.0
    env   = np.sinc(t) * np.exp(-t**2 / (2 * sigma**2))
    return _norm(env, flip_deg), np.zeros(n)


def preset_gauss(n: int, flip_deg: float, tbw: float):
    """Pure Gaussian — no lobes, smooth, poor slice selectivity."""
    t   = np.linspace(-0.5, 0.5, n)
    sig = 0.5 / max(tbw, 1)
    return _norm(np.exp(-t**2 / (2 * sig**2)), flip_deg), np.zeros(n)


def preset_rect(n: int, flip_deg: float, tbw: float):
    """Rectangular (hard pulse) — worst slice profile, easiest to implement."""
    env = np.zeros(n)
    env[n // 4 : 3 * n // 4] = 1.0
    return _norm(env, flip_deg), np.zeros(n)


def preset_chirp(n: int, flip_deg: float, tbw: float):
    """Linear-phase chirp (adiabatic-like sweep) — TBW sets sweep bandwidth."""
    t     = np.linspace(0, 1, n)
    phase = np.pi * tbw * (t - 0.5) ** 2
    return _norm(sig_windows.hann(n), flip_deg), phase


# ── registry ───────────────────────────────────────────────────────────────────

PRESETS: dict = {
    "Sinc + Hann":            preset_sinc_hann,
    "Sinc + Hamming":         preset_sinc_hamming,
    "Sinc infinite (no win)": preset_sinc_infinite,
    "Sinc × Gauss":           preset_sinc_gauss,
    "Sinc × Gauss (causal)":  preset_sinc_gauss_causal,
    "Gaussian":               preset_gauss,
    "Rectangular":            preset_rect,
    "Chirp":                  preset_chirp,
}
