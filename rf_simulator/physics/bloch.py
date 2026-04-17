"""
Bloch equation solver — hard-pulse approximation, no T1/T2 relaxation.

Two implementations are provided:
  1. Numba JIT + parallel (fast, preferred)
  2. Pure NumPy vectorised fallback (always available)

Call ``bloch_simulate`` directly; it resolves to whichever implementation
is available at import time.  Call ``warmup()`` once before the UI starts
to trigger JIT compilation and avoid a slow first interaction.
"""

import numpy as np

# ── Numba implementation ───────────────────────────────────────────────────────

try:
    from numba import njit, prange
    import numba  # noqa: F401 — kept so callers can check _NUMBA_AVAILABLE

    @njit(parallel=True, cache=True, fastmath=True)
    def bloch_simulate(
        rf_amp_rad:   np.ndarray,   # (N,) float64 — flip angle per step [rad]
        rf_phase_rad: np.ndarray,   # (N,) float64 — RF phase per step [rad]
        offsets_hz:   np.ndarray,   # (S,) float64 — off-resonance per slice [Hz]
        dt_s:         float,        # dwell time [s]
    ):
        """
        Hard-pulse Bloch simulation (Numba parallel).

        Outer loop over slice positions runs in parallel.
        Inner loop over RF time steps is sequential (state-dependent).

        Returns
        -------
        Mxy : complex ndarray (S,)
        Mz  : float ndarray   (S,)
        """
        N = rf_amp_rad.shape[0]
        S = offsets_hz.shape[0]
        Mx_out = np.empty(S)
        My_out = np.empty(S)
        Mz_out = np.empty(S)

        for s in prange(S):
            dw = 2.0 * np.pi * offsets_hz[s] * dt_s
            mx = 0.0; my = 0.0; mz = 1.0

            for i in range(N):
                a   = rf_amp_rad[i]
                phi = rf_phase_rad[i]
                wx  = a * np.cos(phi)
                wy  = a * np.sin(phi)
                wz  = dw

                w  = np.sqrt(wx*wx + wy*wy + wz*wz) + 1e-30
                nx = wx / w;  ny = wy / w;  nz = wz / w
                c  = np.cos(w);  s_ = np.sin(w);  oc = 1.0 - c

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

    def bloch_simulate(
        rf_amp_rad:   np.ndarray,
        rf_phase_rad: np.ndarray,
        offsets_hz:   np.ndarray,
        dt_s:         float,
    ):
        """
        Hard-pulse Bloch simulation (pure-NumPy fallback).

        All S slice positions are vectorised per RF step.
        Install numba for ~10–20× speedup.
        """
        N  = len(rf_amp_rad)
        dw = 2.0 * np.pi * offsets_hz * dt_s

        Mx = np.zeros_like(offsets_hz)
        My = np.zeros_like(offsets_hz)
        Mz = np.ones_like(offsets_hz)

        for i in range(N):
            a   = rf_amp_rad[i]
            phi = rf_phase_rad[i]
            wx  = a * np.cos(phi)
            wy  = a * np.sin(phi)
            wz  = dw

            w  = np.sqrt(wx**2 + wy**2 + wz**2) + 1e-30
            nx = wx / w;  ny = wy / w;  nz = wz / w
            c  = np.cos(w);  s_ = np.sin(w);  oc = 1.0 - c

            nMx = Mx*(c + nx*nx*oc)     + My*(nx*ny*oc - nz*s_) + Mz*(nx*nz*oc + ny*s_)
            nMy = Mx*(ny*nx*oc + nz*s_) + My*(c + ny*ny*oc)     + Mz*(ny*nz*oc - nx*s_)
            nMz = Mx*(nz*nx*oc - ny*s_) + My*(nz*ny*oc + nx*s_) + Mz*(c + nz*nz*oc)
            Mx, My, Mz = nMx, nMy, nMz

        return Mx + 1j * My, Mz


# ── warm-up ────────────────────────────────────────────────────────────────────

def warmup() -> None:
    """
    Trigger Numba JIT compilation before the UI opens.
    A no-op when Numba is not available.
    """
    if not _NUMBA_AVAILABLE:
        return
    bloch_simulate(
        np.zeros(8,  dtype=np.float64),
        np.zeros(8,  dtype=np.float64),
        np.zeros(16, dtype=np.float64),
        1e-6,
    )
