"""
Physical constants used throughout the simulator.
Keep all magic numbers here — never inline them in physics or UI code.
"""

import numpy as np

# Proton gyromagnetic ratio
GAMMA_HZ_PER_T: float = 42.577e6   # Hz / T
GAMMA_RAD:      float = 2 * np.pi * GAMMA_HZ_PER_T

# Fat-water chemical shift (protons vs lipids)
FAT_WATER_PPM: float = 3.5e-6      # dimensionless (3.5 ppm)
