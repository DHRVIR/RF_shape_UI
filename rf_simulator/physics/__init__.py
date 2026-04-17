"""Physics layer — pure NumPy/Numba, zero Qt dependency."""

from .bloch    import bloch_simulate, warmup, _NUMBA_AVAILABLE
from .presets  import PRESETS
from .gradient import from_tbw, from_shape