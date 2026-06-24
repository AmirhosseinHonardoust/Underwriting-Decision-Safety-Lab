"""Shared numpy typing aliases used across the package.

These give array parameters and returns a dtype-level intent (integer label
arrays vs. float probability arrays) instead of a bare ``np.ndarray``. The
families ``np.integer`` / ``np.floating`` are used so any int/float width is
accepted. Shape typing is intentionally not attempted: stable numpy typing
does not yet support array shapes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# Integer label / prediction arrays (0/1 classes).
IntArray = npt.NDArray[np.integer[Any]]

# Float probability / score / threshold arrays.
FloatArray = npt.NDArray[np.floating[Any]]
