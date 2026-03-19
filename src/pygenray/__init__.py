# SPDX-FileCopyrightText: 2025-present John Ragland <john.ragland@whoi.edu>
#
# SPDX-License-Identifier: MIT

import jax
jax.config.update("jax_enable_x64", True)   # default: float64

def set_precision(bits: int = 64) -> None:
    """Set JAX floating-point precision (32 or 64).
    Must be called before any other pygenray or JAX operations.
    """
    if bits not in (32, 64):
        raise ValueError("bits must be 32 or 64")
    jax.config.update("jax_enable_x64", bits == 64)

from .environment import *
from .launch_rays import *
from .integration_processes import *
from .multi_processing import *
from .eigenrays import *
from .ray_objects import *
