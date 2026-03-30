import jax
import jax.numpy as jnp


def _float():
    """Return jnp.float64 if x64 is enabled, jnp.float32 otherwise."""
    return jnp.float64 if jax.config.x64_enabled else jnp.float32
