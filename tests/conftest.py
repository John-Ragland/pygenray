"""
Shared test fixtures for pygenray tests.
"""
import matplotlib
matplotlib.use('Agg')


def pytest_addoption(parser):
    """Register --regenerate-physics CLI flag for physics regression tests."""
    parser.addoption(
        '--regenerate-physics', action='store_true', default=False,
        help='Regenerate physics regression fixture and skip comparison.',
    )

import numpy as np
import pytest
import xarray as xr

from pygenray.ray_objects import Ray, RayFan


def _make_ray(launch_angle: float, source_depth: float, n_bottom: int = 0,
              n_surface: int = 0, N: int = 10, R: float = 10000.0) -> Ray:
    """Helper: build a synthetic Ray without running the ODE solver.

    The y array uses the positive-z convention expected by Ray.__init__:
        y[0,:] = travel times
        y[1,:] = depth (positive = deeper)
        y[2,:] = ray parameter sin(θ)/c (positive for downward ray in ODE)
    """
    r = np.linspace(0.0, R, N)
    t = r / 1500.0
    # Depth increases linearly (simulating a shallow downward ray)
    z_ode = np.linspace(source_depth, source_depth + R * 0.01, N)
    p_ode = np.ones(N) * np.sin(np.radians(abs(launch_angle) + 1e-3)) / 1500.0
    y = np.vstack([t, z_ode, p_ode])  # shape (3, N)
    return Ray(r=r, y=y, n_bottom=n_bottom, n_surface=n_surface,
               launch_angle=launch_angle, source_depth=source_depth)


@pytest.fixture
def simple_ray():
    """Single synthetic Ray with 10 range steps."""
    return _make_ray(launch_angle=-10.0, source_depth=100.0)


@pytest.fixture
def simple_rayfan():
    """Small RayFan of 3 synthetic Rays — no ray tracing needed."""
    rays = [
        _make_ray(launch_angle=-5.0,  source_depth=100.0, n_bottom=0),
        _make_ray(launch_angle=5.0,   source_depth=150.0, n_bottom=1),
        _make_ray(launch_angle=-10.0, source_depth=200.0, n_bottom=0),
    ]
    return RayFan(rays)
