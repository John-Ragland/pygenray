"""
Physics correctness tests for pygenray ray tracing.

All tests use flatearth=False to avoid flat-earth correction complicating
the comparison with analytical solutions.
"""
import os
import pathlib

import numpy as np
import pytest
import xarray as xr

from pygenray.environment import OceanEnvironment2D, munk_ssp
from pygenray.launch_rays import shoot_ray, shoot_rays

FIXTURE_DIR = pathlib.Path(__file__).parent / 'fixtures'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _const_c_env(c0=1500.0, z_max=5000.0, r_max=100e3,
                 bathy_depth=4500.0, nz=200, nr=20):
    """Build a range-independent constant sound-speed environment."""
    z = np.linspace(0.0, z_max, nz)
    r = np.linspace(0.0, r_max, nr)
    c_2d = np.full((nr, nz), c0)
    ssp = xr.DataArray(c_2d, dims=['range', 'depth'],
                       coords={'range': r, 'depth': z})
    bathy = xr.DataArray(np.full(nr, bathy_depth), dims=['range'],
                         coords={'range': r})
    return OceanEnvironment2D(sound_speed=ssp, bathymetry=bathy,
                              flat_earth_transform=False)


def _linear_gradient_env(c0=1500.0, g=0.05, z_max=5000.0, r_max=100e3,
                         bathy_depth=4500.0, nz=500, nr=50):
    """Build a range-independent linear-gradient environment: c(z) = c0 + g*z."""
    z = np.linspace(0.0, z_max, nz)
    r = np.linspace(0.0, r_max, nr)
    c_1d = c0 + g * z
    c_2d = np.outer(np.ones(nr), c_1d)
    ssp = xr.DataArray(c_2d, dims=['range', 'depth'],
                       coords={'range': r, 'depth': z})
    bathy = xr.DataArray(np.full(nr, bathy_depth), dims=['range'],
                         coords={'range': r})
    return OceanEnvironment2D(sound_speed=ssp, bathymetry=bathy,
                              flat_earth_transform=False)


def _munk_env(r_max=100e3, nr=50, nz=600, bathy_depth=5000.0):
    """Build a range-independent Munk-profile environment."""
    z = np.linspace(0.0, 6000.0, nz)
    r = np.linspace(0.0, r_max, nr)
    c_1d = munk_ssp(z)
    c_2d = np.outer(np.ones(nr), c_1d)
    ssp = xr.DataArray(c_2d, dims=['range', 'depth'],
                       coords={'range': r, 'depth': z})
    bathy = xr.DataArray(np.full(nr, bathy_depth), dims=['range'],
                         coords={'range': r})
    return OceanEnvironment2D(sound_speed=ssp, bathymetry=bathy,
                              flat_earth_transform=False)


# ---------------------------------------------------------------------------
# A.  Snell's invariant in range-independent constant-c medium
# ---------------------------------------------------------------------------

class TestSnellInvariant:
    """
    In a constant sound-speed medium (dc/dz = 0), the ray-parameter
    p = sin(θ)/c satisfies dp/dx = 0 exactly.  The stored value ray.p = -p_ode
    should therefore be constant to within ODE integration error.
    """

    @pytest.mark.parametrize("user_angle", [-5.0, -10.0, -15.0])
    def test_p_constant_along_ray(self, user_angle):
        c0 = 1500.0
        env = _const_c_env(c0=c0)
        ray = shoot_ray(200.0, 0.0, user_angle, 30e3, 60,
                        env, rtol=1e-9, flatearth=False, debug=False)
        assert ray is not None, "shoot_ray returned None unexpectedly"
        # |p| should be constant; sign may flip at boundary reflections
        abs_p = np.abs(ray.p)
        p_std = np.std(abs_p)
        p_mean = np.mean(abs_p)
        assert p_std / p_mean < 1e-5, (
            f"|p| not constant in constant-c medium: std/mean = {p_std/p_mean:.2e}"
        )


# ---------------------------------------------------------------------------
# B.  Constant sound speed — straight-line rays
# ---------------------------------------------------------------------------

class TestConstantSSPStraightLine:
    """
    In a homogeneous medium the ray paths are straight lines.
    Analytical travel time: t = R / (c · cos θ)
    Analytical final depth:  z = z0 + R · tan θ  (positive downward, ODE convention)
    Stored convention:        ray.z = -(ODE z)
    """

    def test_travel_time_analytical(self):
        c0 = 1500.0
        user_angle = -10.0          # downward
        theta_ode = abs(user_angle)  # internal ODE angle (degrees)
        z0 = 200.0                   # source depth [m]
        R = 20e3                     # receiver range [m]
        env = _const_c_env(c0=c0, r_max=R + 1e3)

        ray = shoot_ray(z0, 0.0, user_angle, R, 50,
                        env, rtol=1e-9, flatearth=False, debug=False)
        assert ray is not None

        t_analytical = R / (c0 * np.cos(np.radians(theta_ode)))
        assert abs(ray.t[-1] - t_analytical) / t_analytical < 1e-3

    def test_final_depth_analytical(self):
        c0 = 1500.0
        user_angle = -10.0
        theta_ode = abs(user_angle)
        z0 = 200.0
        R = 20e3
        env = _const_c_env(c0=c0, r_max=R + 1e3)

        ray = shoot_ray(z0, 0.0, user_angle, R, 50,
                        env, rtol=1e-9, flatearth=False, debug=False)
        assert ray is not None

        z_ode_end = z0 + R * np.tan(np.radians(theta_ode))
        z_expected = -z_ode_end   # stored sign convention
        assert abs(ray.z[-1] - z_expected) / abs(z_expected) < 1e-3

    def test_p_constant_in_const_c(self):
        c0 = 1500.0
        user_angle = -10.0
        theta_ode = abs(user_angle)
        z0 = 200.0
        R = 20e3
        env = _const_c_env(c0=c0, r_max=R + 1e3)

        ray = shoot_ray(z0, 0.0, user_angle, R, 50,
                        env, rtol=1e-9, flatearth=False, debug=False)
        assert ray is not None

        p_expected = -np.sin(np.radians(theta_ode)) / c0  # stored sign
        np.testing.assert_allclose(ray.p, p_expected,
                                   rtol=1e-5, atol=0,
                                   err_msg="p not constant in homogeneous medium")


# ---------------------------------------------------------------------------
# C.  Linear gradient — turning depth
# ---------------------------------------------------------------------------

class TestLinearGradientTurningDepth:
    """
    For c(z) = c0 + g·z the Hamiltonian is conserved:
        H = sqrt(1/c(z)² - p_ode²) = sqrt(1/c_source² - p0²)
    At the turning point (ray horizontal, dz/dx = 0) we have p_ode = 0, so:
        c(z_turn) = 1 / sqrt(1/c_source² - p0²)
    where p0 = sin(θ_ode) / c_source.

    Substituting: c_source = c0 + g·z_source, p0 = sin(θ)/c_source
        c(z_turn) = c_source / cos(θ)
        z_turn    = (c_source/cos(θ) − c0) / g
    """

    C0 = 1500.0
    G = 0.05       # m/s/m
    Z_SRC = 200.0  # source depth [m]
    THETA = 20.0   # degrees downward (user passes -20)

    def _z_turn_analytical(self):
        c_source = self.C0 + self.G * self.Z_SRC
        return (c_source / np.cos(np.radians(self.THETA)) - self.C0) / self.G

    def test_turning_depth_approx(self):
        z_turn = self._z_turn_analytical()
        env = _linear_gradient_env(c0=self.C0, g=self.G)

        ray = shoot_ray(self.Z_SRC, 0.0, -self.THETA, 80e3, 400,
                        env, rtol=1e-9, flatearth=False, debug=False)
        assert ray is not None

        # ray.z uses sign convention: negative = deep
        z_turn_numerical = -np.min(ray.z)   # max depth reached
        assert abs(z_turn_numerical - z_turn) < 50.0, (
            f"Turning depth: expected {z_turn:.1f} m, got {z_turn_numerical:.1f} m"
        )

    def test_hamiltonian_conserved_linear_gradient(self):
        """The Hamiltonian H = sqrt(1/c(z)² − p_ode²) is conserved."""
        env = _linear_gradient_env(c0=self.C0, g=self.G)

        ray = shoot_ray(self.Z_SRC, 0.0, -self.THETA, 80e3, 400,
                        env, rtol=1e-9, flatearth=False, debug=False)
        assert ray is not None

        z_ode = -ray.z                  # positive depth (ODE convention)
        p_ode = -ray.p                  # positive for downward ray
        c_along = self.C0 + self.G * z_ode

        H = np.sqrt(1.0 / c_along**2 - p_ode**2)
        H_std = np.std(H)
        H_mean = np.mean(H)
        assert H_std / H_mean < 1e-4, (
            f"Hamiltonian not conserved: std/mean = {H_std/H_mean:.2e}"
        )


# ---------------------------------------------------------------------------
# D.  Hamiltonian conservation in range-independent Munk profile
# ---------------------------------------------------------------------------

class TestMunkHamiltonianConservation:
    """
    In any range-independent environment the Hamiltonian
        H = sqrt(1/c(z)² − p_ode²)
    is an exact invariant of the ray equations.  Numerical integration
    should preserve it to within ODE tolerance.
    """

    @pytest.mark.parametrize("user_angle", [-5.0, -10.0, -15.0])
    def test_hamiltonian_conserved_munk(self, user_angle):
        env = _munk_env(r_max=100e3)
        z_src = 1000.0   # below SOFAR axis — ray stays in the water column

        ray = shoot_ray(z_src, 0.0, user_angle, 100e3, 200,
                        env, rtol=1e-9, flatearth=False, debug=False)
        assert ray is not None, "shoot_ray returned None; ray may have exited domain"

        z_ode = -ray.z
        p_ode = -ray.p
        c_along = munk_ssp(z_ode)

        arg = 1.0 / c_along**2 - p_ode**2
        # Clamp small negatives from floating-point near turning point
        arg = np.clip(arg, 0.0, None)
        H = np.sqrt(arg)

        # Exclude points at/near the turning point where H → 0
        mask = H > 1e-6 / 1500.0
        if mask.sum() < 5:
            pytest.skip("Too few valid points away from turning point")

        H_valid = H[mask]
        H_std = np.std(H_valid)
        H_mean = np.mean(H_valid)
        assert H_std / H_mean < 1e-3, (
            f"Hamiltonian not conserved in Munk profile: std/mean = {H_std/H_mean:.2e}"
        )


# ---------------------------------------------------------------------------
# E.  Regression / golden-file tests
# ---------------------------------------------------------------------------

def _regenerate_fixture():
    """Run shoot_rays and save regression fixture. Call manually to regenerate."""
    env = _munk_env(r_max=50e3, nr=30, nz=400)
    angles = [-8.0, -4.0, 0.0, 4.0, 8.0]
    rf = shoot_rays(1300.0, 0.0, angles, 50e3, 50, env,
                    n_processes=1, debug=False, flatearth=False)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        FIXTURE_DIR / 'munk_regression.npz',
        ts=rf.ts, zs=rf.zs, ps=rf.ps,
        n_botts=rf.n_botts, n_surfs=rf.n_surfs,
        thetas=rf.thetas,
    )
    return rf


class TestMunkRegression:
    """
    Golden-file regression: compare shoot_rays output against a saved fixture.

    To regenerate the fixture (e.g., after intentional physics changes):
        python -c "from tests.test_physics import _regenerate_fixture; _regenerate_fixture()"
    or run pytest with --regenerate-physics flag.
    """

    FIXTURE = FIXTURE_DIR / 'munk_regression.npz'
    ANGLES = [-8.0, -4.0, 0.0, 4.0, 8.0]

    def _run_rays(self):
        env = _munk_env(r_max=50e3, nr=30, nz=400)
        return shoot_rays(1300.0, 0.0, self.ANGLES, 50e3, 50, env,
                          n_processes=1, debug=False, flatearth=False)

    def test_regression(self, request):
        regenerate = request.config.getoption('--regenerate-physics',
                                              default=False)
        if regenerate or not self.FIXTURE.exists():
            rf = self._run_rays()
            self.FIXTURE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                self.FIXTURE,
                ts=rf.ts, zs=rf.zs, ps=rf.ps,
                n_botts=rf.n_botts, n_surfs=rf.n_surfs,
                thetas=rf.thetas,
            )
            if regenerate:
                pytest.skip("Fixture regenerated; skipping comparison")
            # First-run: fixture just created, nothing to compare
            return

        ref = np.load(self.FIXTURE)
        rf = self._run_rays()

        np.testing.assert_allclose(rf.ts, ref['ts'], atol=1e-6,
                                   err_msg="Travel times changed")
        np.testing.assert_allclose(rf.zs, ref['zs'], atol=0.1,
                                   err_msg="Depths changed")
        np.testing.assert_allclose(rf.ps, ref['ps'], atol=0.1,
                                   err_msg="Ray parameters changed")
        np.testing.assert_array_equal(rf.n_botts, ref['n_botts'])
        np.testing.assert_array_equal(rf.n_surfs, ref['n_surfs'])


