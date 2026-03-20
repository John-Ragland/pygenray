"""
Tests verifying that vmap parallel ray tracing matches serial results.
"""
import numpy as np
import pytest
import xarray as xr

import pygenray as pr
from pygenray.environment import OceanEnvironment2D, munk_ssp
from pygenray.launch_rays import shoot_rays, shoot_ray
from pygenray.eigenrays import find_eigenrays


# ---------------------------------------------------------------------------
# Environment helpers (same as test_physics.py)
# ---------------------------------------------------------------------------

def _const_c_env(c0=1500.0, z_max=5000.0, r_max=100e3,
                 bathy_depth=4500.0, nz=200, nr=20):
    z = np.linspace(0.0, z_max, nz)
    r = np.linspace(0.0, r_max, nr)
    c_2d = np.full((nr, nz), c0)
    ssp = xr.DataArray(c_2d, dims=['range', 'depth'], coords={'range': r, 'depth': z})
    bathy = xr.DataArray(np.full(nr, bathy_depth), dims=['range'], coords={'range': r})
    return OceanEnvironment2D(sound_speed=ssp, bathymetry=bathy, flat_earth_transform=False)


def _munk_env(r_max=100e3, nr=50, nz=600, bathy_depth=5000.0):
    z = np.linspace(0.0, 6000.0, nz)
    r = np.linspace(0.0, r_max, nr)
    c_1d = munk_ssp(z)
    c_2d = np.outer(np.ones(nr), c_1d)
    ssp = xr.DataArray(c_2d, dims=['range', 'depth'], coords={'range': r, 'depth': z})
    bathy = xr.DataArray(np.full(nr, bathy_depth), dims=['range'], coords={'range': r})
    return OceanEnvironment2D(sound_speed=ssp, bathymetry=bathy, flat_earth_transform=False)


# ---------------------------------------------------------------------------
# A.  vmap matches serial for shoot_rays
# ---------------------------------------------------------------------------

class TestVmapMatchesSerial:
    """vmap and serial shoot_rays produce numerically identical ray paths."""

    def _compare_fans(self, fan_serial, fan_vmap, atol_t=1e-6, atol_z=0.1):
        """Helper: assert serial and vmap fans agree in travel time and depth."""
        assert len(fan_serial) == len(fan_vmap), (
            f"Ray count differs: serial {len(fan_serial)}, vmap {len(fan_vmap)}"
        )
        for i in range(len(fan_serial)):
            np.testing.assert_allclose(
                fan_serial.ts[i], fan_vmap.ts[i], atol=atol_t,
                err_msg=f"Travel time mismatch for ray {i}"
            )
            np.testing.assert_allclose(
                fan_serial.zs[i], fan_vmap.zs[i], atol=atol_z,
                err_msg=f"Depth mismatch for ray {i}"
            )

    @pytest.mark.parametrize("angles", [
        [-5.0, 0.0, 5.0],
        [-15.0, -10.0, -5.0, 5.0, 10.0, 15.0],
    ])
    def test_const_c(self, angles):
        env = _const_c_env()
        kwargs = dict(
            source_depth=100.0, source_range=0.0,
            receiver_range=50e3, num_range_save=20,
            environment=env, flatearth=False, debug=False,
        )
        fan_serial = shoot_rays(launch_angles=np.array(angles), parallel=False, **kwargs)
        fan_vmap   = shoot_rays(launch_angles=np.array(angles), parallel=True,  **kwargs)
        self._compare_fans(fan_serial, fan_vmap)

    @pytest.mark.parametrize("angles", [
        [-5.0, 0.0, 5.0],
        [-10.0, -5.0, 5.0, 10.0],
    ])
    def test_munk(self, angles):
        env = _munk_env()
        kwargs = dict(
            source_depth=800.0, source_range=0.0,
            receiver_range=50e3, num_range_save=20,
            environment=env, flatearth=False, debug=False,
        )
        fan_serial = shoot_rays(launch_angles=np.array(angles), parallel=False, **kwargs)
        fan_vmap   = shoot_rays(launch_angles=np.array(angles), parallel=True,  **kwargs)
        self._compare_fans(fan_serial, fan_vmap)

    def test_launch_angles_stored_correctly(self):
        """Stored launch angles must match the user-supplied angles."""
        env = _const_c_env()
        angles = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        fan = shoot_rays(
            launch_angles=angles, source_depth=100.0, source_range=0.0,
            receiver_range=50e3, num_range_save=10, environment=env,
            flatearth=False, debug=False, parallel=True,
        )
        np.testing.assert_array_equal(fan.thetas, angles)


# ---------------------------------------------------------------------------
# B.  Failed (near-vertical) rays handled the same way
# ---------------------------------------------------------------------------

class TestVmapFailedRays:
    """Truly vertical rays should be dropped the same way in both paths."""

    def test_vertical_rays_dropped(self):
        """Rays that go vertical (|theta| >= 89.999°) must be dropped by both paths."""
        env = _const_c_env()
        # 89.9999° triggers the vertical event (threshold: 90 - 1e-3 = 89.999°)
        angles = np.array([-10.0, 0.0, 10.0, 89.9999])
        kwargs = dict(
            source_depth=100.0, source_range=0.0,
            receiver_range=50e3, num_range_save=10,
            environment=env, flatearth=False, debug=False,
        )
        fan_serial = shoot_rays(launch_angles=angles, parallel=False, **kwargs)
        fan_vmap   = shoot_rays(launch_angles=angles, parallel=True,  **kwargs)
        assert len(fan_serial) == len(fan_vmap), (
            f"Valid ray count differs: serial {len(fan_serial)}, vmap {len(fan_vmap)}"
        )


# ---------------------------------------------------------------------------
# C.  Eigenray vmap matches serial
# ---------------------------------------------------------------------------

class TestEigenrayVmapMatchesSerial:
    """vmap eigenray finder converges to same depths as serial."""

    def test_const_c_eigenrays(self):
        env = _const_c_env()
        source_depth = 100.0
        receiver_range = 50e3
        num_range_save = 20

        angles = np.linspace(-20.0, 20.0, 41)
        fan = shoot_rays(
            launch_angles=angles, source_depth=source_depth, source_range=0.0,
            receiver_range=receiver_range, num_range_save=num_range_save,
            environment=env, flatearth=False, debug=False, parallel=True,
        )

        receiver_depths = [200.0, 500.0]
        ztol = 1.0

        erays_serial = find_eigenrays(
            fan, receiver_depths, source_depth=source_depth,
            source_range=0.0, receiver_range=receiver_range,
            num_range_save=num_range_save, environment=env,
            ztol=ztol, max_iter=20, parallel=False,
            flatearth=False, debug=False,
        )
        erays_vmap = find_eigenrays(
            fan, receiver_depths, source_depth=source_depth,
            source_range=0.0, receiver_range=receiver_range,
            num_range_save=num_range_save, environment=env,
            ztol=ztol, max_iter=20, parallel=True,
            flatearth=False, debug=False,
        )

        for rd_idx, rd in enumerate(receiver_depths):
            assert erays_serial.num_eigenrays_found[rd_idx] == erays_vmap.num_eigenrays_found[rd_idx], (
                f"Eigenray count mismatch for receiver depth {rd}: "
                f"serial {erays_serial.num_eigenrays_found[rd_idx]}, "
                f"vmap {erays_vmap.num_eigenrays_found[rd_idx]}"
            )
            # Final depths should be within ztol of each other
            if erays_serial.num_eigenrays_found[rd_idx] > 0:
                z_serial = erays_serial.zs[rd_idx][:, -1]
                z_vmap   = erays_vmap.zs[rd_idx][:, -1]
                # Both should be within ztol of receiver depth
                np.testing.assert_allclose(
                    z_serial, -rd * np.ones_like(z_serial), atol=ztol * 2,
                    err_msg=f"Serial eigenray depth not near receiver depth {rd}"
                )
                np.testing.assert_allclose(
                    z_vmap, -rd * np.ones_like(z_vmap), atol=ztol * 2,
                    err_msg=f"vmap eigenray depth not near receiver depth {rd}"
                )
