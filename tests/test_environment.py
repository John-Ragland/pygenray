"""
Tests for pygenray.environment: munk_ssp, OceanEnvironment2D, eflat, eflatinv.
"""
import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from pygenray.environment import (
    OceanEnvironment2D,
    eflat,
    eflatinv,
    munk_ssp,
)


# ---------------------------------------------------------------------------
# munk_ssp
# ---------------------------------------------------------------------------

class TestMunkSSP:
    def test_output_shape_matches_input(self):
        z = np.arange(0, 5000, 10)
        c = munk_ssp(z)
        assert c.shape == z.shape

    def test_minimum_at_sofar_depth(self):
        sofar = 1300.0
        z = np.arange(0, 6000, 1)
        c = munk_ssp(z, sofar_depth=sofar)
        # Minimum should be at the SOFAR depth
        assert z[np.argmin(c)] == pytest.approx(sofar, abs=2.0)

    def test_default_params_near_1500_at_sofar(self):
        sofar = 1300.0
        c_sofar = munk_ssp(np.array([sofar]))
        assert c_sofar[0] == pytest.approx(1500.0, abs=5.0)

    def test_scalar_input(self):
        c = munk_ssp(np.array([0.0]))
        assert c.shape == (1,)


# ---------------------------------------------------------------------------
# OceanEnvironment2D construction
# ---------------------------------------------------------------------------

class TestOceanEnvironment2DConstruction:
    def test_default_init_attributes_exist(self):
        env = OceanEnvironment2D()
        for attr in ('sound_speed', 'bathymetry', 'dcdz', 'bottom_angle',
                     'bottom_angle_interp'):
            assert hasattr(env, attr), f"Missing attribute: {attr}"

    def test_default_sound_speed_is_2d(self):
        env = OceanEnvironment2D()
        assert env.sound_speed.ndim == 2
        assert set(env.sound_speed.dims) == {'range', 'depth'}

    def test_default_flat_earth_attributes_exist(self):
        env = OceanEnvironment2D(flat_earth_transform=True)
        assert hasattr(env, 'sound_speed_fe')
        assert hasattr(env, 'bathymetry_fe')

    def test_flat_earth_false_no_fe_attributes(self):
        env = OceanEnvironment2D(flat_earth_transform=False)
        assert not hasattr(env, 'sound_speed_fe')
        assert not hasattr(env, 'bathymetry_fe')

    def test_custom_1d_sound_speed(self):
        z = np.arange(0.0, 3000.0, 10.0)
        c_vals = munk_ssp(z)
        ssp = xr.DataArray(c_vals, dims=['depth'], coords={'depth': z})
        bathy = xr.DataArray(
            np.ones(20) * 4000.0, dims=['range'],
            coords={'range': np.linspace(0, 50e3, 20)}
        )
        env = OceanEnvironment2D(sound_speed=ssp, bathymetry=bathy,
                                 flat_earth_transform=False)
        assert env.sound_speed.ndim == 1
        assert 'depth' in env.sound_speed.dims

    def test_custom_2d_sound_speed(self):
        z = np.arange(0.0, 3000.0, 50.0)
        r = np.linspace(0.0, 50e3, 20)
        c_2d = np.outer(np.ones(len(r)), munk_ssp(z))
        ssp = xr.DataArray(c_2d, dims=['range', 'depth'],
                           coords={'range': r, 'depth': z})
        env = OceanEnvironment2D(sound_speed=ssp, flat_earth_transform=False)
        assert env.sound_speed.ndim == 2

    def test_custom_bathymetry_stored(self):
        bathy_vals = np.ones(20) * 3500.0
        r = np.linspace(0.0, 50e3, 20)
        bathy = xr.DataArray(bathy_vals, dims=['range'], coords={'range': r})
        env = OceanEnvironment2D(bathymetry=bathy, flat_earth_transform=False)
        np.testing.assert_array_equal(env.bathymetry.values, bathy_vals)

    # --- invalid inputs ---

    def test_sound_speed_not_dataarray_raises_type_error(self):
        with pytest.raises(TypeError):
            OceanEnvironment2D(sound_speed=np.ones(100))

    def test_sound_speed_3d_raises_value_error(self):
        da = xr.DataArray(
            np.ones((5, 10, 20)),
            dims=['range', 'depth', 'extra'],
            coords={'range': np.arange(5), 'depth': np.arange(10),
                    'extra': np.arange(20)}
        )
        with pytest.raises(ValueError):
            OceanEnvironment2D(sound_speed=da)

    def test_sound_speed_missing_depth_dim_raises_value_error(self):
        da = xr.DataArray(np.ones(50), dims=['range'],
                          coords={'range': np.arange(50)})
        with pytest.raises(ValueError):
            OceanEnvironment2D(sound_speed=da)

    def test_2d_sound_speed_missing_range_dim_raises_value_error(self):
        da = xr.DataArray(
            np.ones((10, 20)),
            dims=['depth', 'extra'],
            coords={'depth': np.arange(10), 'extra': np.arange(20)}
        )
        with pytest.raises(ValueError):
            OceanEnvironment2D(sound_speed=da)

    def test_bathymetry_not_dataarray_raises_type_error(self):
        with pytest.raises(TypeError):
            OceanEnvironment2D(bathymetry=np.ones(50))

    def test_bathymetry_missing_range_dim_raises_value_error(self):
        da = xr.DataArray(np.ones(50), dims=['depth'],
                          coords={'depth': np.arange(50)})
        with pytest.raises(ValueError):
            OceanEnvironment2D(bathymetry=da)


# ---------------------------------------------------------------------------
# eflat / eflatinv round-trip
# ---------------------------------------------------------------------------

class TestEflat:
    LAT = 35.0

    def test_depth_roundtrip(self):
        dep = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0])
        depf, _ = eflat(dep, self.LAT)
        dep_rec, _ = eflatinv(depf, np.array([self.LAT]))
        np.testing.assert_allclose(dep_rec, dep, atol=1.0,
                                   err_msg="Depth round-trip outside 1 m tolerance")

    def test_sound_speed_roundtrip(self):
        dep = np.array([100.0, 500.0, 1000.0, 2000.0])
        cs = np.array([1500.0, 1490.0, 1480.0, 1510.0])
        depf, csf = eflat(dep, self.LAT, cs)
        _, cs_rec = eflatinv(depf, np.array([self.LAT]), csf)
        np.testing.assert_allclose(cs_rec, cs, rtol=1e-4,
                                   err_msg="Sound speed round-trip outside 0.01% tolerance")

    def test_eflat_increases_depth(self):
        """Flat-earth transformation should increase effective depths."""
        dep = np.array([100.0, 1000.0, 3000.0])
        depf, _ = eflat(dep, self.LAT)
        assert np.all(depf > dep)


# ---------------------------------------------------------------------------
# OceanEnvironment2D.plot smoke test
# ---------------------------------------------------------------------------

class TestOceanEnvironment2DPlot:
    def test_plot_runs_without_error(self):
        env = OceanEnvironment2D()
        fig, ax = plt.subplots()
        plt.sca(ax)
        env.plot()
        plt.close('all')
