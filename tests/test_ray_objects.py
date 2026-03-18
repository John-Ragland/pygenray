"""
Tests for pygenray.ray_objects: Ray, RayFan.
"""
import numpy as np
import pytest
import scipy.io
from matplotlib import pyplot as plt

from pygenray.ray_objects import Ray, RayFan


# ---------------------------------------------------------------------------
# Ray
# ---------------------------------------------------------------------------

class TestRay:
    N = 10
    R = 10000.0

    def _make_ray(self, launch_angle=-10.0, source_depth=100.0,
                  n_bottom=0, n_surface=0):
        r = np.linspace(0.0, self.R, self.N)
        t = r / 1500.0
        z_ode = np.linspace(source_depth, source_depth + self.R * 0.01, self.N)
        p_ode = np.ones(self.N) * np.sin(np.radians(abs(launch_angle) + 1e-3)) / 1500.0
        y = np.vstack([t, z_ode, p_ode])
        return Ray(r=r, y=y, n_bottom=n_bottom, n_surface=n_surface,
                   launch_angle=launch_angle, source_depth=source_depth), y

    def test_attribute_shapes(self):
        ray, _ = self._make_ray()
        assert ray.r.shape == (self.N,)
        assert ray.t.shape == (self.N,)
        assert ray.z.shape == (self.N,)
        assert ray.p.shape == (self.N,)

    def test_z_sign_convention(self):
        """ray.z should equal -y[1,:]."""
        ray, y = self._make_ray()
        np.testing.assert_array_equal(ray.z, -y[1, :])

    def test_p_sign_convention(self):
        """ray.p should equal -y[2,:]."""
        ray, y = self._make_ray()
        np.testing.assert_array_equal(ray.p, -y[2, :])

    def test_launch_angle_stored(self):
        ray, _ = self._make_ray(launch_angle=-15.0)
        assert ray.launch_angle == pytest.approx(-15.0)

    def test_source_depth_stored(self):
        ray, _ = self._make_ray(source_depth=250.0)
        assert ray.source_depth == pytest.approx(250.0)

    def test_optional_launch_angle_not_set(self):
        r = np.linspace(0.0, self.R, self.N)
        t = r / 1500.0
        y = np.vstack([t, np.ones(self.N) * 100.0, np.ones(self.N) * 0.1])
        ray = Ray(r=r, y=y, n_bottom=0, n_surface=0)
        assert not hasattr(ray, 'launch_angle')

    def test_optional_source_depth_not_set(self):
        r = np.linspace(0.0, self.R, self.N)
        t = r / 1500.0
        y = np.vstack([t, np.ones(self.N) * 100.0, np.ones(self.N) * 0.1])
        ray = Ray(r=r, y=y, n_bottom=0, n_surface=0)
        assert not hasattr(ray, 'source_depth')

    def test_n_bottom_n_surface_stored(self):
        ray, _ = self._make_ray(n_bottom=3, n_surface=1)
        assert ray.n_bottom == 3
        assert ray.n_surface == 1

    def test_plot_smoke(self):
        ray, _ = self._make_ray()
        plt.figure()
        ray.plot()
        plt.close('all')


# ---------------------------------------------------------------------------
# RayFan
# ---------------------------------------------------------------------------

class TestRayFan:
    M = 3
    N = 10
    R = 10000.0

    def _make_rays(self, M=None, N=None, R=None):
        M = M or self.M
        N = N or self.N
        R = R or self.R
        rays = []
        for i in range(M):
            r = np.linspace(0.0, R, N)
            theta = float(-5 + i * 5)
            t = r / 1500.0
            z_ode = np.linspace(100.0 + i * 50, 200.0 + i * 50, N)
            p_ode = np.ones(N) * np.sin(np.radians(abs(theta) + 1e-3)) / 1500.0
            y = np.vstack([t, z_ode, p_ode])
            rays.append(
                Ray(r=r, y=y, n_bottom=i % 2, n_surface=0,
                    launch_angle=theta, source_depth=100.0 + i * 50)
            )
        return rays

    # --- Construction ---

    def test_shapes(self, simple_rayfan):
        rf = simple_rayfan
        assert rf.thetas.shape == (self.M,)
        assert rf.rs.shape == (self.M, self.N)
        assert rf.ts.shape == (self.M, self.N)
        assert rf.zs.shape == (self.M, self.N)
        assert rf.ps.shape == (self.M, self.N)
        assert rf.n_botts.shape == (self.M,)
        assert rf.n_surfs.shape == (self.M,)
        assert rf.source_depths.shape == (self.M,)

    def test_ray_ids_set_on_construction(self, simple_rayfan):
        assert hasattr(simple_rayfan, 'ray_ids')
        assert len(simple_rayfan.ray_ids) == self.M

    # --- compute_rayids ---

    def test_compute_rayids_returns_strings(self, simple_rayfan):
        simple_rayfan.compute_rayids()
        assert all(isinstance(rid, str) for rid in simple_rayfan.ray_ids)

    def test_compute_rayids_length(self, simple_rayfan):
        simple_rayfan.compute_rayids()
        assert len(simple_rayfan.ray_ids) == len(simple_rayfan.thetas)

    # --- __len__ ---

    def test_len(self, simple_rayfan):
        assert len(simple_rayfan) == self.M

    # --- __getitem__ int ---

    def test_getitem_int_returns_ray(self, simple_rayfan):
        ray = simple_rayfan[0]
        assert isinstance(ray, Ray)

    def test_getitem_int_correct_index(self, simple_rayfan):
        ray = simple_rayfan[1]
        np.testing.assert_array_equal(ray.r, simple_rayfan.rs[1])

    def test_getitem_negative_int(self, simple_rayfan):
        ray = simple_rayfan[-1]
        assert isinstance(ray, Ray)
        np.testing.assert_array_equal(ray.r, simple_rayfan.rs[-1])

    def test_getitem_out_of_bounds_raises_index_error(self, simple_rayfan):
        with pytest.raises(IndexError):
            _ = simple_rayfan[100]

    # --- __getitem__ slice ---

    def test_getitem_slice_returns_rayfan(self, simple_rayfan):
        result = simple_rayfan[0:2]
        assert isinstance(result, RayFan)
        assert len(result) == 2

    def test_getitem_slice_correct_thetas(self, simple_rayfan):
        result = simple_rayfan[1:]
        np.testing.assert_array_equal(result.thetas, simple_rayfan.thetas[1:])

    # --- __getitem__ bool mask ---

    def test_getitem_bool_mask_returns_rayfan(self, simple_rayfan):
        mask = np.array([True, False, True])
        result = simple_rayfan[mask]
        assert isinstance(result, RayFan)
        assert len(result) == 2

    def test_getitem_bool_mask_correct_subset(self, simple_rayfan):
        mask = np.array([False, True, False])
        result = simple_rayfan[mask]
        np.testing.assert_array_equal(result.thetas, simple_rayfan.thetas[1:2])

    # --- __getitem__ int array ---

    def test_getitem_int_array_returns_rayfan(self, simple_rayfan):
        idx = np.array([0, 2])
        result = simple_rayfan[idx]
        assert isinstance(result, RayFan)
        assert len(result) == 2
        np.testing.assert_array_equal(result.thetas,
                                      simple_rayfan.thetas[np.array([0, 2])])

    # --- __add__ ---

    def test_add_correct_length(self):
        rays_a = self._make_rays(M=2)
        rays_b = self._make_rays(M=3)
        rf_a = RayFan(rays_a)
        rf_b = RayFan(rays_b)
        result = rf_a + rf_b
        assert len(result) == 5

    def test_add_rs_preserved(self):
        rays_a = self._make_rays(M=2)
        rays_b = self._make_rays(M=1)
        rf_a = RayFan(rays_a)
        rf_b = RayFan(rays_b)
        result = rf_a + rf_b
        # All rays should have the same range array
        for i in range(len(result)):
            np.testing.assert_array_equal(result.rs[i], rf_a.rs[0])

    def test_add_incompatible_ranges_raises_value_error(self):
        rays_a = self._make_rays(M=1, R=10000.0)
        rays_b = self._make_rays(M=1, R=20000.0)
        rf_a = RayFan(rays_a)
        rf_b = RayFan(rays_b)
        with pytest.raises(ValueError):
            _ = rf_a + rf_b

    def test_add_non_rayfan_raises_type_error(self, simple_rayfan):
        with pytest.raises(TypeError):
            _ = simple_rayfan + 42

    # --- save_mat ---

    def test_save_mat_creates_file(self, simple_rayfan, tmp_path):
        path = str(tmp_path / 'test_rayfan.mat')
        simple_rayfan.save_mat(path)
        assert (tmp_path / 'test_rayfan.mat').exists()

    def test_save_mat_loadable(self, simple_rayfan, tmp_path):
        path = str(tmp_path / 'test_rayfan.mat')
        simple_rayfan.save_mat(path)
        data = scipy.io.loadmat(path)
        assert 'rayfan' in data

    def test_save_mat_contains_required_keys(self, simple_rayfan, tmp_path):
        path = str(tmp_path / 'test_rayfan.mat')
        simple_rayfan.save_mat(path)
        data = scipy.io.loadmat(path)
        rayfan = data['rayfan']
        # MATLAB struct stored as structured array; check dtype field names
        expected_keys = {'thetas', 'xs', 'ts', 'zs', 'ps',
                         'n_botts', 'n_surfs', 'source_depths'}
        actual_keys = set(rayfan.dtype.names)
        assert expected_keys <= actual_keys

    def test_save_mat_values_match(self, simple_rayfan, tmp_path):
        path = str(tmp_path / 'test_rayfan.mat')
        simple_rayfan.save_mat(path)
        data = scipy.io.loadmat(path)
        rayfan = data['rayfan']
        saved_thetas = rayfan['thetas'][0, 0].flatten()
        np.testing.assert_allclose(saved_thetas, simple_rayfan.thetas,
                                   atol=1e-10)

    # --- Plotting smoke tests ---

    def test_plot_ray_fan_smoke(self, simple_rayfan):
        plt.figure()
        simple_rayfan.plot_ray_fan()
        plt.close('all')

    def test_plot_time_front_smoke(self, simple_rayfan):
        plt.figure()
        simple_rayfan.plot_time_front()
        plt.close('all')

    def test_plot_time_front_include_lines_smoke(self, simple_rayfan):
        plt.figure()
        simple_rayfan.plot_time_front(include_lines=True)
        plt.close('all')

    def test_plot_depth_v_angle_smoke(self, simple_rayfan):
        plt.figure()
        simple_rayfan.plot_depth_v_angle()
        plt.close('all')
