# PyGenRay

Python package for simulated 2D ray-based acoustic propagation modeling in underwater environments, designed for acoustic tomography applications.

## Environment

Use `uv` for all environment and package management — never `pip` or `python` directly.

```bash
uv sync --group dev   # install with dev dependencies
uv run pytest tests/  # run tests
```

## Testing

```bash
uv run pytest tests/              # full suite
uv run pytest tests/ --cov=pygenray  # with coverage
uv run pytest --regenerate-physics tests/  # regenerate physics regression fixtures
```

## Architecture

Source lives in `src/pygenray/`. Key modules:

- `environment.py` — `OceanEnvironment2D`, sound speed profiles (e.g. `munk_ssp()`)
- `launch_rays.py` — `shoot_rays()` (fan, parallel), `shoot_ray()` (single)
- `ray_objects.py` — `Ray` and `RayFan` data classes with plotting methods
- `eigenrays.py` — `find_eigenrays()` using regula falsi root-finding
- `integration_processes.py` — core ray ODEs (`derivsrd()`), Numba JIT-compiled

## Coordinate Convention

There is a sign flip between the user-facing API and internal ODE integration:

- **User-facing:** positive `z` = downward (ocean depth), positive launch angle = toward surface
- **Internal ODE:** negative `z` convention

Respect this distinction when modifying ray integration or `Ray`/`RayFan` attribute handling.

## Performance

Hot-path interpolation functions (`bilinear_interp`, `linear_interp`) are decorated with `@jax.jit`; `derivsrd` and event conditions are compiled by diffrax as part of the ODE solver's XLA computation. Avoid introducing pure-Python loops in hot paths. `shoot_rays()` loops serially with tqdm (`n_processes` is deprecated/ignored); vmap support is planned.
