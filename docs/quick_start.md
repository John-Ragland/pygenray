# Quick Start

## Installation
```bash
pip install pygenray
```

## Run a simple ray code
Environment specification is handled with the `ocean_acoustic_environment` package.
The default environment is a munk sound speed profile for 1000 km.

```python
import ocean_acoustic_environment as oaenv
import pygenray as pr

envi = oaenv.OceanEnvironment2D(flat_earth_transform=False)
thetas = np.linspace(-15,15,6000)# 6 thousand launch angles

result = pr.shoot_ray(
    source_depth = 1000, # m
    source_range = 0, # m
    thetas=thetas,
    reciever_range = 200e3, #m
    x_eval = np.linspace(0,200e3,10000) # range points to save ray position,
    environment = envi, # environment specification
    rtol=1e-9, # relative tolerance of numerical integrator
)
```


