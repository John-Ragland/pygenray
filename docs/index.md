# PygenRay
rhymes with "eigen-ray"

Modern python based ray code for modelling underwater acoustic propagation, built on top of existing numerical integration methods in scipy :mod:`scipy.integrate`. This allows for more specific control over the numerical integration methods used for solving the ray equations.

```{warning}
Some of the main functionality of this package is still under development. If you run into any errors, please feel free to submit an issue. I may or may not be able to get to it, so you might also have to play around with trying to fix it yourself.


```{toctree}
:maxdepth: 2
:hidden:

api

```

## Quick Start

### Installation
```bash
pip install pygenray
```

### Run a simple ray code

Set up the environment, model parameters and shoot a ray fan.
```python
import pygenray as pr
import numpy as np
import matplotlib.pyplot as plt

source_depth = 1000
source_range = 0
launch_angles = np.linspace(-20, 20, 1000)  # launch angles in degrees
receiver_range = 100e3  # receiver range in meters
num_range_save = 10000  # number of range points to save
environment = pr.OceanEnvironment2D()
receiver_depths=[1000]

rays = pr.shoot_rays(
    source_depth,
    source_range,
    launch_angles,
    receiver_range,
    num_range_save,
    environment,
)
```

Solve for eigen rays
```python
eigenrays = pr.find_eigenrays(
    rays,
    receiver_depths,
    source_depth,
    source_range,
    receiver_range,
    num_range_save,
    environment
)
```

Plot time-front, arrival times and angles, and eigenray paths.
```python
fig = plt.figure(figsize=(13,3))

plt.subplot(1,3,1)
rays.plot_time_front()

plt.subplot(1,3,2)
environment.plot(cmap='plasma')
eigenrays.plot(c='w')

plt.ylim([5000,0])
plt.subplot(1,3,3)
eigenrays.plot_angle_time()

plt.tight_layout()
```

![img](../imgs/getting_started1.png)

## Indices and tables
- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
