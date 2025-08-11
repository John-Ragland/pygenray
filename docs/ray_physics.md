# Ray Physics

**coordinate convention**
- Environment depth and source / receiver depths are specified with positive z, defined by distance from surface of ocean in meters.
- The ray state, $y$, defined by $y = \left[\mathrm{T, z, p_z} \right]^T$, uses sign convention where z is negative and increases in z correspond to moving towards the surface.
    - Subsequently, a positive ray angle corresponds to a ray moving towards the surface.
    - A positive ray ID corresponds to a positive launch angle (towards the surface)