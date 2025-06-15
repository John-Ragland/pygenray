# Building Notes
*A place for me to make notes while I build the package*


## List of features that would be nice to be made accessible to user
- numerical integrator
    - this would also come with all subsequent parameters that the numerical solver needs
    - The first step, would probably be to just include rk45 and a 2nd order leapfrog
- compiler info (used within numba jit)
    - is this something that would actually be useful to change?
    - do you use fastmath or not

## JAX or Julia?
- Jax has the highest ceiling, Julia has the highest floor.
- My current game plan is to shift course and try to get something built with Julia
    - I was running into some issues with the Event detection in diffrax, where the numerical root solver for detecting the event for the surface reflection was failing for some reason. I'm not completely sure what might be causing this, but in order to avoid looming significant pain, I think its wise to switch to the mature Julia and get *something* that works. I can always port it over to jax later.

### ... or scipy?
actually... scipy might be worth it. Because I can have the ray code done in just a few days. I could still get the benefit of parelizing across cpu cores (which fits better into my current understanding of how to accelerate, compared to GPUs). I would get something working, and then could always go back and play around with adding jax support, which probably wouldn't be *that* hard. There's still a lot of optimization to be had with regular ole numba. And then I can use scipy.ode which is battle tested instead of diffrax.


## Scipy Build
**To Do**
- [ ] verify that bottom reflection angles are correct
- [x] understand the root finding for eigen rays
- [ ] implement eigen ray calculation
- [ ] Build reasonable error handling
- [ ] Build ray object that makes sense


### eigen-ray road map
- shoot ray fan, save as *ray object*
- for a given receiver depth, find all bracketing ray angles
    - i.e. plot ray depth vs ray launch angle and find every depth crossing
- do root finding using bisection method
    - find linear zero crossing launch angle
    - launch that ray
    - use launched ray and previous bracketing rays and repeat until convergence


**where I left off**: I have built out `shoot_rays` and `shoot_ray`, and created a single method that can be called from within the functions that take an environment as input, and when accessing shared memory. But I haven't tested all of that functionality yet.