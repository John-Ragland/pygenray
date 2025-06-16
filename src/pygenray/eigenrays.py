"""
Tools and methods for calculating eigenrays for specifed receiver depths.
"""
import pygenray as pr
import ocean_acoustic_env as oeanv
import numpy as np


def find_eigenrays(rays, receiver_depths, source_depth, source_range, receiver_range, x_eval, envi_munk, ztol=1, max_iter=20):
    '''
    Given an initial ray fan, find eigenrays with bisection method of root finding.

    Parameters
    ----------
    rays : pr.RayFan
        RayFan object containing sweep of rays to be used for finding eigenrays. Can be computed with `pr.shoot_rays()`.
    receiver_depths : array like
        one dimensional array, or list containing reciever depths
    source_depth : float
        source depth in meters
    source_range : float
        source range in meters
    receiver_range : float
        receiver range in meters
    x_eval : np.array
        one dimensional array of range values to evaluate rays at
    envi_munk : oeanv.OceanAcousticEnvironment
        OceanAcousticEnvironment object containing environment parameters for ray tracing.
    ztol : float, optional
        depth tolerance for eigenrays, by default 1 m
    max_iter : int, optional
        maximum number of iterations for bisection method, by default 20

    Returns
    -------
    erays : dict
        dictionary of eigen rays. Key values are values in `receiver_depths`.
    '''
    erays = {}

    for rd_idx, receiver_depth in enumerate(receiver_depths):
        print(f'Reciever depth: {receiver_depth} [m]')

        ## get initial bracketing rays
        # get indices before sign changes
        depth_sign = np.sign(rays.zs[:,-1] - receiver_depth)
        sign_change = np.diff(depth_sign)
        bracket_idxs_start = np.where(sign_change)[0]

        # Get bracket indices
        bracket_idxs = np.column_stack([bracket_idxs_start, bracket_idxs_start + 1])
        print(f'trying to find {len(bracket_idxs)} eigen rays...')

        # compute bisection launch angles
        z1s = rays.zs[bracket_idxs[:,0].astype(int),-1]
        z2s = rays.zs[bracket_idxs[:,1].astype(int),-1]
        theta1s = rays.thetas[bracket_idxs[:,0].astype(int)]
        theta2s = rays.thetas[bracket_idxs[:,1].astype(int)]

        bisection_thetas = theta1s + (theta2s - theta1s) * ((receiver_depth - z1s) / (z2s - z1s))

        erays[rd_idx] = []
        # Solve for each eigen ray at receiver depth
        for k in range(len(bisection_thetas)):
            iter_count = 0

            z1 = z1s[k]
            z2 = z2s[k]

            within_tolerance = False
            bisection_theta = bisection_thetas[k]

            # Bisection root finding loop
            while not within_tolerance:
                ray = pr.shoot_ray(source_depth, source_range, bisection_theta, receiver_range, x_eval, envi_munk)
                

                if np.abs(ray.z[-1] - receiver_depth) < ztol:
                    erays[rd_idx].append(ray)
                    within_tolerance = True

                if ray.z[-1] < receiver_depth:
                    z1 = z1
                    z2 = ray.z[-1]
                    theta1 = theta1s[k]
                    theta2 = bisection_theta
                else:
                    z1 = ray.z[-1]
                    z2 = z2
                    theta1 = bisection_theta
                    theta2 = theta2s[k]
                
                bisection_theta = theta1 + (theta2 - theta1) * ((receiver_depth - z1) / (z2 - z1))

                if iter_count > max_iter:
                    print(f'Failed to find eigen ray for receiver depth {receiver_depth} [m] and approximate launch angle {bisection_thetas[k]} [m] after {max_iter} iterations.')
                    within_tolerance = True
                    break
                iter_count += 1

    # convert lists of rays to RayFan objects
    for rdepth in erays:
        erays[rdepth] = pr.RayFan(erays[rdepth])
    return erays

__all__ = ['find_eigenrays']