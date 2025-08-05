"""
Tools and methods for calculating eigenrays for specifed receiver depths.
"""
import pygenray as pr
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def find_eigenrays(rays, receiver_depths, source_depth, source_range, receiver_range, num_range_save, environment, ztol=1, max_iter=20, num_workers=None):
    '''
    Given an initial ray fan, find eigenrays with bisection method of root finding.

    Parameters
    ----------
    rays : pr.RayFan
        RayFan object containing sweep of rays to be used for finding eigenrays. Can be computed with `pr.shoot_rays()`.
    receiver_depths : array like
        one dimensional array, or list containing receiver depths
    source_depth : float
        source depth in meters
    source_range : float
        source range in meters
    receiver_range : float
        receiver range in meters
    num_range_save : int
        number of range values to save the ray state at
    environment : pr.OceanEnvironment2D
        OceanEnvironment2D object containing environment parameters for ray tracing.
    ztol : float, optional
        depth tolerance for eigenrays, by default 1 m
    max_iter : int, optional
        maximum number of iterations for bisection method, by default 20
    num_workers : int, optional
        number of workers for parallel processing, by default None (uses all available cores, i.e. `mp.cpu_count()`)

    Returns
    -------
    erays : dict
        dictionary of eigen rays. Key values are values in `receiver_depths`.
    '''
    erays_dict = {}
    num_eigenrays = {}
    num_eigenrays_found = {}

    for rd_idx, receiver_depth in enumerate(tqdm(receiver_depths, desc="Finding Eigenrays")):
        ## get initial bracketing rays
        # get indices before sign changes
        depth_sign = np.sign(rays.zs[:,-1] - receiver_depth)
        sign_change = np.diff(depth_sign)
        bracket_idxs_start = np.where(sign_change)[0]

        # Get bracket indices
        bracket_idxs = np.column_stack([bracket_idxs_start, bracket_idxs_start + 1])

        # compute bisection launch angles
        z1s = rays.zs[bracket_idxs[:,0].astype(int),-1]
        z2s = rays.zs[bracket_idxs[:,1].astype(int),-1]
        theta1s = -rays.thetas[bracket_idxs[:,0].astype(int)]
        theta2s = -rays.thetas[bracket_idxs[:,1].astype(int)]

        bisection_thetas = theta1s + (theta2s - theta1s) * ((receiver_depth - z1s) / (z2s - z1s))
        
        num_eigenrays[receiver_depth] = len(bisection_thetas)

        # Solve for each eigen ray at receiver depth
        erays_dict[rd_idx] = []
        if len(bisection_thetas) > 20: # use parallel processing for large number of rays
            # contsruct argment iterable for parellel processing
            args_list = []
            for k in range (len(bisection_thetas)):
                args = (k, z1s[k], z2s[k], theta1s[k], theta2s[k], bisection_thetas[k],
                        receiver_depth, source_depth, source_range, receiver_range,
                        num_range_save, environment, ztol, max_iter)
                args_list.append(args)

            # map individual eigen ray finding to different workers
            if num_workers is None:
                num_workers = mp.cpu_count()
            with mp.Pool(num_workers) as pool:
                results = list(tqdm(pool.imap(_find_single_eigenray, args_list), total=len(args_list), desc="Finding eigenrays"))
            
            # Filter out None results and add successful rays
            for result in results:
                if result is not None:
                    erays_dict[rd_idx].append(result)
        
        else:  # use sequential processing for small number of rays
            for k in range(len(bisection_thetas)):
                ray = _find_single_eigenray((k, z1s[k], z2s[k], theta1s[k], theta2s[k], bisection_thetas[k],
                                             receiver_depth, source_depth, source_range, receiver_range,
                                             num_range_save, environment, ztol, max_iter))
                if ray is not None:
                    erays_dict[rd_idx].append(ray)
        num_eigenrays_found[rd_idx] = len(erays_dict[rd_idx])

    # Create EigenRays object after processing all receiver depths
    erays = pr.EigenRays(receiver_depths, erays_dict, environment, num_eigenrays, num_eigenrays_found)
    return erays


def _find_single_eigenray(args):
    """
    Find single Eigen ray given the bracketing ray depths, and launch angles.
    """
    k, z1, z2, theta1, theta2, bisection_theta, receiver_depth, source_depth, source_range, receiver_range, num_range_save, environment, ztol, max_iter = args
    
    iter_count = 0
    within_tolerance = False
    
    # Bisection root finding loop
    while not within_tolerance:
        ray = pr.shoot_ray(source_depth, source_range, bisection_theta, receiver_range, num_range_save, environment)

        if ray is None:
            print(f'Failed to find eigen ray for receiver depth {receiver_depth} [m] and approximate launch angle {bisection_theta} [m] ray θ = 90°')
            return None
        
        if np.abs(ray.z[-1] - receiver_depth) < ztol:
            return ray

        if ray.z[-1] < receiver_depth:
            z1 = z1
            z2 = ray.z[-1]
            theta1 = theta1
            theta2 = bisection_theta
        else:
            z1 = ray.z[-1]
            z2 = z2
            theta1 = bisection_theta
            theta2 = theta2
        
        bisection_theta = theta1 + (theta2 - theta1) * ((receiver_depth - z1) / (z2 - z1))

        if iter_count > max_iter:
            print(f'Failed to find eigen ray for receiver depth {receiver_depth} [m] and approximate launch angle {bisection_theta} [m] after {max_iter} iterations.')
            return None
        iter_count += 1
    
    return None


__all__ = ['find_eigenrays']