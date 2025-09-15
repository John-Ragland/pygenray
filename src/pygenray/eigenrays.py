"""
Tools and methods for calculating eigenrays for specifed receiver depths.
"""
import pygenray as pr
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def find_eigenrays(
        rays,
        receiver_depths,
        source_depth,
        source_range,
        receiver_range,
        num_range_save,
        environment,
        ztol=1,
        max_iter=20,
        num_workers=None,
        **kwargs,
    ):
    '''
    Given an initial ray fan, find eigenrays with [regula falsi](https://en.wikipedia.org/wiki/Regula_falsi#The_regula_falsi_(false_position)_method) method of root finding.

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
        maximum number of root finding iterations, default 20
    num_workers : int, optional
        number of workers for parallel processing, by default None (uses all available cores, i.e. `mp.cpu_count()`)
    kwargs : keyword arguments
        additional keyword arguments passed to `pr.shoot_ray`

    Returns
    -------
    erays : dict
        dictionary of eigen rays. Key values are values in `receiver_depths`.
    '''
    erays_dict = {}
    num_eigenrays = {}
    num_eigenrays_found = {}
    failed_eray_theta_brackets = {}

    for rd_idx, receiver_depth in enumerate(receiver_depths):
        ## get initial bracketing rays
        # get indices before sign changes
        depth_sign = np.sign(rays.zs[:,-1] + receiver_depth)
        sign_change = np.diff(depth_sign)
        bracket_idxs_start = np.where(sign_change)[0]

        num_eigenrays[receiver_depth] = len(bracket_idxs_start)

        # Get bracket indices
        bracket_idxs = np.column_stack([bracket_idxs_start, bracket_idxs_start + 1])

        # compute regula falsi launch angles
        z1s = rays.zs[bracket_idxs[:,0].astype(int),-1]
        z2s = rays.zs[bracket_idxs[:,1].astype(int),-1]

        theta1s = rays.thetas[bracket_idxs[:,0].astype(int)]
        theta2s = rays.thetas[bracket_idxs[:,1].astype(int)]

        erays_dict[rd_idx] = []
        failed_eray_theta_brackets[rd_idx] = []

        # This code block checks if ray fan already contains ray within depth tolerance.
        # If so, that eigenray is removed from the search and the ray fan ray is used.
        # This block works, but if num_range_save is different between RayFan and `find_eigenrays`
        # there is a bug that breaks re-shooting eigenrays
        # time efficiency that this block saves is not worth finding the bug at this time.
        # will likely be removed in the future.
        """
        eray_found_idxs = []
        for eray_idx in range(num_eigenrays[receiver_depth]):
            z1_distance = np.abs(z1s[eray_idx] + receiver_depth)
            z2_distance = np.abs(z2s[eray_idx] + receiver_depth)
      
            if (z1_distance < ztol) or (z2_distance < ztol):
                bracket_winner = np.argmin([z1_distance, z2_distance])

                ray = rays[int(bracket_idxs[eray_idx, bracket_winner])]

                if num_range_save == len(ray.r):
                    erays_dict[rd_idx].append(ray)
                else:
                    print(ray.launch_angle)
                    erays_dict[rd_idx].append(pr.shoot_ray(source_depth, source_range, -ray.launch_angle, receiver_range, num_range_save, environment, **kwargs))

                eray_found_idxs.append(eray_idx)
            else:
                continue
        
        # remove found eigenrays from search
        z1s = np.delete(z1s, eray_found_idxs)
        z2s = np.delete(z2s, eray_found_idxs)
        theta1s = np.delete(theta1s, eray_found_idxs)
        theta2s = np.delete(theta2s, eray_found_idxs)
        """

        regula_falsi_thetas =  theta1s - (z1s + receiver_depth) * (theta2s - theta1s) / (z2s - z1s)

        if len(regula_falsi_thetas) > 5: # use parallel processing for large number of rays
            # construct argument iterable for parallel processing
            args_list = []
            for k in range (len(regula_falsi_thetas)):
                args = (k, z1s[k], z2s[k], theta1s[k], theta2s[k], regula_falsi_thetas[k],
                        receiver_depth, source_depth, source_range, receiver_range,
                        num_range_save, environment, ztol, max_iter, kwargs)
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
                else:
                    failed_eray_theta_brackets[rd_idx].append((theta1s[k], theta2s[k]))

        else:  # use sequential processing for small number of rays
            for k in tqdm(range(len(regula_falsi_thetas)), desc='Finding eigenrays:'):
                ray = _find_single_eigenray((k, z1s[k], z2s[k], theta1s[k], theta2s[k], regula_falsi_thetas[k],
                                             receiver_depth, source_depth, source_range, receiver_range,
                                             num_range_save, environment, ztol, max_iter, kwargs))
                if ray is not None:
                    erays_dict[rd_idx].append(ray)
                else:
                    failed_eray_theta_brackets[rd_idx].append((theta1s[k], theta2s[k]))

        num_eigenrays_found[rd_idx] = len(erays_dict[rd_idx])

    # Create EigenRays object after processing all receiver depths
    erays = pr.EigenRays(receiver_depths, erays_dict, environment, num_eigenrays, num_eigenrays_found, failed_eray_theta_brackets)
    return erays


def _find_single_eigenray(args):
    """
    Find single Eigen ray given the bracketing ray depths, and launch angles.
    """
    k, z1, z2, theta1, theta2, regula_falsi_theta, receiver_depth, source_depth, source_range, receiver_range, num_range_save, environment, ztol, max_iter, kwargs = args
    
    iter_count = 0
    # Regula Falsi root finding loop
    while True:

        ray = pr.shoot_ray(source_depth, source_range, regula_falsi_theta, receiver_range, num_range_save, environment, **kwargs)

        if ray is None:
            print(f'Failed to find eigen ray for receiver depth {receiver_depth} [m] and approximate launch angle {regula_falsi_theta} [m] ray θ = 90°')
            return None
        
        if np.abs(ray.z[-1] + receiver_depth) < ztol:

            # flip launch angle to match sign convention
            ray.launch_angle = -ray.launch_angle
            return ray

        # Ray is on z1 side of receiver
        if np.sign(ray.z[-1] + receiver_depth) == np.sign(z1 + receiver_depth):
            z1 = ray.z[-1]
            theta1 = regula_falsi_theta
        # Ray is on z2 side of receiver
        else:
            z2 = ray.z[-1]
            theta2 = regula_falsi_theta

        regula_falsi_theta =  theta1 - (z1 + receiver_depth) * (theta2 - theta1) / (z2 - z1)

        if iter_count > max_iter:
            return None
        
        iter_count += 1



__all__ = ['find_eigenrays']