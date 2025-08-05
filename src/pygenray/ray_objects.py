import numpy as np
from matplotlib import pyplot as plt
import pygenray as pr
from scipy import io

class Ray:
    """
    Single Ray Object - python object that store all parameters associated with a single ray.
    """
    def __init__(
        self,
        r : float,
        y : np.array,
        n_bottom : int,
        n_surface : int,
        launch_angle : float = None,
        source_depth : float = None,
    ):
        """
        Parameters
        ----------
        r : float
            Range value for ray
        y : np.array
            ray variables (3,) [travel time, depth, ray parameter (sin(θ)/c)]
        launch_angle : float
            launch angle of ray
        n_bottom : int
            number of bottom reflections for ray
        n_surface : int
            number of surface reflections for ray
        launch_angle : float
            launch angle of ray

        Attributes
        ----------
        r : np.array
        t : np.array
        z : np.array
        p : np.array
        n_bottom : np.array
        n_surface : np.array
        launch_angle : np.array
        source_depth : np.array
        """

        self.r = r
        self.t = y[0,:]
        self.z = y[1,:]
        self.p = y[2,:]
        self.n_bottom = n_bottom
        self.n_surface = n_surface
        if launch_angle is not None:
            self.launch_angle = launch_angle
        if source_depth is not None:
            self.source_depth = source_depth
        return

    def plot(self,**kwargs):
        """
        Plot ray in time-depth space
        """

        plot_kwargs = {'c':'k', 'lw': 1, 'alpha': 0.5}
        plot_kwargs.update(kwargs)
        plt.plot(self.r, self.z, **kwargs)
        plt.xlabel('time [s]')
        plt.ylabel('depth [m]')
        plt.ylim([self.z.max(), self.z.min()])
        return


class RayFan:
    """
    RayFan Object - python object that store all parameters associated with a ray fan.
    """
    def __init__(
        self,
        Rays : list
    ):
        """
        Parameters
        ----------
        Rays : list
            List of `Ray` objects

        Attributes
        ----------
        thetas : np.array
            launch angles (M,) of rays in fan
        rs : np.array
            range values of rays. (M,N): M is launch angle dimension, N is range step dimension
        ts : np.array
            travel times of rays. (M,N): M is launch angle dimension, N is range step dimension
        zs : np.array
            depths of rays in fan. (M,N): M is launch angle dimension, N is range step dimension
        ps : np.array
            ray parameters (sin(θ)/c) of rays. (M,N): M is launch angle dimension, N is range step dimension
        n_botts : np.array
            number of bottom reflections for rays. (M,) where M is launch angle dimension
        n_surfs : np.array
            number of surface reflections for rays, (M,) where M is launch angle dimension
        source_depths : np.array
            source depth for each launched ray, (M,) where M is launch angle
        """
        # unpack and combine Rays
        thetas = []
        rs = []
        ts = []
        zs = []
        ps = []
        n_botts = []
        n_surfs = []
        source_depths = []

        for Ray in Rays:
            thetas.append(Ray.launch_angle)
            rs.append(Ray.r)
            ts.append(Ray.t)
            zs.append(Ray.z)
            ps.append(Ray.p)
            n_botts.append(Ray.n_bottom)
            n_surfs.append(Ray.n_surface)
            source_depths.append(Ray.source_depth)

        self.thetas = np.array(thetas)
        self.rs = np.array(rs)
        self.ts = np.array(ts)
        self.zs = np.array(zs)
        self.ps = np.array(ps)
        self.n_botts = np.array(n_botts)
        self.n_surfs = np.array(n_surfs)
        self.source_depths = np.array(source_depths)
        
        return

    def plot_time_front(self,include_lines=False, range_idx=-1, **kwargs):
        '''
        plot time front. Key word arguments are passed to plt.scatter.

        Parameters
        ----------
        include_lines : bool
            if True, lines between received rays on time plots are plotted
        '''

        if include_lines:
            plt.plot(self.ts[:,range_idx], self.zs[:,range_idx], c='#aaaaaa', lw=0.5, zorder=5)


        scatter_kwargs = {'c': self.thetas, 'cmap': 'managua', 's': 2, 'lw': 0, 'zorder': 6}
        scatter_kwargs.update(kwargs)
        plt.scatter(x=self.ts[:,range_idx], y=self.zs[:,range_idx], **scatter_kwargs)


        plt.ylim([self.zs.max(), self.zs.min()])
        plt.colorbar(label='launch angle [degrees]')
        plt.xlabel('time [s]')
        plt.ylabel('depth [m]')
        plt.title('Time Front')
    
    def plot_ray_fan(self,**kwargs):
        '''
        plot ray fan
        '''

        # set alpha value
        if (10*1/len(self.thetas) > 1) | (10*1/len(self.thetas) < 0):
            alpha_val=1
        else:
            alpha_val = 10*1/len(self.thetas)

        plot_kwargs = {'c':'k', 'lw': 1, 'alpha': alpha_val}
        plot_kwargs.update(kwargs)
        _ = plt.plot(self.rs.T, self.zs.T, **plot_kwargs)

        plt.xlabel('range [m]')
        plt.ylabel('depth [m]')
        plt.ylim([self.zs.max(), self.zs.min()])
        plt.title('Ray Fan')

        return
    
    def plot_depth_v_angle(self,include_line=False, **kwargs):

        scatter_kwargs = {'c': self.thetas, 'cmap': 'managua', 's': 2, 'lw': 0, 'zorder': 6}
        scatter_kwargs.update(kwargs)
        if include_line:
            plt.plot(self.thetas, self.zs[:,-1], c='#aaaaaa', lw=0.5, zorder=5)
        plt.scatter(x=self.thetas, y=self.zs[:,-1],**kwargs)

        return
    
    def save_mat(self, filename):
        """
        Save RayFan object to a .mat file.

        Parameters
        ----------
        filename : str
            Name of the output .mat file to save the RayFan data.
        
        """

        # Create a dictionary to hold the data
        data = {'rayfan': {
            'thetas': self.thetas,
            'xs': self.rs,
            'ts': self.ts,
            'zs': self.zs,
            'ps': self.ps,
            'n_botts': self.n_botts,
            'n_surfs': self.n_surfs,
            'source_depths': self.source_depths
        }}

        # Save the dictionary to a .mat file
        io.savemat(filename, data)

    def __add__(self, other):
        """
        Add two RayFan objects by concatenating along the launch angle dimension (M).
        
        Parameters
        ----------
        other : RayFan
            Another RayFan object to add to this one
            
        Returns
        -------
        RayFan
            New RayFan object with concatenated rays
            
        Raises
        ------
        TypeError
            If other is not a RayFan object
        ValueError
            If the range arrays (rs) are not compatible
        """
        if not isinstance(other, RayFan):
            raise TypeError("Can only add RayFan objects together")
        
        # Check if range arrays are compatible
        if not np.array_equal(self.rs[0], other.rs[0]):
            raise ValueError("Range arrays (rs) must be equivalent for concatenation")
        
        # Create combined Ray objects
        combined_rays = []
        
        # Add rays from self
        for i in range(len(self.thetas)):
            ray = Ray(
                r=self.rs[i],
                y=np.array([self.ts[i], self.zs[i], self.ps[i]]),
                n_bottom=self.n_botts[i],
                n_surface=self.n_surfs[i],
                launch_angle=self.thetas[i],
                source_depth=self.source_depths[i]
            )
            combined_rays.append(ray)
        
        # Add rays from other
        for i in range(len(other.thetas)):
            ray = Ray(
                r=other.rs[i],
                y=np.array([other.ts[i], other.zs[i], other.ps[i]]),
                n_bottom=other.n_botts[i],
                n_surface=other.n_surfs[i],
                launch_angle=other.thetas[i],
                source_depth=other.source_depths[i]
            )
            combined_rays.append(ray)
        
        return RayFan(combined_rays)

    def __len__(self):
        """
        Return the number of rays in the RayFan.
        
        Returns
        -------
        int
            Number of rays (length of launch angle dimension M)
        """
        return len(self.thetas)

    def __getitem__(self, key):
        """
        Slice the RayFan along the launch angle dimension (M).
        
        Parameters
        ----------
        key : int, slice, or array-like
            Index or slice to select rays. Can be:
            - int: single ray index
            - slice: slice object (e.g., 0:10:2)
            - array-like: boolean mask or integer indices
            
        Returns
        -------
        RayFan
            New RayFan object with selected rays
        """
        # Create Ray objects for the selected indices
        selected_rays = []
        
        # Handle the slicing to get the indices
        if isinstance(key, (int, slice)):
            # Use numpy's advanced indexing to handle slices
            selected_indices = np.arange(len(self.thetas))[key]
        else:
            # Handle array-like indices (boolean masks or integer arrays)
            selected_indices = np.asarray(key)
            if selected_indices.dtype == bool:
                selected_indices = np.where(selected_indices)[0]
        
        # Ensure selected_indices is iterable (convert single int to array)
        if np.isscalar(selected_indices):
            selected_indices = [selected_indices]
        
        # Create Ray objects for selected indices
        for i in selected_indices:
            ray = Ray(
                r=self.rs[i],
                y=np.array([self.ts[i], self.zs[i], self.ps[i]]),
                n_bottom=self.n_botts[i],
                n_surface=self.n_surfs[i],
                launch_angle=self.thetas[i],
                source_depth=self.source_depths[i]
            )
            selected_rays.append(ray)
        
        return RayFan(selected_rays)


class EigenRays:
    '''
    EigenRays Object - python object that store all parameters associated with eigen rays for given receiver depths

    Parameters
    ----------
    receiver_depths : list
        List of receiver depths for which eigen rays are computed.
    eigenray_dict : dict
        dictionary of eigen rays. Key values are indices of receiver depths, and values are lists of pr.Ray objects.
    environment : pr.OceanEnvironment2D
        OceanEnvironment2D environment used for ray tracing.

    Attributes
    ----------
    receiver_depths : list
        List of receiver depths for which eigen rays are computed. This is used to index the eigen rays.
    source_depth : float
        source depth for eigen rays.
    rs : dict
        dictionary of eigenray ranges. keys are range depth indices. values are arrays of shape (M,N), where M is number of eigen rays and N is number of range steps
    ts : dict
        dictionary of eigenray times. keys are range depth indices. values are arrays of shape (M,N), where M is number of eigen rays and N is number of range steps
    zs : dict
        dictionary of eigenray depths. keys are range depth indices. values are arrays of shape (M,N), where M is number of eigen rays and N is number of range steps
    ps : dict
        dictionary of eigenray ray parameters (sin(θ)/c). keys are range depth indices. values are arrays of shape (M,N), where M is number of eigen rays and N is number of range steps
    received_angles : dict
        dictionary of eigenray launch angles. keys are range depth indices. values are arrays of shape (M,), where M is number of eigen rays
    launch_angles : dict
        dictionary of eigenray launch angles. keys are range depth indices. values are arrays of shape (M,), where M is number of eigen rays
    n_bottom : dict
        dictionary of number of bottom reflections for eigen rays. keys are range depth indices. values are arrays of shape (M,), where M is number of eigen rays
    n_surface : dict
        dictionary of number of surface reflections for eigen rays. keys are range depth indices. values are arrays of shape (M,), where M is number of eigen rays

    '''

    def __init__(self,receiver_depths, eigenray_dict, environment, num_eigenrays, num_eigenrays_found):
        self.receiver_depths = receiver_depths

        self.rs = {}
        self.ts = {}
        self.zs = {}
        self.ps = {}
        self.received_angles = {}
        self.launch_angles = {}
        self.ray_id = {}
        self.num_eigenrays = num_eigenrays
        self.num_eigenrays_found = num_eigenrays_found

        for ridx in range(len(receiver_depths)):
            # use ray fan concatenation to construct arrays
            eray_fan = RayFan(eigenray_dict[ridx])

            self.rs[ridx] = eray_fan.rs
            self.ts[ridx] = eray_fan.ts
            self.zs[ridx] = eray_fan.zs
            self.ps[ridx] = eray_fan.ps

            received_angles_single = []
            ray_ids = []
            # compute receive angle
            for eray_idx in range(eray_fan.rs.shape[0]):
                y_last = np.stack((eray_fan.ts[eray_idx, -1], eray_fan.zs[eray_idx, -1], eray_fan.ps[eray_idx, -1]))
                theta, c = pr.ray_angle(
                    eray_fan.rs[eray_idx, -1],
                    y_last, environment.sound_speed.values,
                    environment.sound_speed.range.values,
                    environment.sound_speed.depth.values
                )
                received_angles_single.append(-theta)
                ray_id_single = np.sum(np.diff(np.sign(eray_fan.ps[eray_idx,:])) != 0) * (-np.sign(eray_fan.thetas[eray_idx]))
                ray_ids.append(ray_id_single)
            self.received_angles[ridx] = np.array(received_angles_single)
            self.launch_angles[ridx] = eray_fan.thetas
            self.ray_id[ridx] = np.array(ray_ids)

    def plot_angle_time(self,ridxs = None, **kwargs):

        if ridxs is None:
            ridxs = list(self.received_angles.keys())
        
        for ridx in ridxs:
            plt.scatter(self.ts[ridx][:,-1], self.received_angles[ridx], **kwargs)
        
        plt.xlabel('time [s]')
        plt.ylabel('received angle [deg]')
        plt.title('Received Angle vs Time')

    def plot(self, ridxs = [0], **kwargs):
        '''
        Plot all eigen rays in time-depth space

        Parameters
        ----------
        ridxs : list
            list of receiver depth indices to plot. Default is [0], which plots the first receiver depth.
        '''

        # if ridx is int, make list of length 1
        if isinstance(ridxs, int):
            ridxs = [ridxs]

        ray_kwargs = {'c':'k'}
        ray_kwargs.update(kwargs)

        for ridx in ridxs:
            plt.plot(self.rs[ridx].T, self.zs[ridx].T, **ray_kwargs)

        plt.xlabel('range [m]')
        plt.ylabel('depth [m]')
        plt.title('Eigen Rays')
        plt.ylim([self.zs[ridx].max(), self.zs[ridx].min()])

    def save_mat(self, filename):
        """
        Save EigenRays object to a .mat file.

        Parameters
        ----------
        filename : str
            Name of the output .mat file to save the EigenRays data.
        """

        data = {}
        for ridx,rdepth in enumerate(self.receiver_depths):
            data[f'receiver_depth_{ridx}'] = {
                'receiver_depth' : rdepth,
                'xs' : self.rs[ridx],
                'ts' : self.ts[ridx],
                'zs' : self.zs[ridx],
                'ps' : self.ps[ridx],
                'received_angles' : self.received_angles[ridx],
                'launch_angles' : self.launch_angles[ridx],
                'ray_id' : self.ray_id[ridx],
                'n_bottom' : self.n_botts[ridx] if hasattr(self, 'n_botts') else np.nan,
                'n_surface' : self.n_surfs[ridx] if hasattr(self, 'n_surfs') else np.nan,
                'source_depth' : self.source_depths[ridx] if hasattr(self, 'source_depths') else np.nan,
                'num_eigenrays' : self.num_eigenrays,
                'num_eigenrays_found' : self.num_eigenrays_found,
            }

        # Save the dictionary to a .mat file
        io.savemat(filename, {'eigenrays':data})

__all__ = ['Ray','RayFan','EigenRays']