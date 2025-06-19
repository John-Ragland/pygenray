import numpy as np
from matplotlib import pyplot as plt

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

    def plot_time_front(self,include_lines=False, **kwargs):
        '''
        plot time front. Key word arguments are passed to plt.scatter.

        Parameters
        ----------
        include_lines : bool
            if True, lines between recieved rays on time plots are plotted
        '''

        if include_lines:
            plt.plot(self.ts[:,-1], self.zs[:,-1], c='#aaaaaa', lw=0.5, zorder=5)
        
        
        scatter_kwargs = {'c': self.thetas, 'cmap': 'managua', 's': 2, 'lw': 0, 'zorder': 6}
        scatter_kwargs.update(kwargs)
        plt.scatter(x=self.ts[:,-1], y=self.zs[:,-1], **scatter_kwargs)


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

__all__ = ['Ray','RayFan']