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
    
    def plot_ray_fan(self,):
        '''
        plot ray fan
        '''

        _ = plt.plot(self.rs.T, self.zs.T, c='k', lw=1, alpha=10*1/len(self.thetas))
        plt.xlabel('range [m]')
        plt.ylabel('depth [m]')
        plt.ylim([self.zs.max(), self.zs.min()])
        plt.title('Ray Fan')

        return
    
    def plot_depth_v_angle(self,):
        plt.scatter(x=self.thetas, y=self.zs[:,-1], s=2, c='k', lw=0)

        return
    

__all__ = ['Ray','RayFan']