"""
Specific priors for the stock classes.
"""
from __future__ import annotations
import typing
from numpy.typing import ArrayLike
import numpy as np
from scipy.interpolate import interp1d

from eryn.priors import Prior, JointPrior

class F0FdotPrior(JointPrior):
    """
    A class representing the joint prior for f0 and fdot, inheriting from JointPrior.
    Reference implementation: https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/c2dff19627c2fe9bb258a0d2f32882cf87dc47f2/src/lisatools/globalfit/priors/joint.py#L556-L629
    """

    def __init__(self, f0_min: float, f0_max: float, f0_units: typing.Optional[str] = "mHz", **kwargs):
        """
        Initializes the F0FdotPrior instance.

        Args:
            f0_min (float): The minimum value for f0.
            f0_max (float): The maximum value for f0.
            f0_units (str): The units for f0.
            **kwargs: Additional keyword arguments for the JointPrior.
        """

        if f0_units not in ["mHz", "Hz"]:
            raise ValueError("f0_units must be either 'mHz' or 'Hz'.")

        self.f0_units = f0_units
        self.is_mHz = f0_units == "mHz"

        self.f0_min = f0_min
        self.f0_max = f0_max

        super().__init__(
            names=("f0", "fdot"),
            names_phys=("f0", "fdot"),
            latex_labels=(r"f_0", r"\dot{f}"),
            units=(f0_units, "Hz/s"),
            **kwargs
        )

    def get_fdot_bounds(self, f0: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
        r"""
        Calculate the bounds for fdot based on the given f0. Compute with:
        .. math::
            \dot{f}_{\rm min} = -2\times 10^{-20} \left(\frac{f_0 [Hz]}{4\times 10^{-4}}\right)^{16/3}
        .. math::
            \dot{f}_{\rm max} = 3\times 10^{-21} \left(\frac{f_0 [Hz]}{10^{-4}}\right)^{11/3}

        Args:
            f0 (float | np.ndarray): The value of f0.
        
        Returns:
            tuple[float | np.ndarray, float | np.ndarray]: A tuple containing the minimum and maximum bounds for fdot.
        """

        if self.is_mHz:
            f0_Hz = f0 * 1e-3
        else:
            f0_Hz = f0

        min_val = -2e-20 * (f0_Hz / 4e-4) ** (16 / 3)
        max_val = 3e-21 * (f0_Hz / 1e-4) ** (11 / 3)

        return min_val, max_val

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> ArrayLike:
        if isinstance(size, int):
            size = (size,)

        # 1. Sample f0 in mHz
        f0_mHz_samples = self.xp.random.uniform(self.f0_min, self.f0_max, size=size)
        
        # 2. Get bounds and sample fdot
        fdot_min, fdot_max = self.get_fdot_bounds(f0_mHz_samples)
        fdot_samples = self.xp.random.uniform(fdot_min, fdot_max, size=size)
        
        samples = self.xp.stack([f0_mHz_samples, fdot_samples], axis=-1)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> ArrayLike:
        x_arr = self.xp.asarray(x)
        
        if x_arr.shape[-1] != 2:
            raise ValueError("Expected last dimension of x to be 2.")

        f0_mHz = x_arr[..., 0]
        fdot = x_arr[..., 1]
        
        out = self.xp.full_like(f0_mHz, -self.xp.inf, dtype=self.xp.float64)
        
        mask_f0 = (f0_mHz >= self.f0_min) & (f0_mHz <= self.f0_max)
        fdot_min, fdot_max = self.get_fdot_bounds(f0_mHz)
        mask_fdot = (fdot >= fdot_min) & (fdot <= fdot_max)
        
        valid = mask_f0 & mask_fdot
        valid_fdot_diff = self.xp.where(valid, fdot_max - fdot_min, 1.0)
        
        # p(f0, fdot) = p(f0) * p(fdot|f0)
        f0_log_norm = -self.xp.log(self.f0_max - self.f0_min)
        out[valid] = f0_log_norm - self.xp.log(valid_fdot_diff[valid])
        
        return self._to_device(out)

class UniformInComovingVolume(Prior):
    """
    A class representing a uniform prior in comoving volume.
    """

    def __init__(self, minimum: float, maximum: float, cosmology: typing.Optional[str] = "Planck18", num_points: typing.Optional[int] = 1000, **kwargs):
        """
        Initializes the UniformInComovingVolume instance.

        Args:
            minimum (float): The minimum luminosity distance in Gpc.
            maximum (float): The maximum luminosity distance in Gpc.
            cosmology (str): The cosmology to use for the comoving volume calculation. Default is "Planck18".
            num_points (int): The number of points to use for the interpolation grid. Default is 1000.
            **kwargs: Additional keyword arguments for the Prior.
        """
        import astropy.cosmology as cosmo
        import astropy.constants as ac
    
        super().__init__(minimum=minimum, maximum=maximum, **kwargs)
        self.d_L_min = minimum
        self.d_L_max = maximum
        self.cosmology = cosmology  # keep the name: eryn's __repr__ and to_yaml read init args back from attributes
        self._cosmology: cosmo.Cosmology = getattr(cosmo, cosmology)
        if not self._cosmology.is_flat:
            # _comoving_volume uses V_c = 4/3 pi D_c^3, which only holds for a flat universe
            raise ValueError(f"UniformInComovingVolume requires a flat cosmology, but '{cosmology}' is not flat.")
        self.num_points = num_points

        self._setup_interpolation_grid()

        self._volume_min = self._comoving_volume(self.d_L_min)
        self._volume_max = self._comoving_volume(self.d_L_max)
        self._log_norm = self.xp.log(self._volume_max - self._volume_min)
        self._c = ac.c.to_value("km/s")  # Speed of light in km/s

    def _setup_interpolation_grid(self):
        """
        Set up the interpolation grid for the comoving volume calculation. Based on the implentation in https://gitlab.in2p3.fr/lisa-apc/geemoo-priors/-/blob/main/src/geemoo_priors/dist.py.
        """
        import astropy.units as u
        import astropy.cosmology.units as cu
        from astropy.cosmology import z_at_value

        z_min = z_at_value(self._cosmology.luminosity_distance, self.d_L_min * u.Gpc).to_value(cu.redshift)
        z_max = z_at_value(self._cosmology.luminosity_distance, self.d_L_max * u.Gpc).to_value(cu.redshift)

        _z_grid = np.linspace(z_min, z_max, self.num_points)
        _h_grid = self._cosmology.H(_z_grid).to_value(u.km / u.s / u.Gpc)
        _d_c_grid = self._cosmology.comoving_distance(_z_grid).to_value(u.Gpc)
        _d_l_grid = self._cosmology.luminosity_distance(_z_grid).to_value(u.Gpc)

        # Store the grids on device for later use
        self._z_grid = self.xp.asarray(_z_grid)
        self._h_grid = self.xp.asarray(_h_grid)
        self._d_c_grid = self.xp.asarray(_d_c_grid)
        self._d_l_grid = self.xp.asarray(_d_l_grid)

    def _comoving_distance(self, d_l: ArrayLike) -> ArrayLike:
        """
        Convert luminosity distance to comoving distance using the interpolation grid.

        Args:
            d_l (float | np.ndarray): Luminosity distance in Gpc.

        Returns:
            np.ndarray: Comoving distance in Gpc.
        """

        return self._to_device(self.xp.interp(d_l, self._d_l_grid, self._d_c_grid))

    def _hubble_parameter(self, d_l: ArrayLike) -> ArrayLike:
        """
        Get the Hubble parameter for a given luminosity distance using the interpolation grid.

        Args:
            d_l (float | np.ndarray): Luminosity distance in Gpc.

        Returns:
            np.ndarray: Hubble parameter in km/s/Gpc.
        """
        return self._to_device(self.xp.interp(d_l, self._d_l_grid, self._h_grid))

    def _redshift(self, d_l: ArrayLike) -> ArrayLike:
        """
        Get the redshift for a given luminosity distance using the interpolation grid.

        Args:
            d_l (float | np.ndarray): Luminosity distance in Gpc.
        
        Returns:
            np.ndarray: Redshift.
        """
        return self._to_device(self.xp.interp(d_l, self._d_l_grid, self._z_grid))

    def _comoving_volume(self, d_l: ArrayLike) -> ArrayLike:
        """
        Calculate the comoving volume for a given luminosity distance.

        Args:
            d_l (float | np.ndarray): Luminosity distance in Gpc.
        
        Returns:
            ArrayLike: Comoving volume in Gpc^3.
        """
        d_c = self._comoving_distance(d_l)
        return self._to_device((4 / 3) * self.xp.pi * d_c**3)

    def logpdf(self, x: ArrayLike, **kwargs) -> ArrayLike:
        """
        Calculate the log probability density function for the uniform in comoving volume prior.

        Args:
            x (float | np.ndarray): Luminosity distance in Gpc.
        
        Returns:
            ArrayLike: Log probability density values.
        """
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -self.xp.inf, dtype=self.xp.float64)

        mask = (x_arr >= self.d_L_min) & (x_arr <= self.d_L_max)
        if np.any(mask):

            d_c = self._comoving_distance(x_arr[mask])
            h_z = self._hubble_parameter(x_arr[mask])
            z = self._redshift(x_arr[mask])

            out[mask] = (
                - self._log_norm 
                + self.xp.log(4 * self.xp.pi * self._c) 
                + 2 * self.xp.log(d_c)
                - self.xp.log(d_c * h_z + (1 + z) * self._c)
            )

        return self._to_device(out)

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> ArrayLike:
        """
        Sample random values from the uniform in comoving volume prior.

        Args:
            size (int | tuple[int, ...]): The shape of the output array. Default is (1,).

        Returns:
            ArrayLike: Random samples from the prior.
        """
        if isinstance(size, int):
            size = (size,)

        u_samples = self.xp.random.uniform(0, 1, size=size)
        volume_samples = self._volume_min + u_samples * (self._volume_max - self._volume_min)

        # Invert the comoving volume to get the corresponding luminosity distance
        d_c_samples = (3 * volume_samples / (4 * self.xp.pi)) ** (1 / 3)
        d_l_samples = self.xp.interp(d_c_samples, self._d_c_grid, self._d_l_grid)

        return self._to_device(d_l_samples)