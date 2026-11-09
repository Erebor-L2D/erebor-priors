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

    def __init__(self, minimum: float, maximum: float, cosmology: typing.Optional[str] = "Planck15", num_points: typing.Optional[int] = 1000, **kwargs):
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

def _component_mass_coefficients(eta: ArrayLike, xp=np) -> tuple[ArrayLike, ArrayLike]:
    """
    Coefficients c1, c2 such that m1 = c1 * Mc and m2 = c2 * Mc (m1 >= m2).

    Args:
        eta (ArrayLike): Symmetric mass ratio, in (0, 0.25].
        xp: numpy or cupy.

    Returns:
        tuple[ArrayLike, ArrayLike]: (c1, c2).
    """
    total_over_mc = eta ** (-3 / 5)
    
    # discriminant is (1 - 4 eta)^(1/2), but we clip to avoid NaNs for eta > 0.25 due to floating-point rounding.
    disc = xp.sqrt(xp.clip(1 - 4 * eta, 0.0, None))

    # m1 = 0.5 * Mc * eta^(-3/5) * (1 + sqrt(1 - 4 eta)), m2 = 0.5 * Mc * eta^(-3/5) * (1 - sqrt(1 - 4 eta))
    return 0.5 * total_over_mc * (1 + disc), 0.5 * total_over_mc * (1 - disc)


class ChirpMassEtaUniformComponentMasses(JointPrior):
    r"""
    Joint prior on (chirp mass, symmetric mass ratio) that is uniform in component masses.

    Port of ``lisatools.utils.utility.ErynEtaMcPrior`` used by the gwtf SOBBH branch
    (``SOBBHTFSetup``). The support is the box ``Mc in [mc_min, mc_max]``,
    ``eta in [eta_min, eta_max]``, cut by ``m_min < m2 <= m1 < m_max``. On it,

    .. math::
        p(\mathcal{M}_c, \eta) = \frac{1}{A}\, \mathcal{M}_c\, \eta^{-6/5} (1 - 4\eta)^{-1/2},

    the Jacobian of (m1, m2) -> (Mc, eta), with ``A`` the area of the support in the
    (m1 >= m2) component-mass plane.

    Samples are drawn from the (seperable) factorised density (Mc ∝ Mc, eta ∝ eta^-6/5 (1-4 eta)^-1/2)
    and rejected outside the component-mass cut, which is exact because the density factorises.
    """

    def __init__(
        self,
        mc_min: float,
        mc_max: float,
        eta_min: float,
        eta_max: float,
        m_min: float,
        m_max: float,
        names: typing.Sequence[str] = ("Mc", "eta"),
        num_grid: int = 100000,
        **kwargs,
    ):
        """
        Initializes the ChirpMassEtaUniformComponentMasses instance.

        Args:
            mc_min (float): Minimum chirp mass (Msun).
            mc_max (float): Maximum chirp mass (Msun).
            eta_min (float): Minimum symmetric mass ratio (> 0).
            eta_max (float): Maximum symmetric mass ratio (< 0.25, the density diverges at 0.25).
            m_min (float): Minimum component mass (Msun).
            m_max (float): Maximum component mass (Msun).
            names (Sequence[str]): Parameter names of the two columns. Default ("Mc", "eta").
            num_grid (int): Number of evenly spaced eta nodes for the tabulated eta inverse CDF.
                Default 100000. The trapezoid CDF resolves the eta -> 0.25 spike only when the
                spacing (eta_max - eta_min) / num_grid is no coarser than 0.25 - eta_max; at the
                default this holds for eta_max = 0.249999 but not 0.2499999 (rvs then slightly
                over-samples eta near eta_max; logpdf is unaffected).
            **kwargs: Additional keyword arguments for the JointPrior.
        """
        if not 0 < mc_min < mc_max:
            raise ValueError(f"Need 0 < mc_min < mc_max, got ({mc_min}, {mc_max}).")
        if not 0 < eta_min < eta_max < 0.25:
            raise ValueError(f"Need 0 < eta_min < eta_max < 0.25, got ({eta_min}, {eta_max}).")
        if not 0 < m_min < m_max:
            raise ValueError(f"Need 0 < m_min < m_max, got ({m_min}, {m_max}).")

        self.mc_min = mc_min
        self.mc_max = mc_max
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.m_min = m_min
        self.m_max = m_max
        self.num_grid = num_grid

        kwargs.setdefault("latex_labels", (r"$\mathcal{M}_c$", r"$\eta$"))
        kwargs.setdefault("units", (r"$M_\odot$", None))
        super().__init__(names=tuple(names), **kwargs)

        self._setup_eta_sampler()

        # Compute the area of the support in the (m1 >= m2) plane, for the normalisation constant.
        area = self._support_area()

        if not area > 0:
            raise ValueError(
                "Empty prior support: no (Mc, eta) in the box satisfies the component-mass cut "
                f"[{m_min}, {m_max}]."
            )
        self.log_norm = -float(np.log(area))

    def _mc_range_at_eta(self, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Allowed Mc interval at fixed eta (Mc box intersected with the component-mass cut)."""
        c1, c2 = _component_mass_coefficients(eta)
        lo = np.maximum(self.mc_min, self.m_min / c2)
        hi = np.minimum(self.mc_max, self.m_max / c1)
        return lo, np.maximum(hi, lo)

    def _support_area(self) -> float:
        """
        Area of the support in the (m1 >= m2) plane, i.e. the integral of the joint unnormalised density over (Mc,eta).

        Change of variables to s = sqrt(1 - 4 eta) removes the eta -> 0.25 singularity in the integrand.

        Integrated in s = sqrt(1 - 4 eta), where deta / sqrt(1 - 4 eta) = ds / 2 removes the
        eta -> 0.25 singularity in the density. 
        """
        from scipy.integrate import quad

        def integrand(s):
            eta = (1 - s**2) / 4
            # Evaluate the mcs allowed at this specific eta (lo, hi) and return the area of the Mc interval times the Jacobian factor.
            lo, hi = self._mc_range_at_eta(np.asarray(eta))

            # Return the (numerical) integrand of the area integral, which is the product of the Mc range and the Jacobian factor.
            return 0.5 * eta ** (-6 / 5) * 0.5 * (hi**2 - lo**2)

        # Minimum and maximum s = sqrt(1 - 4 eta) for the eta box. The integrand is zero outside this range.

        # s decreases as eta increases, so eta_min -> s_hi and eta_max -> s_lo: the integral runs s_hi -> s_lo.
        # Integrating s_lo -> s_hi instead flips the sign, absorbing the minus from deta = -(s/2) ds.
        
        s_lo, s_hi = np.sqrt(1 - 4 * self.eta_max), np.sqrt(1 - 4 * self.eta_min)
        area, _ = quad(integrand, s_lo, s_hi, epsabs=0.0, epsrel=1e-10, limit=500)
        return float(area)

    def _setup_eta_sampler(self):

        """Tabulate the CDF of eta ∝ eta^-6/5 (1-4 eta)^-1/2 on an evenly spaced eta grid, for
        inverse-CDF sampling by linear interpolation (monotone, so draws cannot overshoot
        [eta_min, eta_max])."""

        eta_grid = np.linspace(self.eta_min, self.eta_max, self.num_grid)

        density = eta_grid ** (-6 / 5) * (1 - 4 * eta_grid) ** (-1 / 2)

        # Trapezoid CDF, anchored at 0 at eta_min.
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(eta_grid))))

        self._eta_grid = self.xp.asarray(eta_grid)

        self._eta_cdf = self.xp.asarray(cdf / cdf[-1])

    def _in_support(self, mc: ArrayLike, eta: ArrayLike) -> ArrayLike:

        # Check if (mc, eta) is in the support of the prior, i.e. in the box and satisfying the component-mass cut.
        xp = self.xp

        # Check if (mc, eta) is in the box.
        in_box = (mc >= self.mc_min) & (mc <= self.mc_max) & (eta >= self.eta_min) & (eta <= self.eta_max)
        
        # Check if (mc, eta) satisfies the component-mass cut: m_min < m2 <= m1 < m_max.
        c1, c2 = _component_mass_coefficients(xp.clip(eta, self.eta_min, self.eta_max), xp=xp)

        return in_box & (c2 * mc > self.m_min) & (c1 * mc < self.m_max)

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> ArrayLike:
        xp = self.xp

        if isinstance(size, int):
            size = (size,)

        # Number of samples to draw in total.
        n = int(np.prod(size))

        mc_out = xp.empty(n)
        eta_out = xp.empty(n)
        filled = 0

        # Rejection sampling: draw from the factorised density and reject outside the component-mass cut.
        while filled < n:
            # How many samples to draw. 
            # Minimum 1024 as its possible alot of the draws are rejected. 
            n_draw = max(2 * (n - filled), 1024)

            # Draw from factorised mc 
            u = xp.random.uniform(0, 1, size=n_draw)
            mc = xp.sqrt(self.mc_min**2 + u * (self.mc_max**2 - self.mc_min**2))

            # Draw from factorised eta using the tabulated inverse CDF.
            eta = xp.interp(xp.random.uniform(0, 1, size=n_draw), self._eta_cdf, self._eta_grid)
            eta = xp.clip(eta, self.eta_min, self.eta_max)  # guards floating-point rounding only

            # Check if the drawn (mc, eta) are in the support of the prior.
            keep = self._in_support(mc, eta)

            # Fill the output arrays with the accepted samples, up to the requested number of samples.
            n_keep = min(int(keep.sum()), n - filled)
            mc_out[filled:filled + n_keep] = mc[keep][:n_keep]
            eta_out[filled:filled + n_keep] = eta[keep][:n_keep]
            filled += n_keep

        samples = xp.stack([mc_out, eta_out], axis=-1).reshape(size + (2,))
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> ArrayLike:
        xp = self.xp
        x_arr = xp.asarray(x)
        if x_arr.shape[-1] != 2:
            raise ValueError("Expected last dimension of x to be 2.")

        mc = x_arr[..., 0]
        eta = x_arr[..., 1]
        out = xp.full_like(mc, -xp.inf, dtype=xp.float64)

        valid = self._in_support(mc, eta)
        mc_v, eta_v = mc[valid], eta[valid]
        
        # log p = −log A + log Mc − (6/5)·log η −1/2*log(1 − 4η)
        out[valid] = self.log_norm + xp.log(mc_v) - 1.2 * xp.log(eta_v) - 0.5 * xp.log(1 - 4 * eta_v)

        return self._to_device(out)


class CosineDeclination(Prior):
    r"""
    Prior p(x) ∝ cos(x) on [minimum, maximum] within [-pi/2, pi/2]: isotropic sky when sampling
    the declination directly (rather than sin(dec), as eryn's SineUniform does).

    .. math::
        p(\delta) = \frac{\cos\delta}{\sin\delta_{\max} - \sin\delta_{\min}}
    """

    def __init__(self, minimum: float = -np.pi / 2, maximum: float = np.pi / 2, **kwargs):
        """
        Initializes the CosineDeclination instance.

        Args:
            minimum (float): Lower bound (rad), >= -pi/2. Default -pi/2.
            maximum (float): Upper bound (rad), <= pi/2. Default pi/2.
            **kwargs: Additional keyword arguments for the Prior.
        """
        if minimum < -np.pi / 2 or maximum > np.pi / 2:
            raise ValueError(f"CosineDeclination bounds must lie within [-pi/2, pi/2], got ({minimum}, {maximum}).")

        super().__init__(minimum=minimum, maximum=maximum, **kwargs)

        self._sin_min = float(np.sin(minimum))

        self._sin_range = float(np.sin(maximum) - np.sin(minimum))

        self._log_norm = -float(np.log(self._sin_range))

    def logpdf(self, x: ArrayLike, **kwargs) -> ArrayLike:
        x_arr = self.xp.asarray(x)
        # Return -inf outside the bounds, log(cos(x)) inside, with normalisation.
        out = self.xp.full_like(x_arr, -self.xp.inf, dtype=self.xp.float64)
        
        mask = (x_arr >= self.minimum) & (x_arr <= self.maximum)
        
        out[mask] = self._log_norm + self.xp.log(self.xp.cos(x_arr[mask]))
        
        return self._to_device(out)

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> ArrayLike:
        if isinstance(size, int):
        
            size = (size,)
        
        u = self.xp.random.uniform(0, 1, size=size)
        
        return self._to_device(self.xp.arcsin(self._sin_min + u * self._sin_range))

