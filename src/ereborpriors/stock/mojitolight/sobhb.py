from __future__ import annotations
import typing

import numpy as np
from eryn.priors import UniformDistribution
from eryn.utils.transform import TransformContainer

from ereborpriors.base import GFBasePrior
from ereborpriors.priors import ChirpMassEtaUniformComponentMasses, CosineDeclination, UniformInComovingVolume
from ereborpriors.utility import over_thousands, times_thousands

# Parameter order of the gwtf (time-frequency) SOBHB branch. Same names and order as
# lisatools.globalfit.stock.erebor.SOBBHTF_BASIS / the GWTF_generator parameter order, used for both
# the sampling (input) and GWTF_generator (output) basis. 

SOBHB_BASIS = [
    "Mc",        # chirp mass (Msun, SSB frame)
    "eta",       # symmetric mass ratio
    "cosinc",    # cos(inclination)
    "dist",      # luminosity distance (sampled in Gpc, Mpc in the GWTF_generator basis)
    "f0",        # GW frequency (Hz) at the start of the data (gwtf t=0)
    "s1",        # primary aligned spin
    "s2",        # secondary aligned spin
    "phi_coal",  # coalescence phase
    "psi",       # polarisation angle (ICRS)
    "ra",        # right ascension (ICRS)
    "dec",       # declination (ICRS)
]

# Default ranges: lisatools.globalfit.stock.erebor.SOBBHTF_DEFAULT_LIMS, with dist converted Mpc -> Gpc.
# Mc and f0 are per-source in practice; override them via prior_factory(mc_bounds=..., f0_bounds=...)
# or lims={"Mc": ..., "f0": ...}.
SOBHB_DEFAULT_LIMS = {
    "Mc": (5.0, 100.0),   # Msun
    "eta": (0.1, 0.2499999),
    "cosinc": (-1.0, 1.0),
    "dist": (0.01, 1.0),  # Gpc
    "f0": (1.0e-3, 0.08),  # Hz
    "s1": (-0.99, 0.99),
    "s2": (-0.99, 0.99),
    "phi_coal": (0.0, 2 * np.pi),
    "psi": (0.0, np.pi),
    "ra": (0.0, 2 * np.pi),
    "dec": (-np.pi / 2, np.pi / 2),
}

# Component-mass box [m_min, m_max] (Msun) cutting the joint (Mc, eta) prior (SOBBHTFSettings.component_mass_lims).
SOBHB_DEFAULT_COMPONENT_MASS_LIMS = (5.0, 130.0)


class SOBHBPrior(GFBasePrior):
    """
    Prior for the gwtf (time-frequency) SOBHB branch, inheriting from GFBasePrior.

    Port of ``lisatools.globalfit.stock.erebor.SOBBHTFSetup.setup_priors``. Shapes:
    (Mc, eta) jointly uniform in component masses (cut by the component-mass box),
    dist uniform in comoving volume (Planck15, sampled in Gpc), dec uniform on the sphere,
    everything else uniform. Sampled in the GWTF_generator basis except dist, which the
    transform converts Gpc -> Mpc.
    """

    branch_name = "sobhb"

    @classmethod
    def prior_factory(cls,
                      f0_bounds: tuple[float, float] | None = None,
                      mc_bounds: tuple[float, float] | None = None,
                      component_mass_bounds: tuple[float, float] = SOBHB_DEFAULT_COMPONENT_MASS_LIMS,
                      lims: dict[str, tuple[float, float]] | None = None,
                      ) -> SOBHBPrior:
        """
        Factory method to create an instance of SOBHBPrior.

        Args:
            f0_bounds (tuple[float, float] | None): Bounds on f0 (Hz, at data start) for a single source.
                Default SOBHB_DEFAULT_LIMS["f0"].
            mc_bounds (tuple[float, float] | None): Bounds on the chirp mass (Msun) for a single source.
                Default SOBHB_DEFAULT_LIMS["Mc"].
            component_mass_bounds (tuple[float, float]): Component-mass box [m_min, m_max] (Msun). Default (5, 130).
            lims (dict | None): Overrides of SOBHB_DEFAULT_LIMS, keyed by parameter name.

        Returns:
            SOBHBPrior: An instance of SOBHBPrior.
        """
        priors_in = cls.build_priors(f0_bounds, mc_bounds, component_mass_bounds, lims)
        return cls(branch_name=cls.branch_name, priors_in=priors_in)

    @staticmethod
    def build_priors(f0_bounds: tuple[float, float] | None = None,
                     mc_bounds: tuple[float, float] | None = None,
                     component_mass_bounds: tuple[float, float] = SOBHB_DEFAULT_COMPONENT_MASS_LIMS,
                     lims: dict[str, tuple[float, float]] | None = None,
                     ) -> dict[str, typing.Any]:
        """
        Builds the per-source priors for the SOBHB sources.

        Args:
            f0_bounds (tuple[float, float] | None): Bounds on f0 (Hz, at data start) for a single source.
                Default SOBHB_DEFAULT_LIMS["f0"].
            mc_bounds (tuple[float, float] | None): Bounds on the chirp mass (Msun) for a single source.
                Default SOBHB_DEFAULT_LIMS["Mc"].
            component_mass_bounds (tuple[float, float]): Component-mass box [m_min, m_max] (Msun).
            lims (dict | None): Overrides of SOBHB_DEFAULT_LIMS, keyed by parameter name.

        Returns:
            dict[str, typing.Any]: The priors, in SOBHB_BASIS column order.
        """
        lims = dict(lims or {})
        unknown = set(lims) - set(SOBHB_DEFAULT_LIMS)
        if unknown:
            raise KeyError(f"Unknown parameters in lims: {sorted(unknown)}.")
        for key, bounds in (("Mc", mc_bounds), ("f0", f0_bounds)):
            if bounds is not None:
                if key in lims:
                    raise ValueError(f"{key} bounds given twice (as an argument and in lims).")
                lims[key] = bounds
        lims = {**SOBHB_DEFAULT_LIMS, **lims}

        # ProbDistContainer assigns columns by insertion order, so this dict must follow SOBHB_BASIS.
        priors_in = {
            ("Mc", "eta"): ChirpMassEtaUniformComponentMasses(*lims["Mc"], *lims["eta"], *component_mass_bounds),
            "cosinc": UniformDistribution(*lims["cosinc"], name="cosinc", latex_label=r"$\cos \iota$", unit=None),
            "dist": UniformInComovingVolume(*lims["dist"], name="dist", latex_label=r"$d_L$", unit=r"$\rm Gpc$"),
            "f0": UniformDistribution(*lims["f0"], name="f0", latex_label=r"$f_0$", unit=r"$\rm Hz$"),
            "s1": UniformDistribution(*lims["s1"], name="s1", latex_label=r"$\chi_1$", unit=None),
            "s2": UniformDistribution(*lims["s2"], name="s2", latex_label=r"$\chi_2$", unit=None),
            "phi_coal": UniformDistribution(*lims["phi_coal"], name="phi_coal", latex_label=r"$\phi_c$", unit=r"$\rm rad$", boundary="periodic"),
            "psi": UniformDistribution(*lims["psi"], name="psi", latex_label=r"$\psi$", unit=r"$\rm rad$", boundary="periodic"),
            "ra": UniformDistribution(*lims["ra"], name="ra", latex_label=r"$\alpha$", unit=r"$\rm rad$", boundary="periodic"),
            "dec": CosineDeclination(*lims["dec"], name="dec", latex_label=r"$\delta$", unit=r"$\rm rad$"),
        }

        return priors_in

    def get_output_basis(self) -> list[str]:
        """
        Returns the output basis for the SOBHB prior (the GWTF_generator basis).

        Returns:
            list[str]: A list of parameter names in the output basis.
        """
        return list(SOBHB_BASIS)

    def get_transforms(self) -> TransformContainer:
        """
        Returns the transforms associated with the SOBHB prior. The sampling basis is the
        GWTF_generator basis apart from dist, which is sampled in Gpc and converted to the Mpc
        GWTF_generator expects (as the MBHB prior does for d_L).

        Returns:
            TransformContainer: The transforms associated with the SOBHB prior.
        """
        assert self.get_input_basis() == SOBHB_BASIS, self.get_input_basis()
        return TransformContainer(
            input_basis=self.get_input_basis(),
            output_basis=self.get_output_basis(),
            parameter_transforms={"dist": times_thousands},          # Gpc -> Mpc
            inverse_parameter_transforms={"dist": over_thousands},   # Mpc -> Gpc
        )

    def get_periodic(self) -> dict[str, float] | None:
        """
        Returns the periodic parameters associated with the SOBHB prior (as in
        lisatools SOBBHTFSetup). Hardcoded rather than read from each prior's ``boundary``
        attribute, because ``boundary`` is not written by to_yaml, so a prior loaded with
        from_yaml would otherwise lose its periodicity.

        Returns:
            dict[str, float] | None: A dictionary of periodic parameters and their periods.
        """
        return {
            "phi_coal": 2 * np.pi,
            "psi": np.pi,
            "ra": 2 * np.pi,
        }


if __name__ == "__main__":
    sobhb_prior = SOBHBPrior.prior_factory()  # all defaults (SOBHB_DEFAULT_LIMS)
    print("Parameter Names:", sobhb_prior.parameter_names)
    print("Parameter LaTeX Names:", sobhb_prior.parameter_latex_labels)
    print("Periodic:", sobhb_prior.get_periodic())
    sobhb_prior.to_yaml("sobhb_prior.yaml")  # Save the prior to a YAML file
    print("Transforms:", sobhb_prior.get_transforms())
