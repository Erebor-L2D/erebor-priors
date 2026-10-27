from __future__ import annotations
import typing

import numpy as np
from eryn.priors import UniformDistribution, CosineUniform, SineUniform, LogUniform, UniformInVolume
from eryn.utils.transform import TransformContainer
from eryn.utils.periodic import PeriodicContainer

from ereborpriors.base import GFBasePrior
from ereborpriors.priors import F0FdotPrior
from ereborpriors.utility import EPS, over_thousands, times_thousands

class GBPrior(GFBasePrior):
    """
    A class representing the GB prior, inheriting from GFBasePrior.
    """

    def __init__(self, priors_in: dict[str, typing.Any] = None, branch_name: str = "gb", include_fdot: bool = True, **kwargs):
        """
        Initializes the GBPrior instance.

        Args:
            priors_in (dict[str, typing.Any]): A dictionary containing the priors for the GB prior.
            branch_name (str): The name of the branch associated with this prior. Default is "gb".
            include_fdot (bool): Whether to include the fdot parameter in the physical space.
            **kwargs: Additional keyword arguments to be passed to the GFBasePrior constructor.
        """

        super().__init__(branch_name=branch_name, priors_in=priors_in, **kwargs)

        self.include_fdot = include_fdot

    @classmethod
    def prior_factory(cls, branch_name: str = "gb", include_fdot: bool = True) -> GBPrior:
        """
        Factory method to create an instance of GBPrior.

        Args:
            branch_name (str): The name of the branch associated with this prior. Default is "gb". Included for future flexibility for vgbs.
            include_fdot (bool): Whether to include the fdot parameter in the physical space.

        Returns:
            GBPrior: An instance of GBPrior.
        """
        priors_in = cls.build_priors(cls)
        out = cls(branch_name=branch_name, priors_in=priors_in, include_fdot=include_fdot)
        return out

    def build_priors(self) -> dict[str, typing.Any]:
        """
        Builds the priors for the GB prior.

        Returns:
            dict[str, typing.Any]: A dictionary containing the priors for the GB prior.
        """
        priors_in = {
            "logA": UniformDistribution(np.log(10**(-23.2)), np.log(10**(-20)), name="log amplitude", latex_label=r"$\log A$", unit=None),
            ("f0", "fdot"): F0FdotPrior(f0_min=0.1, f0_max=1e3, f0_units="mHz"),
            "phi0": UniformDistribution(0, 2 * np.pi, name="starting phase", latex_label=r"$\phi_{0}$", unit=r"$\rm rad$", boundary="periodic"),
            "cos_iota": UniformDistribution(-1 + EPS, 1 - EPS, name="cos inclination", latex_label=r"$\cos \iota$", unit=None),
            "psi": UniformDistribution(0, 2 * np.pi, name="polarization angle", latex_label=r"$\psi$", unit=r"$\rm rad$", boundary="periodic"),
            "alpha": UniformDistribution(0, 2 * np.pi, name="right ascension", latex_label=r"$\alpha$", unit=r"$\rm rad$", boundary="periodic"),
            "sin_delta": UniformDistribution(-1 + EPS, 1 - EPS, name="sin declination", latex_label=r"$\sin \delta$", unit=None),
        }

        return priors_in

    def get_output_basis(self) -> list[str]:
        """
        Returns the output basis for the GB prior.

        Returns:
            list[str]: A list of parameter names in the output basis.
        """
        if self.include_fdot:
            return ["A", "f0", "fdot", "fddot", "phi0", "iota", "psi", "alpha", "delta"]
        
        return ["A", "f0", "fdot", "phi0", "iota", "psi", "alpha", "delta"]

    def get_transforms(self) -> TransformContainer:
        """
        Returns the transforms associated with the GB prior.

        Returns:
            TransformContainer: The transforms associated with the GB prior.
        """

        output_basis = self.get_output_basis()

        key_map = {
            "logA": "A",
            "cos_iota": "iota",
            "sin_delta": "delta",
        }

        forward_transforms = {
            "A": np.exp,
            "f0": over_thousands,
            "iota": np.arccos,
            "delta": np.arcsin,
        }

        inverse_transforms = {
            "A": np.log,
            "f0": times_thousands,
            "iota": np.cos,
            "delta": np.sin,
        }

        if self.include_fdot:
            fill_dict = {"fddot": 0.0}
        else:
            fill_dict = None

        transform_container = TransformContainer(
            input_basis=self.get_input_basis(),
            output_basis=output_basis,
            key_map=key_map,
            parameter_transforms=forward_transforms,
            inverse_parameter_transforms=inverse_transforms,
            fill_dict=fill_dict
        )

        return transform_container

    def get_periodic(self) -> dict[str, float] | None:
        """
        Returns the periodic parameters associated with the GB prior.

        Returns:
            dict[str, float] | None: A dictionary of periodic parameters and their periods, or None if there are no periodic parameters.
        """

        return {
            "phi0": 2 * np.pi,
            "psi": 2 * np.pi,
            "alpha": 2 * np.pi,
        }

if __name__ == "__main__":
    gb_prior = GBPrior()
    print("Parameter Names:", gb_prior.parameter_names)
    print("Parameter LaTeX Names:", gb_prior.parameter_latex_labels)
    gb_prior.to_yaml("gb_prior.yaml")  # Save the prior to a YAML file
    transforms = gb_prior.get_transforms()
    print("Transforms:", transforms)

    breakpoint()  # For debugging purposes