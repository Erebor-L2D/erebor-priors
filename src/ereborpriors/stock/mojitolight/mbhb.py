from __future__ import annotations
import typing

import numpy as np
from eryn.priors import UniformDistribution, CosineUniform, SineUniform, LogUniform
from eryn.utils.transform import TransformContainer

from ereborpriors.base import GFBasePrior
from ereborpriors.priors import UniformInComovingVolume
from ereborpriors.utility import EPS, over_thousands, times_thousands, total_mass_mass_ratio_to_component_masses, component_masses_to_total_mass_mass_ratio

class MBHBPrior(GFBasePrior):
    """
    A class representing the MBHB prior, inheriting from GFBasePrior.
    """

    branch_name = "mbh"

    @classmethod
    def prior_factory(cls, observation_time: float) -> MBHBPrior:
        """
        Factory method to create an instance of MBHBPrior.

        Args:
            observation_time (float): The observation time for the MBHB prior.

        Returns:
            MBHBPrior: An instance of MBHBPrior.
        """
        priors_in = cls.build_priors(cls, observation_time)
        return cls(branch_name=cls.branch_name, priors_in=priors_in)

    def build_priors(self, observation_time: float) -> dict[str, typing.Any]:
        """
        Builds the priors for the MBHB prior.

        Args:
            observation_time (float): The observation time for the MBHB prior.
        
        Returns:
            dict[str, typing.Any]: A dictionary containing the priors for the MBHB prior.
        """
        priors_in = {
            "logM": UniformDistribution(np.log(1e5).item(), np.log(1e8).item(), name="log total mass", latex_label=r"$\log M$", unit=None),
            "logq": UniformDistribution(np.log(1).item(), np.log(10).item(), name="log mass ratio", latex_label=r"$\log q$", unit=None),
            "s1z": UniformDistribution(-1 + EPS, 1 - EPS, name="s1z", latex_label=r"$s_{1z}$", unit=None),
            "s2z": UniformDistribution(-1 + EPS, 1 - EPS, name="s2z", latex_label=r"$s_{2z}$", unit=None),
            "d_L": UniformInComovingVolume(1, 150, name="luminosity distance", latex_label=r"$d_L$", unit=r"$\rm Gpc$"),
            "phi_ref": UniformDistribution(0, 2 * np.pi, name="reference phase", latex_label=r"$\phi_{\rm ref}$", unit=r"$\rm rad$", boundary="periodic"),
            "cos_iota": UniformDistribution(-1 + EPS, 1 - EPS, name="cos inclination", latex_label=r"$\cos \iota$", unit=None),
            "psi": UniformDistribution(0, 2 * np.pi, name="polarization angle", latex_label=r"$\psi$", unit=r"$\rm rad$", boundary="periodic"),
            "alpha": UniformDistribution(0, 2 * np.pi, name="right ascension", latex_label=r"$\alpha$", unit=r"$\rm rad$", boundary="periodic"),
            "sin_delta": UniformDistribution(-1 + EPS, 1 - EPS, name="sin declination", latex_label=r"$\sin \delta$", unit=None),
            "t_c": UniformDistribution(0, observation_time, name="coalescence time", latex_label=r"$t_c$", unit=r"$\rm s$"),
        }

        return priors_in

    def get_output_basis(self) -> list[str]:
        """
        Returns the output basis for the MBHB prior.

        Returns:
            list[str]: A list of parameter names in the output basis.
        """
        return ["m1", "m2", "s1z", "s2z", "d_L", "phi_ref", "iota", "psi", "alpha", "delta", "t_c"]

    def get_transforms(self) -> TransformContainer:
        """
        Returns the transforms associated with the MBHB prior.

        Returns:
            TransformContainer: The transforms associated with the MBHB prior.
        """

        output_basis = self.get_output_basis()

        key_map = {
            "logM": "m1",
            "logq": "m2",
            "cos_iota": "iota",
            "sin_delta": "delta",
        }

        forward_transforms = {
            "m1": np.exp,
            "m2": np.exp,
            "d_L": times_thousands,
            "iota": np.arccos,
            "delta": np.arcsin,
            ("m1", "m2"): total_mass_mass_ratio_to_component_masses,
        }

        inverse_transforms = {
            "m1": np.log,
            "m2": np.log,
            "d_L": over_thousands,
            "iota": np.cos,
            "delta": np.sin,
            ("m1", "m2"): component_masses_to_total_mass_mass_ratio,
        }

        transform_container = TransformContainer(
            input_basis=self.get_input_basis(),
            output_basis=output_basis,
            key_map=key_map,
            parameter_transforms=forward_transforms,
            inverse_parameter_transforms=inverse_transforms,
        )

        return transform_container

    def get_periodic(self) -> dict[str, float] | None:
        """
        Returns the periodic parameters associated with the MBHB prior.

        Returns:
            dict[str, float] | None: A dictionary of periodic parameters and their periods, or None if there are no periodic parameters.
        """
        return {
            "phi_ref": 2 * np.pi,
            "psi": 2 * np.pi,
            "alpha": 2 * np.pi,
        }

if __name__ == "__main__":
    observation_time = 1e7  # Example observation time
    mbhb_prior = MBHBPrior.prior_factory(observation_time)
    print("Parameter Names:", mbhb_prior.parameter_names)
    print("Parameter LaTeX Names:", mbhb_prior.parameter_latex_labels)
    mbhb_prior.to_yaml("mbhb_prior.yaml")  # Save the prior to a YAML file
    transforms = mbhb_prior.get_transforms()
    print("Transforms:", transforms)

    breakpoint()  # For debugging purposes
