from __future__ import annotations
import typing

import numpy as np
from eryn.priors import UniformDistribution, CosineUniform, SineUniform, LogUniform
from eryn.utils.transform import TransformContainer

from ereborpriors.base import GFBasePrior
from ereborpriors.priors import UniformInComovingVolume
from ereborpriors.utility import EPS, over_thousands, times_thousands, total_mass_mass_ratio_to_component_masses, component_masses_to_total_mass_mass_ratio, decouple_spin_sign, couple_spin_sign

class EMRIPrior(GFBasePrior):
    """
    A class representing the EMRI prior, inheriting from GFBasePrior.
    """

    branch_name = "emri"

    @classmethod
    def prior_factory(cls) -> EMRIPrior:
        """
        Factory method to create an instance of EMRIPrior.
 
        Returns:
            EMRIPrior: An instance of EMRIPrior.
        """
        priors_in = cls.build_priors(cls)
        return cls(branch_name=cls.branch_name, priors_in=priors_in)

    def build_priors(self,) -> dict[str, typing.Any]:
        """
        Builds the priors for the EMRI prior.

        Args:
            self: The instance of the class.

        Returns:
            dict[str, typing.Any]: A dictionary containing the priors for the MBHB prior.
        """
        priors_in = {
            "log_m1": UniformDistribution(np.log(1e5).item(), np.log(1e7).item(), name="log primary mass", latex_label=r"$\log m_1$", unit=None),
            "m2": UniformDistribution(1.0, 200.0, name="secondary mass", latex_label=r"$m_2$", unit=r"$M_\odot$"),
            "a": UniformDistribution(-1 + EPS, 1 - EPS, name="primary spin", latex_label=r"$a$", unit=None),
            "p0": UniformDistribution(5.0, 100.0, name="initial semi-latus rectum", latex_label=r"$p_{0}$", unit=None),
            "e0": UniformDistribution(0.001, 0.8, name="initial eccentricity", latex_label=r"$e_{0}$", unit=None),
            "d_L": UniformInComovingVolume(1, 150, name="luminosity distance", latex_label=r"$d_L$", unit=r"$\rm Gpc$"),
            "cos_qK":  UniformDistribution(-1 + EPS, 1 - EPS, name="cos polar angle", latex_label=r"$\cos \theta_{\rm K}$", unit=None),
            "phiK": UniformDistribution(0, 2 * np.pi, name="azimuthal angle", latex_label=r"$\phi_{\rm K}$", unit=r"$\rm rad$", boundary="periodic"),
            "Phi_phi0": UniformDistribution(0, 2 * np.pi, name="initial azimuthal phase", latex_label=r"$\Phi_{\phi_0}$", unit=r"$\rm rad$", boundary="periodic"),
            "Phi_r0": UniformDistribution(0, 2 * np.pi, name="initial radial phase", latex_label=r"$\Phi_{r_0}$", unit=r"$\rm rad$", boundary="periodic"),
            "alpha": UniformDistribution(0, 2 * np.pi, name="right ascension", latex_label=r"$\alpha$", unit=r"$\rm rad$", boundary="periodic"),
            "sin_delta": UniformDistribution(-1 + EPS, 1 - EPS, name="sin declination", latex_label=r"$\sin \delta$", unit=None),
            }

        return priors_in

    def get_output_basis(self) -> list[str]:
        """
        Returns the output basis for the MBHB prior.

        Returns:
            list[str]: A list of parameter names in the output basis.
        """
        output_basis = [
            "m1",
            "m2",
            "a",
            "p0",
            "e0",
            "xI0",
            "d_L",
            "qK",
            "phiK",
            "Phi_phi0",
            "Phi_theta0",
            "Phi_r0",
            "alpha",
            "delta",
        ]
        return output_basis 

    def get_transforms(self) -> TransformContainer:
        """
        Returns the transforms associated with the MBHB prior.

        Returns:
            TransformContainer: The transforms associated with the MBHB prior.
        """

        output_basis = self.get_output_basis()

        fill_dict = {
            "xI0": 1.0,  # inclination
            "Phi_theta0": 0.0,  # initial polar phase
        }

        key_map = {
            "log_m1": "m1",
            "cos_qK": "qK",
            "sin_delta": "delta",
        }

        forward_transforms = {
            "m1": np.exp,  # M
            ("a", "xI0"): decouple_spin_sign,
            "qK": np.arccos,  # qK
            "delta": np.arcsin,  # delta
        }

        inverse_transforms = {
            "m1": np.log,  # M
            ("a", "xI0"): couple_spin_sign,
            "qK": np.cos,  # qS
            "delta": np.sin,  # delta
        }

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
        Returns the periodic parameters associated with the MBHB prior.

        Returns:
            dict[str, float] | None: A dictionary of periodic parameters and their periods, or None if there are no periodic parameters.
        """
        return {
            "phiK": 2 * np.pi,
            "Phi_phi0": 2 * np.pi,
            "Phi_r0": 2 * np.pi,
            "alpha": 2 * np.pi,
        }