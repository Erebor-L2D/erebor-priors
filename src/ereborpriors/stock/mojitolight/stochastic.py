from __future__ import annotations
import typing

import numpy as np
from eryn.priors import UniformDistribution, CosineUniform, SineUniform, LogUniform, UniformInVolume
from eryn.utils.transform import TransformContainer

from ereborpriors.base import GFBasePrior
from ereborpriors.utility import ten_to_x

ALLOWED_NOISE_MODELS = ["parametric-symmetric", "parametric-asymmetric", "splines"]
MOSA_NAMES = ["12", "23", "31", "13", "32", "21"]

class NoisePrior(GFBasePrior):
    """
    A class representing the noise prior, inheriting from GFBasePrior.
    """

    def __init__(self, priors_in: dict[str, typing.Any] = None, name: str = "psd", **kwargs):
        """
        Initializes the NoisePrior instance.

        Args:
            priors_in (dict[str, typing.Any]): A dictionary containing the priors for the noise prior.
            name (str): The name of the prior. Default is "psd".
            **kwargs: Additional keyword arguments to be passed to the GFBasePrior constructor.
        """

        super().__init__(branch_name=name, priors_in=priors_in, **kwargs)

    @classmethod
    def prior_factory(cls, model_type: str) -> NoisePrior:
        """
        Factory method to create an instance of NoisePrior.

        Args:
            model_type (str): The type of noise model. Can be "parametric-symmetric", "parametric-asymmetric", or "splines".

        Returns:
            NoisePrior: An instance of NoisePrior.
        """

        if model_type not in ALLOWED_NOISE_MODELS:
            raise ValueError(f"Invalid model_type '{model_type}'. Allowed values are: {ALLOWED_NOISE_MODELS}")

        priors_in = cls.build_priors(cls, model_type)
        return cls(priors_in=priors_in)

    def build_priors(self, model_type: str) -> dict[str, typing.Any]:
        """
        Builds the priors for the noise prior.

        Args:
            model_type (str): The type of noise model. Can be "parametric-symmetric", "parametric-asymmetric", or "splines".

        Returns:
            dict[str, typing.Any]: A dictionary containing the priors for the MBHB prior.
        """

        prior_oms = UniformDistribution(-12, -10, name="oms log-amplitude", latex_label=r"$ \log S_{\rm oms}$", unit=r"$\rm m / \sqrt{\rm Hz}$")
        prior_tm = UniformDistribution(-16, -13, name="tm log-amplitude", latex_label=r"$ \log S_{\rm tm}$", unit=r"$\rm m s^{-2} / \sqrt{\rm Hz}$")

        if model_type == "parametric-symmetric":
            priors_in = {
                "logS_oms": prior_oms,
                "logS_tm": prior_tm,
            }
        elif model_type == "parametric-asymmetric":
            priors_in = {}

            for mosa in MOSA_NAMES:
                priors_in[f"logS_oms_{mosa}"] = prior_oms
                priors_in[f"logS_tm_{mosa}"] = prior_tm
        elif model_type == "splines":
            raise NotImplementedError("Spline noise model is not implemented yet.")

        return priors_in

    def get_output_basis(self) -> list[str]:
        """
        Returns the output basis for the MBHB prior.

        Returns:
            list[str]: A list of parameter names in the output basis.
        """
        output_basis = []
        if len(self.priors_in) == 2:
            output_basis = ["S_oms", "S_tm"]
        else:
            for mosa in MOSA_NAMES:
                output_basis.append(f"S_oms_{mosa}")
                output_basis.append(f"S_tm_{mosa}")

        return output_basis

    def get_transforms(self) -> TransformContainer:
        """
        Returns the transforms associated with the MBHB prior.

        Returns:
            TransformContainer: The transforms associated with the MBHB prior.
        """

        output_basis = self.get_output_basis()

        if len(output_basis) == 2:
            key_map = {
                "logS_oms": "S_oms",
                "logS_tm": "S_tm",
            }

            forward_transforms = {
                "S_oms": ten_to_x,
                "S_tm": ten_to_x,
            }

            inverse_transforms = {
                "S_oms": np.log10,
                "S_tm": np.log10,
            }

        else:
            key_map = {}
            forward_transforms = {}
            inverse_transforms = {}

            for mosa in MOSA_NAMES:
                key_map[f"logS_oms_{mosa}"] = f"S_oms_{mosa}"
                key_map[f"logS_tm_{mosa}"] = f"S_tm_{mosa}"

                forward_transforms[f"S_oms_{mosa}"] = ten_to_x
                forward_transforms[f"S_tm_{mosa}"] = ten_to_x

                inverse_transforms[f"S_oms_{mosa}"] = np.log10
                inverse_transforms[f"S_tm_{mosa}"] = np.log10

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
        Returns the periodic parameters associated with the prior.

        Returns:
            dict[str, float] | None: A dictionary of periodic parameters and their periods, or None if there are no periodic parameters.
        """
        return None

class GalacticForegroundPrior(GFBasePrior):
    """
    A class representing the galactic foreground prior, inheriting from GFBasePrior.
    """

    def __init__(self, priors_in: dict[str, typing.Any] = None, name: str = "galfor", **kwargs):
        """
        Initializes the GalacticForegroundPrior instance.

        Args:
            priors_in (dict[str, typing.Any]): A dictionary containing the priors for the galactic foreground prior.
            name (str): The name of the prior. Default is "galfor".
            **kwargs: Additional keyword arguments to be passed to the GFBasePrior constructor.
        """
        if priors_in is None:
            priors_in = self.build_priors()

        super().__init__(branch_name=name, priors_in=priors_in, **kwargs)

    @classmethod
    def prior_factory(cls) -> GalacticForegroundPrior:
        """
        Factory method to create an instance of GalacticForegroundPrior.

        Returns:
            GalacticForegroundPrior: An instance of GalacticForegroundPrior.
        """
        priors_in = cls.build_priors(cls)
        return cls(priors_in=priors_in)

    def build_priors(self) -> dict[str, typing.Any]:
        """
        Builds the priors for the galactic foreground prior.

        Returns:
            dict[str, typing.Any]: A dictionary containing the priors for the galactic foreground prior.
        """

        priors_in = {
            "log10_A": UniformDistribution(-48, -40, name="log10 amplitude", latex_label=r"$\log_{10} A$", unit=None),
            "alpha_gal": UniformDistribution(1.0, 20.0, name="spectral index", latex_label=r"$\alpha_{\rm gal}$", unit=None),
            "log10_f1": UniformDistribution(-4.0, -2.0, name="log10 f1", latex_label=r"$\log_{10} f_1$", unit=None),
            "log10_fknee": UniformDistribution(-4.0, -2.0, name="log10 fknee", latex_label=r"$\log_{10} f_{\rm knee}$", unit=None),
            "log10_f2": UniformDistribution(-3.0, -1.0, name="log10 f2", latex_label=r"$\log_{10} f_2$", unit=None),
        }

        return priors_in

    def get_output_basis(self) -> list[str]:
        """
        Returns the output basis for the galactic foreground prior.

        Returns:
            list[str]: A list of parameter names in the output basis.
        """
        return ["A", "alpha_gal", "f1", "fknee", "f2"]

    def get_transforms(self) -> TransformContainer:
        """
        Returns the transforms associated with the galactic foreground prior.

        Returns:
            TransformContainer: The transforms associated with the galactic foreground prior.
        """

        output_basis = self.get_output_basis()

        key_map = {
            "log10_A": "A",
            "log10_f1": "f1",
            "log10_fknee": "fknee",
            "log10_f2": "f2",
        }

        forward_transforms = {
            "log10_A": ten_to_x,
            "log10_f1": ten_to_x,
            "log10_fknee": ten_to_x,
            "log10_f2": ten_to_x,
        }

        inverse_transforms = {
            "A": np.log10,
            "f1": np.log10,
            "fknee": np.log10,
            "f2": np.log10,
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
        Returns the periodic parameters associated with the prior.

        Returns:
            dict[str, float] | None: A dictionary of periodic parameters and their periods, or None if there are no periodic parameters.
        """
        return None