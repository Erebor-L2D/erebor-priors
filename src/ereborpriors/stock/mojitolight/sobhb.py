from __future__ import annotations
import typing

import numpy as np
from eryn.priors import UniformDistribution
from eryn.utils.transform import TransformContainer

from ereborpriors.base import GFBasePrior
from ereborpriors.priors import ChirpMassEtaUniformComponentMasses, UniformInComovingVolume
from ereborpriors.utility import over_thousands, times_thousands, EPS, component_masses_to_chirp_mass_eta, chirp_mass_eta_to_component_masses

# Default [min, max] (Msun) of the primary (m1) and secondary (m2) mass cutting the joint (Mc, eta) prior.
# Both are lisatools' shared SOBBHTFSettings.component_mass_lims = (5, 130).
SOBHB_DEFAULT_M1_LIMS = (5.0, 100.0)
SOBHB_DEFAULT_M2_LIMS = (1.0, 80.0)
SOBHB_DEFAULT_ETA_LIMS = (0.0 + EPS, 0.25 - EPS)  # symmetric mass ratio limits (m1 >= m2)

# Per-source (index = Source 0..5) search tiles 
# Note in our usuaul run we do not sample source 5 
TILES_F0_MIN = [  # Hz
    0.01523044572556501, 
    0.01,
    0.0055,
    0.01523044572556501,
    0.007,
    0.007,  # Source 5 (placeholder tile: in the data but not sampled)
]
TILES_F0_MAX = [  # Hz
    0.01566440950100069,
    0.010539116009603954,
    0.0056,
    0.01566440950100069,
    0.0071,
    0.0071,  # Source 5
]
TILES_CHIRP_MASS_MIN = [  # Msun
    58.31641286375667, 
    29.783729431335413,
    60.0,
    37.59285709049653,
    70.0,
    45.0,  # Source 5
]

TILES_CHIRP_MASS_MAX = [  # Msun
    60.59973201883494,
    32.498510135448974,
    65.0,
    39.88177477999402,
    75.0,
    50.0,  # Source 5
]

CHIRP_MASS_SORTING = np.argsort(TILES_CHIRP_MASS_MIN)  # sort by chirp mass

SORTED_TILES_F0_MIN = (np.array(TILES_F0_MIN)[CHIRP_MASS_SORTING]).tolist()
SORTED_TILES_F0_MAX = (np.array(TILES_F0_MAX)[CHIRP_MASS_SORTING]).tolist()
SORTED_TILES_CHIRP_MASS_MIN = (np.array(TILES_CHIRP_MASS_MIN)[CHIRP_MASS_SORTING]).tolist()
SORTED_TILES_CHIRP_MASS_MAX = (np.array(TILES_CHIRP_MASS_MAX)[CHIRP_MASS_SORTING]).tolist()

class SOBHBPrior(GFBasePrior):
    """
    Prior for the gwtf (time-frequency) SOBHB branch, inheriting from GFBasePrior.

    Shapes:
    - (Mc, eta) jointly uniform in component masses (cut by the component-mass box),
    - dist uniform in comoving volume (Planck15, sampled in Gpc), dec uniform on the sphere,
    - everything else uniform.

    The orbital phase is sampled as ``phi_0``, the orbital phase at the data start (where the GW
    frequency is ``f0``), and passed to the waveforms unchanged: GWTF_generator and
    SOBBHTaylorT3TDIWaveform convert it to their coalescence phase themselves
    (``phase_at_coalescence=False``).
    """

    branch_name = "sobbh"

    @classmethod
    def prior_factory(cls,
                      f0_min: float = 1e-3,
                      f0_max: float = 0.1,
                      chirp_mass_min: float = 5.0,
                      chirp_mass_max: float = 100.0,
                      ) -> SOBHBPrior:
        """
        Factory method to create an instance of SOBHBPrior.

        Args:
            f0_min (float): Minimum value for f0 (Hz, at data start) for a single source.
                Default 1e-3.
            f0_max (float): Maximum value for f0 (Hz, at data start) for a single source.
                Default 0.1.
            chirp_mass_min (float): Minimum value for chirp mass (Msun) for a single source.
                Default 5.0.
            chirp_mass_max (float): Maximum value for chirp mass (Msun) for a single source.
                Default 100.0.

        Returns:
            SOBHBPrior: An instance of SOBHBPrior.
        """
        priors_in = cls.build_priors(f0_min=f0_min, f0_max=f0_max, chirp_mass_min=chirp_mass_min, chirp_mass_max=chirp_mass_max)
        return cls(branch_name=cls.branch_name, priors_in=priors_in)

    @classmethod
    def per_source_prior_factory(cls, 
                                sort_with_chirp_mass: bool = False,
                                num_leaves: typing.Optional[int] = None
                                ) -> list[SOBHBPrior]:
        """
        Factory method to create a list of per-source SOBHB priors.

        Args:
            sort_with_chirp_mass (bool): If True, the returned list of priors will be sorted by chirp mass.
                Default is False.

        Returns:
            list[SOBHBPrior]: A list of per-source SOBHB priors.
        """
        if sort_with_chirp_mass:
            f0_min = SORTED_TILES_F0_MIN
            f0_max = SORTED_TILES_F0_MAX
            chirp_mass_min = SORTED_TILES_CHIRP_MASS_MIN
            chirp_mass_max = SORTED_TILES_CHIRP_MASS_MAX
        else:
            f0_min = TILES_F0_MIN
            f0_max = TILES_F0_MAX
            chirp_mass_min = TILES_CHIRP_MASS_MIN
            chirp_mass_max = TILES_CHIRP_MASS_MAX
            
        if not (len(f0_min) == len(f0_max) == len(chirp_mass_min) == len(chirp_mass_max)):
            raise ValueError("All input lists must have the same length.")

        if num_leaves is not None:
            if num_leaves > len(f0_min):
                raise ValueError("num_leaves cannot be greater than the number of sources.")
        else:
            num_leaves = len(f0_min)

        return [
            cls.prior_factory(f0_min=f0_min[i],
                                f0_max=f0_max[i],
                                chirp_mass_min=chirp_mass_min[i],
                                chirp_mass_max=chirp_mass_max[i])
            for i in range(num_leaves)
        ]

    @staticmethod
    def build_priors(f0_min: float = 1e-3,
                     f0_max: float = 0.1,
                     chirp_mass_min: float = 5.0,
                     chirp_mass_max: float = 100.0,
                     ) -> dict[str, typing.Any]:
        """
        Builds the per-source priors for the SOBHB sources.

        Args:
            f0_min (float): Minimum value for f0 (Hz, at data start)
            f0_max (float): Maximum value for f0 (Hz, at data start)
            chirp_mass_min (float): Minimum value for chirp mass (Msun) for a single source.
            chirp_mass_max (float): Maximum value for chirp mass (Msun) for a single source.

        Returns:
            dict[str, typing.Any]: The priors, in sampling column order.
        """

        # ProbDistContainer assigns columns by insertion order.
        priors_in = {
            ("Mc", "eta"): ChirpMassEtaUniformComponentMasses(chirp_mass_min, chirp_mass_max, *SOBHB_DEFAULT_ETA_LIMS, *SOBHB_DEFAULT_M1_LIMS, *SOBHB_DEFAULT_M2_LIMS),
            "s1": UniformDistribution(-1.0 + EPS, 1.0 - EPS, name="s1", latex_label=r"$\chi_1$", unit=None),
            "s2": UniformDistribution(-1.0 + EPS, 1.0 - EPS, name="s2", latex_label=r"$\chi_2$", unit=None),
            "d_L": UniformInComovingVolume(1e-7, 1., name="luminosity distance", latex_label=r"$d_L$", unit=r"$\rm Gpc$"),
            "cos_iota": UniformDistribution(-1.0 + EPS, 1.0 - EPS, name="cos inclination", latex_label=r"$\cos \iota$", unit=None),
            "f0": UniformDistribution(f0_min, f0_max, name="Starting frequency", latex_label=r"$f_0$", unit=r"$\rm Hz$"),
            "phi_0": UniformDistribution(0, 2 * np.pi, name="Orbital phase at data start", latex_label=r"$\phi_0$", unit=r"$\rm rad$", boundary="periodic"),
            "psi": UniformDistribution(0, 2 * np.pi, name="polarization angle", latex_label=r"$\psi$", unit=r"$\rm rad$", boundary="periodic"),
            "alpha": UniformDistribution(0, 2 * np.pi, name="right ascension", latex_label=r"$\alpha$", unit=r"$\rm rad$", boundary="periodic"),
            "sin_delta": UniformDistribution(-1 + EPS, 1 - EPS, name="sin declination", latex_label=r"$\sin \delta$", unit=None),
        }

        return priors_in

    def get_output_basis(self) -> list[str]:
        """
        Returns the output basis for the SOBHB prior (the GWTF_generator basis).

        Returns:
            list[str]: A list of parameter names in the output basis.
        """
        return ["m1", "m2", "s1", "s2", "d_L", "iota", "f0", "phi_0", "psi", "alpha", "delta"]

    def get_transforms(self) -> TransformContainer:
        """
        Returns the transforms associated with the SOBHB prior. The output basis is the
        GWTF_generator / SOBBHTaylorT3TDIWaveform basis, in the same column order as the
        sampling basis (d_L in Gpc, as both waveforms expect). ``phi_0`` passes through unchanged.

        Returns:
            TransformContainer: The transforms associated with the SOBHB prior.
        """
        output_basis = self.get_output_basis()

        key_map = {
            "Mc": "m1",
            "eta": "m2",
            "cos_iota": "iota",
            "sin_delta": "delta",        
        }

        forward_transforms = {
            ("m1", "m2"): chirp_mass_eta_to_component_masses,
            "delta": np.arcsin,
            "iota": np.arccos,
        }

        inverse_transforms = {
            ("Mc", "eta"): component_masses_to_chirp_mass_eta,
            "delta": np.sin,
            "iota": np.cos,
        }

        return TransformContainer(
            input_basis=self.get_input_basis(),
            output_basis=output_basis,
            key_map=key_map,
            parameter_transforms=forward_transforms,
            inverse_parameter_transforms=inverse_transforms,
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
            "phi_0": 2 * np.pi,
            "psi": 2 * np.pi,
            "alpha": 2 * np.pi,
        }

if __name__ == "__main__":
    sobhb_prior = SOBHBPrior.prior_factory()  # all defaults (SOBHB_DEFAULT_LIMS)
    print("Parameter Names:", sobhb_prior.parameter_names)
    print("Parameter LaTeX Names:", sobhb_prior.parameter_latex_labels)
    print("Periodic:", sobhb_prior.get_periodic())
    sobhb_prior.to_yaml("sobhb_prior.yaml")  # Save the prior to a YAML file
    print("Transforms:", sobhb_prior.get_transforms())

