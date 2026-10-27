"""
Base class for all priors in the Erebors Priors package.
"""

from __future__ import annotations
import typing
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
import eryn
from eryn.priors import ProbDistContainer, Prior, JointPrior
from eryn.utils.transform import TransformContainer
from eryn.utils.periodic import PeriodicContainer

from ereborpriors import priors
from ereborpriors.io import _Loader

PRIOR_MODULES = [eryn.priors, priors]

def get_supported_priors() -> list[str]:
    """
    Get a list of all supported prior names in the eryn.priors module.

    Returns:
        list[str]: A list of supported prior names.
    """
    supported_priors = []
    for prior_module in PRIOR_MODULES:
        for cls in prior_module.__dict__.values():
            if isinstance(cls, type) and issubclass(cls, (Prior, JointPrior)) and cls is not Prior and cls is not JointPrior:
                supported_priors.append(cls.__name__)
    return supported_priors

def get_prior_class(prior_name: str) -> Prior | JointPrior:
    """
    Get the eryn prior class corresponding to the given prior name.

    Args:
        prior_name (str): The name of the prior.
    
    Returns:
        Prior: The corresponding eryn prior class.
    """
    supported_priors = get_supported_priors()
    if prior_name not in supported_priors:
        raise ValueError(f"Prior '{prior_name}' is not supported. Supported priors are: {supported_priors}")
    prior = None
    for prior_module in PRIOR_MODULES:
        prior = getattr(prior_module, prior_name, None)
        if prior is not None:
            break

    if prior is None:
        raise ValueError(f"Prior '{prior_name}' not found in the supported prior modules.")
    return prior

def get_prior_instance(prior_type: str, name: str, prior_kwargs: dict[str, typing.Any]) -> Prior | JointPrior:
    """
    Get an instance of the eryn prior class corresponding to the given prior name.

    Args:
        prior_type (str): The type of the prior.
        name (str): The name of the parameter associated with the prior.
        prior_kwargs (dict[str, typing.Any]): A dictionary of keyword arguments to pass to the prior class constructor.
        *args: Positional arguments to pass to the prior class constructor.
        **kwargs: Keyword arguments to pass to the prior class constructor.
    
    Returns:
        Prior | JointPrior: An instance of the corresponding eryn prior class.
    """
    prior_class = get_prior_class(prior_type)
    if issubclass(prior_class, JointPrior):
        name_key = "names" 
    else:
        name_key = "name"

    if name_key not in prior_kwargs:
        prior_kwargs[name_key] = name
    return prior_class(**prior_kwargs)

def get_prior_from_entry(entry: typing.Dict[str, typing.Any]) -> tuple[str, Prior | JointPrior]:
    """
    Get a prior instance from a dictionary entry.

    Args:
        entry (dict): A dictionary containing the prior information. It should have the following structure:
            {
                "name": parameter_name,
                "prior_type": prior_name,
                "prior_kwargs": {"kwarg1": value1, "kwarg2": value2, ...}
            }

    Returns:
        tuple[str, Prior | JointPrior]: A tuple containing the parameter name and the corresponding prior instance.
    """
    name = entry["name"]
    prior_type = entry["prior_type"]
    prior_kwargs = entry.get("prior_kwargs", {})

    print(f"Creating prior for parameter '{name}' of type '{prior_type}' with kwargs: {prior_kwargs}")

    return (name, get_prior_instance(prior_type, name=name, prior_kwargs=prior_kwargs))

def get_priors_dict(entries: typing.List[typing.Dict[str, typing.Any]]) -> dict[str, Prior | JointPrior]:
    """
    Get a dictionary of prior instances from a list of dictionary entries.

    Args:
        entries (list): A list of dictionaries, each containing the prior information. Each dictionary should have the following structure:
            {
                "name": parameter_name,
                "prior_type": prior_name,
                "prior_kwargs": {"kwarg1": value1, "kwarg2": value2, ...}
            }

    Returns:
        dict[str, Prior | JointPrior]: A dictionary mapping parameter names to their corresponding prior instances.
    """
    return {name: prior for name, prior in (get_prior_from_entry(entry) for entry in entries)}

class GFBasePrior(ProbDistContainer):
    """
    Base class for all priors in the Erebors Priors package.
    """

    def __init__(self, branch_name: str, priors_in: dict[str, Prior | JointPrior], use_cupy: bool = False, return_gpu: bool = False,):
        """
        Initializes the GFBasePrior instance.

        Args:
            branch_name (str | None): The name of the branch associated with this prior. Default is None.
            priors_in (dict[str, Prior | JointPrior]): A dictionary of prior instances.
            use_cupy (bool): Whether to use CuPy for computations. Default is False.
            return_gpu (bool): Whether to return results on the GPU. Default is False.
        """
        super().__init__(priors_in=priors_in, use_cupy=use_cupy, return_gpu=return_gpu)
        self.branch_name = branch_name

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        """
        Returns the keyword arguments for the current prior.

        Returns:
            dict[str, typing.Any]: A dictionary of keyword arguments for the current prior.
        """
        return {"use_cupy": self.use_cupy, "return_gpu": self.return_gpu}

    @classmethod
    def from_yaml(cls, file_path: str) -> GFBasePrior:
        """
        Loads a prior from a YAML file.

        Args:
            file_path (str): Path to the file containing the prior. Each line of the file should contain 
                the parameter(s) name(s), the prior class, and the arguments of the prior. For example:

        Returns:
            GFBasePrior: An instance of the prior loaded from the file.
        """
        with open(file_path, "r") as f:
            data = yaml.load(f, Loader=_Loader)

        branch_name = data.get("branch_name", None)
        prior_kwargs = data.get("kwargs", {})
        priors = data.get("priors", [])

        prior_dict = get_priors_dict(priors)

        return cls(branch_name=branch_name, priors_in=prior_dict, **prior_kwargs)

    def to_yaml(self, file_path: str) -> None:
        """
        Saves the prior to a YAML file.

        Args:
            file_path (str): Path to the file where the prior will be saved.
        """
        data = {
            "branch_name": self.branch_name,
            "priors": [
                {
                    "name": name,
                    "prior_type": prior.__class__.__name__,
                    "prior_kwargs": {k: getattr(prior, k) for k in prior._repr_keys},
                }
                for name, prior in self.priors_in.items()
            ],
        }
        # Save the kwargs if they are not empty
        if self.kwargs:
            for k, v in self.kwargs.items():
                data[k] = v
                
        with open(file_path, "w") as f:
            yaml.dump(data, f)

    @classmethod
    def get_transforms(cls) -> TransformContainer:
        """
        Returns the transforms associated with the prior.

        Returns:
            TransformContainer: The transforms associated with the prior.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    def get_periodic(cls) -> dict[str, float] | None:
        """
        Returns the periodic parameters associated with the prior.

        Returns:
            dict[str, float] | None: A dictionary of periodic parameters and their periods, or None if there are no periodic parameters.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_input_basis(self) -> list[str]:
        """
        Returns the input basis for the prior.

        Returns:
            list[str]: The input basis for the prior.
        """
        names = list(self.priors_in.keys()).copy()
        input_basis = []
        for name in names:
            if isinstance(name, (str, int)):
                input_basis.append(name)
            elif isinstance(name, tuple):
                input_basis.extend(name)
            else:
                raise ValueError(f"Unsupported name type: {type(name)}")
        return input_basis

    @classmethod
    def get_output_basis(cls) -> list[str]:
        """
        Returns the output basis for the likelihood.

        Returns:
            list[str]: The output basis for the likelihood.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @property
    def parameter_names(self) -> list[str]:
        """
        Returns the list of parameter names for the current prior.

        Returns:
            list[str]: A list of parameter names.
        """
        return self.get_input_basis()

    @property
    def parameter_latex_labels(self) -> list[str]:
        """
        Returns the list of parameter LaTeX names for the current prior.

        Returns:
            list[str]: A list of parameter LaTeX names.
        """
        latex_labels = [prior.latex_label for prior in self.priors_in.values()]
        # Handle cases where the prior is a JointPrior with multiple parameters
        expanded_labels = []
        for label in latex_labels:
            if isinstance(label, (list, tuple)):
                expanded_labels.extend(label)
            else:
                expanded_labels.append(label)
        return expanded_labels

    def plot(self, fig: mpl.figure.Figure | None = None, axes: list[mpl.axes.Axes] | None = None, num_samples: int = 5000, max_columns: int = 5, **kwargs) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes]]:
        """
        Plots the priors.

        Args:
            fig (matplotlib.figure.Figure | None): The figure to plot on. If None, a new figure will be created.
            axes (list[matplotlib.axes.Axes] | None): The axes to plot on. If None, new axes will be created.
            num_samples (int): The number of samples to draw from the priors for plotting. Default is 1000.
            max_columns (int): The maximum number of columns in the plot grid. Default is 5.
            **kwargs: Additional keyword arguments to be passed to the ProbDistContainer.plot method.

        Returns:
            tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]: The figure and axes used for plotting.
        """

        samples = self.rvs(size=(num_samples,))

        num_dim = samples.shape[-1]

        if fig is None or axes is None:
            num_rows = (num_dim + max_columns - 1) // max_columns
            fig, axes = plt.subplots(num_rows, min(num_dim, max_columns), figsize=(5 * min(num_dim, max_columns), 4 * num_rows))
            axes = axes.flatten() if num_dim > 1 else [axes]

        for i in range(num_dim):
            ax = axes[i]
            ax.hist(samples[:, i], bins=30, density=True, histtype="stepfilled")
            ax.set_xlabel(f"{list(self.priors_in.keys())[i]}")
            ax.set_ylabel("Density")

        return fig, axes
        

if __name__ == "__main__":
    # Example usage
    prior_name = "Normal"
    prior_class = get_prior_class(prior_name)
    print(f"Retrieved prior class: {prior_class}")

    filename = "prior_example.yaml"

    breakpoint()

    pr = GFBasePrior.from_yaml(filename)
    out = "out.yaml"
    pr.to_yaml(out)