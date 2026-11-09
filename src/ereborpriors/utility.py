"""
Utility functions for the ereborpriors package and transformations between sampling and physical parameters.
"""

import numpy as np

try:
    import cupy as cp # type: ignore
except (ModuleNotFoundError, ImportError):
    cp = None  # type: ignore

EPS = 1e-9 # small number to avoid numerical issues in the priors

def get_array_module(x: float | np.ndarray) -> type:
    if isinstance(x, np.ndarray):
        return np
    elif cp is not None and isinstance(x, cp.ndarray):
        return cp
    else:
        raise TypeError("Input must be a numpy or CuPy array.")

def component_mass_coefficients_from_eta(eta: float | np.ndarray, xp=np) -> tuple[np.ndarray, np.ndarray]:
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
    disc = xp.sqrt(xp.clip(1.0 - 4.0 * eta, 0.0, None))

    # m1 = 0.5 * Mc * eta^(-3/5) * (1 + sqrt(1 - 4 eta)), m2 = 0.5 * Mc * eta^(-3/5) * (1 - sqrt(1 - 4 eta))
    return 0.5 * total_over_mc * (1 + disc), 0.5 * total_over_mc * (1 - disc)


def total_mass_mass_ratio_to_component_masses(total_mass: float | np.ndarray, mass_ratio: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert total mass and mass ratio to component masses.

    Args:
        total_mass (float | np.ndarray): Total mass of the binary system.
        mass_ratio (float | np.ndarray): Mass ratio of the binary system. It is assumed that mass_ratio = m1/m2, where m1 >= m2.

    Returns:
        tuple[np.ndarray, np.ndarray]: Component masses m1 and m2.

    Raises:
        ValueError: If mass_ratio is less than 1, since m1 should be greater than or equal to m2.
    """
    if np.any(mass_ratio < 1):
        raise ValueError("Mass ratio must be greater than or equal to 1 (m1 >= m2).")
    
    m2 = total_mass / (1 + mass_ratio)
    m1 = mass_ratio * m2

    assert np.all(m1 >= m2), "Component mass m1 should be greater than or equal to m2."

    return m1, m2

def component_masses_to_total_mass_mass_ratio(m1: float | np.ndarray, m2: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert component masses to total mass and mass ratio.

    Args:
        m1 (float | np.ndarray): Mass of the first component (m1 >= m2).
        m2 (float | np.ndarray): Mass of the second component (m2 <= m1).

    Returns:
        tuple[np.ndarray, np.ndarray]: Total mass and mass ratio.

    Raises:
        ValueError: If m1 is less than m2, since m1 should be greater than or equal to m2.
    """
    if np.any(m1 < m2):
        raise ValueError("Component mass m1 must be greater than or equal to m2.")
    
    total_mass = m1 + m2
    mass_ratio = m1 / m2

    assert np.all(mass_ratio >= 1), "Mass ratio should be greater than or equal to 1 (m1 >= m2)."

    return total_mass, mass_ratio

def chirp_mass_eta_to_component_masses(chirp_mass: float | np.ndarray, eta: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert chirp mass and symmetric mass ratio to total mass and mass ratio.

    Args:
        chirp_mass (float | np.ndarray): Chirp mass of the binary system.
        eta (float | np.ndarray): Symmetric mass ratio of the binary system.

    Returns:
        tuple[np.ndarray, np.ndarray]: Component masses m1 and m2.

    Raises:
        ValueError: If eta is not in the range (0, 0.25], since eta = m1*m2/(m1+m2)^2 and is maximized at 0.25 for equal masses.
    """
    if np.any(eta <= 0) or np.any(eta > 0.25):
        raise ValueError("Symmetric mass ratio eta must be in the range (0, 0.25].")
    
    c1, c2 = component_mass_coefficients_from_eta(eta)
    m1, m2 = c1 * chirp_mass, c2 * chirp_mass

    return m1, m2

def component_masses_to_chirp_mass_eta(m1: float | np.ndarray, m2: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert component masses to chirp mass and symmetric mass ratio.

    Args:
        m1 (float | np.ndarray): Mass of the first component (m1 >= m2).
        m2 (float | np.ndarray): Mass of the second component (m2 <= m1).

    Returns:
        tuple[np.ndarray, np.ndarray]: Chirp mass and symmetric mass ratio.

    Raises:
        ValueError: If m1 is less than m2, since m1 should be greater than or equal to m2.
    """
    if np.any(m1 < m2):
        raise ValueError("Component mass m1 must be greater than or equal to m2.")
    
    total_mass = m1 + m2
    eta = (m1 * m2) / (total_mass ** 2)
    chirp_mass = total_mass * eta ** (3 / 5)

    return chirp_mass, eta

def times_thousands(x: float | np.ndarray) -> np.ndarray:
    """
    Multiply value(s) by 1000. Used for converting from Hz to mHz, Gpc to Mpc, etc.

    Args:
        x (float | np.ndarray | cp.ndarray): Value(s) to be multiplied.

    Returns:
        np.ndarray: Value(s) multiplied by 1000.
    """
    return x * 1e3

def over_thousands(x: float | np.ndarray) -> np.ndarray:
    """
    Divide value(s) by 1000. Used for converting from mHz to Hz, Mpc to Gpc, etc.

    Args:
        x (float | np.ndarray | cp.ndarray): Value(s) to be divided.

    Returns:
        np.ndarray: Value(s) divided by 1000.
    """
    return x * 1e-3

def ten_to_x(x: float | np.ndarray) -> float | np.ndarray:
    """
    Compute 10 raised to the power of x.

    Args:
        x (float | np.ndarray | cp.ndarray): Exponent(s).

    Returns:
        float | np.ndarray: 10 raised to the power of x.
    """
    return 10 ** x

def decouple_spin_sign(a, xI0):
        # Decouple the spin sign from a

        xI0 = np.sign(a) * 1.0  # inclination is either 1 or -1 depending on the sign of a
        a = np.abs(a)  # take the absolute value of a
        return a, xI0

def couple_spin_sign(a, xI0):
    # Couple the spin sign back into a
    a = np.sign(xI0) * a  # restore the sign of a based on xI0
    xI0 = 1.0
    return a, xI0