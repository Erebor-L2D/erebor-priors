"""
Utility functions for the ereborpriors package and transformations between sampling and physical parameters.
"""

import numpy as np

EPS = 1e-9 # small number to avoid numerical issues in the priors

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

def times_thousands(x: float | np.ndarray) -> np.ndarray:
    """
    Multiply value(s) by 1000. Used for converting from Hz to mHz, Gpc to Mpc, etc.

    Args:
        x (float | np.ndarray): Value(s) to be multiplied.

    Returns:
        np.ndarray: Value(s) multiplied by 1000.
    """
    return np.asarray(x) * 1e3

def over_thousands(x: float | np.ndarray) -> np.ndarray:
    """
    Divide value(s) by 1000. Used for converting from mHz to Hz, Mpc to Gpc, etc.

    Args:
        x (float | np.ndarray): Value(s) to be divided.

    Returns:
        np.ndarray: Value(s) divided by 1000.
    """
    return np.asarray(x) * 1e-3

def ten_to_x(x: float | np.ndarray) -> float | np.ndarray:
    """
    Compute 10 raised to the power of x.

    Args:
        x (float | np.ndarray): Exponent(s).

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