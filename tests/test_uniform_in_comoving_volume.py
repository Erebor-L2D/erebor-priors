"""
Tests for ``UniformInComovingVolume``. Reference values come straight from astropy,
never from the prior's own interpolation grid.
"""
from __future__ import annotations
import numpy as np
import pytest
import astropy.cosmology as cosmo
import astropy.units as u
from astropy.cosmology import z_at_value
from scipy import stats

from ereborpriors.base import GFBasePrior
from ereborpriors.priors import UniformInComovingVolume
from ereborpriors.stock.mojitolight.mbhb import MBHBPrior

# Luminosity-distance range (Gpc) of the MBHB prior in stock/mojitolight/mbhb.py
D_MIN, D_MAX = 1.0, 150.0


@pytest.fixture(scope="module", params=["Planck18", "WMAP9"])
def cosmology_name(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def prior(cosmology_name) -> UniformInComovingVolume:
    return UniformInComovingVolume(D_MIN, D_MAX, cosmology=cosmology_name)


def _redshift(cosmology, d_l) -> np.ndarray:
    return z_at_value(cosmology.luminosity_distance, np.atleast_1d(d_l) * u.Gpc).value


def _astropy_pdf(cosmology, d_l) -> np.ndarray:
    """p(d_L) = (dV_c/dz) / (dd_L/dz) / (V_c(z_max) - V_c(z_min)), all from astropy."""
    z = _redshift(cosmology, d_l)
    dv_dz = 4 * np.pi * cosmology.differential_comoving_volume(z).to_value(u.Gpc**3 / u.sr)
    h = 1e-5
    ddl_dz = (
        cosmology.luminosity_distance(z + h) - cosmology.luminosity_distance(z - h)
    ).to_value(u.Gpc) / (2 * h)
    v_min, v_max = cosmology.comoving_volume(_redshift(cosmology, [D_MIN, D_MAX])).to_value(u.Gpc**3)
    return dv_dz / ddl_dz / (v_max - v_min)


def test_pdf_matches_astropy(prior, cosmology_name):
    d_l = np.geomspace(1.5, 140.0, 7)
    expected = _astropy_pdf(getattr(cosmo, cosmology_name), d_l)

    np.testing.assert_allclose(np.exp(prior.logpdf(d_l)), expected, rtol=1e-3)


def test_pdf_integrates_to_one(prior):
    grid = np.linspace(D_MIN, D_MAX, 20_001)

    assert np.trapezoid(np.exp(prior.logpdf(grid)), grid) == pytest.approx(1.0, abs=1e-3)


def test_logpdf_is_minus_inf_outside_range(prior):
    out = prior.logpdf(np.array([0.5, D_MIN - 1e-9, D_MAX + 1e-9, 200.0]))

    assert np.all(np.isneginf(out))


def test_eryn_sees_the_distance_range(prior):
    # eryn's is_in_prior_range / cdf / width read prior.minimum and prior.maximum
    np.testing.assert_array_equal(
        prior.is_in_prior_range(np.array([0.5, 10.0, 200.0])), [False, True, False]
    )


def test_rejects_curved_cosmology(monkeypatch):
    # astropy's built-in realizations are all flat, so register a curved one under a name
    monkeypatch.setattr(cosmo, "CurvedTestCosmology", cosmo.LambdaCDM(H0=70, Om0=0.3, Ode0=0.6), raising=False)

    with pytest.raises(ValueError, match="flat"):
        UniformInComovingVolume(D_MIN, D_MAX, cosmology="CurvedTestCosmology")


def test_rvs_are_uniform_in_comoving_volume(prior, cosmology_name):
    cosmology = getattr(cosmo, cosmology_name)
    z_grid = np.linspace(*_redshift(cosmology, [D_MIN, D_MAX]), 5000)
    d_l_grid = cosmology.luminosity_distance(z_grid).to_value(u.Gpc)
    v_grid = cosmology.comoving_volume(z_grid).to_value(u.Gpc**3)
    cdf_grid = (v_grid - v_grid[0]) / (v_grid[-1] - v_grid[0])

    np.random.seed(1234)
    samples = prior.rvs(50_000)

    assert samples.min() >= D_MIN and samples.max() <= D_MAX
    assert stats.kstest(samples, lambda x: np.interp(x, d_l_grid, cdf_grid)).pvalue > 0.01


def test_survives_yaml_round_trip(tmp_path):
    # WMAP9, not the default, so a cosmology lost on the way shows up as a ~3% density error
    original = GFBasePrior("mbh", {"d_L": UniformInComovingVolume(D_MIN, D_MAX, cosmology="WMAP9")})
    original.to_yaml(tmp_path / "prior.yaml")

    loaded = GFBasePrior.from_yaml(tmp_path / "prior.yaml")

    d_l = np.geomspace(1.5, 140.0, 7)
    np.testing.assert_allclose(
        np.exp(loaded.priors_in["d_L"].logpdf(d_l)), _astropy_pdf(cosmo.WMAP9, d_l), rtol=1e-3
    )


def test_mbhb_distance_prior_is_uniform_in_comoving_volume():
    d_l = np.array([0.9, 1.5, 10.0, 50.0, 140.0, 151.0])

    pdf = np.exp(MBHBPrior(observation_time=1e7).priors_in["d_L"].logpdf(d_l))

    np.testing.assert_allclose(pdf[1:-1], _astropy_pdf(cosmo.Planck18, d_l[1:-1]), rtol=1e-3)
    assert pdf[0] == 0.0 and pdf[-1] == 0.0
