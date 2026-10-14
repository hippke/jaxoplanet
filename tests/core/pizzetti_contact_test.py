# mypy: ignore-errors
"""Tests of the contact expansions (power series near the stellar limb)"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoplanet.core import pizzetti, pizzetti_contact
from jaxoplanet.core.limb_dark import light_curve as agol_light_curve

LAWS = {
    "quartic": [0.4, 0.1, 0.2, -0.1],
    "octic": [0.3, 0.2, 0.1, 0.05, 0.02, 0.0, -0.01, -0.02],
    "dodecic": list(np.linspace(0.3, -0.02, 12)),
}
TOL = 1e-10
NZ = 2048


def _x64() -> bool:
    return bool(jax.config.jax_enable_x64)  # type: ignore


needs_x64 = pytest.mark.skipif(not _x64(), reason="requires float64")


@partial(jax.jit, static_argnames=("contact", "min_limb"))
def deficit(u, z, r, contact=True, min_limb=0):
    tol = jnp.asarray(TOL, dtype=z.dtype)
    return pizzetti._flat_deficit(u, z, r, 40, tol, contact=contact, min_limb=min_limb)


def limb_grid(r, n=NZ):
    """Separations from just below the cut to the first contact, densest at the
    contact points, padded to a fixed length with samples far from the transit"""
    d = np.geomspace(1e-13, 0.05, 12)
    z = np.concatenate(
        [
            np.linspace(0.5 * (1 - r), 1 + r, n - 80),
            1 - r + d,
            1 - r - d,
            1 + r - d,
        ]
    )
    z = np.sort(z[(z > r) & (z < 1 + r)])
    return jnp.asarray(np.concatenate([z, np.full(n - z.size, 5.0)]))


def central_depth(u, r):
    c = np.asarray(pizzetti.mu_coefficients(jnp.asarray(u)))
    a = np.arange(c.shape[0])
    return r**2 * abs(c.sum()) / (2 * abs((c / (a + 2)).sum()))


@needs_x64
@pytest.mark.parametrize("g", [0.0, 0.5, 1.0, 2.5])
@pytest.mark.parametrize("j", [0, 3, 40])
def test_tables_against_mpmath(g, j):
    """Both representations (Taylor and connection form) of the universal functions
    against mpmath's Gauss function; the truncated series are compared up to the
    size of their last terms"""
    mpmath = pytest.importorskip("mpmath")
    pv = np.polynomial.polynomial.polyval
    T = pizzetti_contact.table(g)
    n = pizzetti_contact.NS + 1
    m = g + 1.5 + j
    params = [
        (0.5, 0.5, g + 2.5 + j, float(mpmath.beta(0.5, g + 2.0 + j))),
        (-(g + 1.0 + j), 0.5, 1.0, float(mpmath.pi)),
    ]
    for fam, (a, b, c, scale) in enumerate(params):
        for x in (0.1, 0.3, 0.45, 0.7, 0.9):
            expect = scale * float(mpmath.hyp2f1(a, b, c, x))
            if x <= pizzetti_contact.XS:
                got = pv(x, T[fam, j, 0])
                tail = 3 * abs(T[fam, j, 0, n - 1]) * x ** (n - 1)
            else:
                y = 1.0 - x
                got = pv(y, T[fam, j, 1]) + y**m * (
                    np.log(y) * pv(y, T[fam, j, 2]) + pv(y, T[fam, j, 3])
                )
                tail = 3 * max(abs(T[fam, j, 1:, n - 1]).max() * y ** (n - 1), 0.0)
            assert abs(got - expect) <= 1e-10 * abs(expect) + tail + 1e-300


@needs_x64
def test_series_arithmetic():
    n = pizzetti_contact.NS + 1
    rng = np.random.default_rng(1)
    a = np.zeros(n)
    a[0] = 0.7
    a[1:] = 0.1 * rng.normal(size=n - 1) * 0.5 ** np.arange(1, n)
    b = rng.normal(size=n)
    np.testing.assert_allclose(
        pizzetti_contact._smul(jnp.asarray(a), jnp.asarray(b)),
        np.convolve(a, b)[:n],
        atol=1e-12,
    )
    inv = np.asarray(pizzetti_contact._sinv(jnp.asarray(a)))
    np.testing.assert_allclose(np.convolve(inv, a)[:n], np.eye(1, n)[0], atol=1e-12)
    alphas = np.array([-0.5, 0.5, 1.0, 7.0, 150.0])
    pw = np.asarray(
        pizzetti_contact._spow(jnp.asarray(a)[None], jnp.asarray(alphas)[None])
    )[0]
    # integer powers by repeated multiplication
    for k, alpha in enumerate(alphas):
        if alpha == int(alpha):
            expect = np.eye(1, n)[0]
            for _ in range(int(alpha)):
                expect = np.convolve(expect, a)[:n]
            np.testing.assert_allclose(
                pw[k], expect, rtol=1e-10, atol=1e-14 * abs(expect).max()
            )
    # (a^0.5)^2 = a
    half = np.asarray(
        pizzetti_contact._spow(jnp.asarray(a)[None], jnp.asarray([0.5])[None])
    )[0, 0]
    np.testing.assert_allclose(np.convolve(half, half)[:n], a, atol=1e-13)


@needs_x64
@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("r", [0.01, 0.03, 0.1, 0.15])
def test_matches_agol(law, r):
    """The contact expansions reproduce the solution of Agol et al. to far better
    than the tolerance, relative to the central depth"""
    u = jnp.asarray(LAWS[law])
    z = limb_grid(r)
    got = np.asarray(deficit(u, z, jnp.asarray(r)))
    off = np.asarray(deficit(u, z, jnp.asarray(r), contact=False))
    ref = np.asarray(agol_light_curve(u, z, r))
    depth = central_depth(LAWS[law], r)
    np.testing.assert_allclose(got, ref, atol=1e-10 * depth, rtol=0)
    np.testing.assert_allclose(off, ref, atol=1e-10 * depth, rtol=0)
    # beyond the second contact only the contact series (or Agol et al.) are used,
    # and they agree much better than the tolerance
    zn = np.asarray(z)
    out = zn > 1 - r
    np.testing.assert_allclose(got[out], ref[out], atol=1e-11 * depth, rtol=0)
    if r >= 0.05:  # (smaller radii are rejected for the longest law: see below)
        assert np.max(np.abs(got - off)) > 0, "the contact expansions were not used"


@needs_x64
@pytest.mark.parametrize("law", ["quartic", "dodecic"])
@pytest.mark.parametrize("r", [0.03, 0.1])
def test_against_mpmath(law, r):
    """An independent 40-digit reference"""
    pytest.importorskip("mpmath")
    from .pizzetti_reference import flux_ref

    u = jnp.asarray(LAWS[law])
    c = np.asarray(pizzetti.mu_coefficients(u))
    alphas = np.arange(c.shape[0])
    zg = np.asarray(limb_grid(r))
    z = zg[zg < 1 + r]
    z = z[:: max(1, z.size // 14)]
    z = np.concatenate([z, [1 - r, 1 - r + 1e-9, 1 - r - 1e-9, 1 + r - 1e-9]])
    pad = np.full(NZ - z.size, 5.0)
    got = np.asarray(deficit(u, jnp.asarray(np.concatenate([z, pad])), jnp.asarray(r)))[
        : z.size
    ]
    expect = np.array([flux_ref(zi, r, alphas, c) - 1.0 for zi in z])
    depth = central_depth(LAWS[law], r)
    np.testing.assert_allclose(got, expect, atol=1e-10 * depth, rtol=0)


@needs_x64
def test_large_radius_falls_back_to_agol():
    """Where the probes find the series inaccurate the result is that of Agol et al."""
    u = jnp.asarray(LAWS["quartic"])
    for r in (0.3, 0.6, 1.5):
        z = limb_grid(r)
        got = np.asarray(deficit(u, z, jnp.asarray(r)))
        off = np.asarray(deficit(u, z, jnp.asarray(r), contact=False))
        np.testing.assert_array_equal(got, off)


@needs_x64
def test_below_the_threshold_nothing_changes():
    u = jnp.asarray(LAWS["quartic"])
    z = limb_grid(0.1)
    got = np.asarray(deficit(u, z, jnp.asarray(0.1), min_limb=10**9))
    off = np.asarray(deficit(u, z, jnp.asarray(0.1), contact=False))
    np.testing.assert_array_equal(got, off)


@pytest.mark.skipif(_x64(), reason="float32 only")
def test_float32_does_not_use_the_expansions():
    """The coefficients span ~38 orders of magnitude, which float32 cannot hold: the
    path is not used and the results are those of the solution of Agol et al."""
    u = jnp.asarray(LAWS["quartic"], dtype=jnp.float32)
    z = limb_grid(0.1).astype(jnp.float32)
    got = np.asarray(deficit(u, z, jnp.asarray(0.1, dtype=jnp.float32)))
    off = np.asarray(deficit(u, z, jnp.asarray(0.1, dtype=jnp.float32), contact=False))
    np.testing.assert_array_equal(got, off)


@needs_x64
@pytest.mark.parametrize("law", ["quartic", "dodecic"])
def test_gradients_match_agol(law):
    u = jnp.asarray(LAWS[law])
    r = jnp.asarray(0.08)
    z = limb_grid(0.08)
    weights = jnp.cos(3.0 * z) * (z < 1.08)

    def loss(contact):
        return lambda u, z, r: jnp.sum(deficit(u, z, r, contact=contact) * weights)

    def loss_agol(u, z, r):
        return jnp.sum(agol_light_curve(u, z, r) * weights)

    calc = jax.grad(loss(True), argnums=(0, 2))(u, z, r)
    expect = jax.grad(loss_agol, argnums=(0, 2))(u, z, r)
    # (the derivative with respect to the radius ratio goes through the construction
    # of the series, which loses a few more digits for the longest law: 1.3e-8)
    for a, b in zip(expect, calc, strict=True):
        a_, b_ = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(b_, a_, atol=5e-8 * np.max(np.abs(a_)), rtol=0)

    # with respect to the separations (forward mode)
    # (not within 1e-4 of the second contact, where the argument of the logarithm
    # loses its significant digits: the derivative there is poorly conditioned)
    tangent = jnp.where((z < 1.07) & (jnp.abs(z - 0.92) > 1e-4), 1.0, 0.0)
    _, jvp_calc = jax.jvp(lambda z: deficit(u, z, r), (z,), (tangent,))
    _, jvp_agol = jax.jvp(lambda z: agol_light_curve(u, z, r), (z,), (tangent,))
    np.testing.assert_allclose(
        jvp_calc, jvp_agol, atol=1e-8 * np.max(np.abs(jvp_agol)), rtol=0
    )


@needs_x64
def test_public_flag_and_threshold():
    """Through the public function, with enough limb samples for the rule"""
    u = jnp.asarray(LAWS["quartic"])
    r = 0.1
    t = np.linspace(0.0, 30.0, 800_000)
    phase = (t + 1.5) % 3.0 - 1.5
    z = np.sqrt((8 * np.sin(2 * np.pi * phase / 3.0)) ** 2 + 0.09)
    z = jnp.asarray(np.where(np.abs(phase) < 0.75, z, 5.0))
    on = pizzetti.light_curve(u, z, r)
    off = pizzetti.light_curve(u, z, r, contact=False)
    depth = central_depth(LAWS["quartic"], r)
    assert int(jnp.sum((z < 1 + r) & (z > 0.72))) >= pizzetti._contact_min_limb(4)
    np.testing.assert_allclose(on, off, atol=1e-10 * depth, rtol=0)
    assert np.max(np.abs(np.asarray(on) - np.asarray(off))) > 0


def test_min_limb_rule():
    assert pizzetti._contact_min_limb(4) == pizzetti.CONTACT_LIMB
    assert pizzetti._contact_min_limb(8) < pizzetti._contact_min_limb(4)
    assert pizzetti._contact_min_limb(21) >= 1
