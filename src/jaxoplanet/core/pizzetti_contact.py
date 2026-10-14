"""Contact expansions: the flux near and across the stellar limb as power series

This module is the second half of :mod:`jaxoplanet.core.pizzetti`. Write the
intensity as ``I = sum_a c_a (1 - r^2)^(a/2)`` (a sum of powers of ``mu``) and let
``P_g(z)`` be the integral of ``(1 - r^2)^g`` over the part of the stellar disk
blocked by an occultor of radius ``p`` at separation ``z`` (``p < z``). With

    k = [1 - (z - p)^2] / (4 z p),    kappa = 1 / k,

``k`` runs from 0 at first contact (``z = 1 + p``) to 1 at second contact
(``z = 1 - p``), and ``kappa`` from 1 at second contact to 0 inside the disk. Let
``m = g + 3/2``, ``y = 1 - k`` and ``y' = 1 - kappa``. Then, in three pieces,

* A, first contact, ``0 <= k <= 1/2``: ``P_g = k^m RA(k)``
* B, second contact from outside, ``y <= 1/2``:
  ``P_g = RB(y) + y^m [ln(y) LB(y) + CB(y)]``
* C, second contact from inside, ``y' <= y'_max``:
  ``P_g = RC(y') + y'^m [ln(y') LC(y') + CC(y')]``

where all the capital letters are power series, and ``LB = LC = 0`` unless ``g`` is
a half-integer (an odd power of ``mu``). The series follow from the
hypergeometric form of the solution of Mandel & Agol (2002): on each piece
``z`` is an algebraic function of the argument, so the elementary factors are power
series in it, and the universal functions of the exponent ``g`` enter through
coefficient tables that do not depend on ``p`` or ``z``. The coefficients of the
series, which depend on ``p`` and the law, are built once per light curve with
exact power-series arithmetic.

Accuracy: the series are truncated at degree ``NS`` and ``JMAX`` terms of the
expansion of ``1/s`` are summed; the pieces end at ``k = 1/2`` and ``y = 1/2`` (A, B)
and at ``y' = Y_MAX`` (C), where the truncation errors are largest. The truncation
error is *not* bounded rigorously (unlike that of the interior series). Instead,
:func:`probe_points` gives separations at the ends and in the middle of every piece,
and :mod:`jaxoplanet.core.pizzetti` uses the series only if they agree with the
solution of Agol et al. at these points to a tenth of ``series_tol`` (relative to
the central depth), for the particular radius ratio and law of the light curve. In
float64 the series agree with the solution of Agol et al. to ~1e-15 of the flux
(3e-13 for the longest laws) over the radius ratios checked in the tests (0.01-0.2).

The coefficient tables span ~38 orders of magnitude (the partial sums cancel), so
the series are for float64 only.
"""

from __future__ import annotations

__all__ = ["build", "evaluate", "probe_points", "table"]

from functools import cache
from math import pi
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import digamma, gammaln

from jaxoplanet.types import Array

#: Degree of the contact series (number of coefficients is ``NS + 1``)
NS = 32

#: Number of terms ``j`` of the expansion of ``1/s`` kept in the tables
JC = 200

#: Number of terms ``j`` used. The terms decrease like ``x^j`` with ``x = 1 - (z - p)^2``
#: at the lower end of the piece C, so this is enough for radius ratios up to ~0.17
JMAX = 128

#: Largest ``y' = 1 - kappa`` of the piece C (the truncation error grows with it)
Y_MAX = 0.4

#: Argument below which the Taylor series of the hypergeometric function is used
#: (the connection formula is used above)
XS = 0.5


# ------------------------------------------------------------------ static tables
def _lgs(x: float) -> tuple[float, float]:
    """(log |Gamma(x)|, sign Gamma(x)); the sign is 0 at the poles"""
    if x <= 0.0 and x == np.floor(x):
        return 0.0, 0.0
    sg = 1.0
    if x < 0.0 and int(np.floor(-x)) % 2 == 0:
        sg = -1.0
    return float(gammaln(x)), sg


def _gratio(a1: float, a2: float, b1: float, b2: float) -> float:
    """Gamma(a1) Gamma(a2) / (Gamma(b1) Gamma(b2)) without overflow"""
    l1, s1 = _lgs(a1)
    l2, s2 = _lgs(a2)
    l3, s3 = _lgs(b1)
    l4, s4 = _lgs(b2)
    if s3 == 0.0 or s4 == 0.0:
        return 0.0
    return s1 * s2 * s3 * s4 * float(np.exp(l1 + l2 - l3 - l4))


def _taylor(a: float, b: float, c: float, n: int) -> np.ndarray:
    """Taylor coefficients of the Gauss function 2F1(a, b; c; x)"""
    out = np.empty(n)
    t = 1.0
    for i in range(n):
        out[i] = t
        t *= (i + a) * (i + b) / ((i + c) * (i + 1.0))
    return out


def _connection(
    a: float, b: float, c: float, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Series ``A, Bl, Bc``: ``F = A(y) + y^m (ln y Bl(y) + Bc(y))``, ``y = 1 - x``"""
    m = c - a - b
    mi = np.floor(m + 0.5)
    A = np.zeros(n)
    Bl = np.zeros(n)
    Bc = np.zeros(n)
    if abs(m - mi) > 1e-9:
        A = _taylor(a, b, 1.0 - m, n) * _gratio(c, m, c - a, c - b)
        Bc = _taylor(c - a, c - b, 1.0 + m, n) * _gratio(c, -m, a, b)
        return A, Bl, Bc
    mm = int(mi)
    t = _gratio(mi, c, a + mi, b + mi)
    for i in range(min(mm, n)):
        A[i] = t
        if i + 1 < mm:
            t *= (i + a) * (i + b) / ((i + 1.0) * (i + 1.0 - mi))
    l1, s1 = _lgs(c)
    l3, s3 = _lgs(a)
    l4, s4 = _lgs(b)
    if s3 == 0.0 or s4 == 0.0:
        return A, Bl, Bc
    pre = -((-1.0) ** mm) * s1 * s3 * s4 * np.exp(l1 - l3 - l4 - gammaln(mi + 1.0))
    i = np.arange(n, dtype=float)
    ratio = (a + mi + i) * (b + mi + i) / ((i + 1.0) * (i + mi + 1.0))
    t = np.concatenate([[1.0], np.cumprod(ratio[:-1])])
    psi = (
        -digamma(i + 1.0)
        - digamma(i + mi + 1.0)
        + digamma(a + i + mi)
        + digamma(b + i + mi)
    )
    Bl = pre * t
    Bc = pre * t * psi
    return A, Bl, Bc


@cache
def table(g: float, jmax: int = JC, n: int = NS + 1) -> np.ndarray:
    """Coefficient table for the exponent ``g``, shape ``(2, jmax + 1, 4, n)``

    Axis 0 is the family (0: limb crossing, scaled by ``B(1/2, g + 2 + j)``;
    1: interior, scaled by ``pi``) and axis 1 the term ``j`` of the expansion of
    ``1/s``. Axis 2 holds the Taylor series in the argument (slot 0) and the series
    ``A, Bl, Bc`` of the connection form (slots 1-3).
    """
    T = np.zeros((2, jmax + 1, 4, n))
    for j in range(jmax + 1):
        for fam in range(2):
            if fam == 0:
                a, b, c = 0.5, 0.5, g + 2.5 + j
                s = np.exp(gammaln(0.5) + gammaln(g + 2.0 + j) - gammaln(g + 2.5 + j))
            else:
                a, b, c = -(g + 1.0 + j), 0.5, 1.0
                s = pi
            T[fam, j, 0] = _taylor(a, b, c, n)
            T[fam, j, 1:] = _connection(a, b, c, n)
            T[fam, j] *= s
    T.setflags(write=False)
    return T


# --------------------------------------------------------------- series arithmetic
# A truncated power series is an array of ``NS + 1`` coefficients along the last axis.
_I = np.arange(NS + 1)
_LOWER = _I[:, None] - _I[None, :]  # [i, k] = i - k
_TRI = _LOWER >= 0
_LOWER_CLIP = np.clip(_LOWER, 0, None)


def _smul(a: Array, b: Array) -> Array:
    """Product of truncated power series (broadcasting over leading axes)"""
    bt = jnp.where(_TRI, b[..., _LOWER_CLIP], 0.0)  # [..., i, k] = b[i - k]
    return jnp.einsum("...k,...ik->...i", a, bt)


def _sinv(a: Array) -> Array:
    """Reciprocal of a power series with a nonzero constant term"""
    toep = jnp.where(_TRI, a[_LOWER_CLIP], 0.0)
    e0 = jnp.zeros_like(a).at[0].set(1.0)
    return jax.scipy.linalg.solve_triangular(toep, e0, lower=True)


def _spow(a: Array, alpha: Array) -> Array:
    """``a ** alpha`` for power series with ``a[..., 0] > 0`` (Miller's recurrence)

    ``a`` has shape ``(*batch, n)`` and ``alpha`` shape ``(*batch, m)``; the result
    has shape ``(*batch, m, n)``.
    """
    alpha = jnp.asarray(alpha, dtype=a.dtype)
    n = a.shape[-1]
    ks = jnp.arange(1, n)
    a0 = a[..., 0, None]  # (*batch, 1)
    ak = a[..., None, 1:]  # (*batch, 1, n - 1)

    def step(f: Array, i: Array) -> tuple[Array, None]:
        idx = i - ks
        fk = jnp.take(f, jnp.clip(idx, 0, None), axis=-1)  # (*batch, m, n - 1)
        coef = ((alpha[..., None] + 1.0) * ks - i) * ak
        s = jnp.sum(jnp.where(idx >= 0, coef * fk, 0.0), axis=-1)
        return f.at[..., i].set(s / (i * a0)), None

    f = jnp.zeros((*alpha.shape, n), dtype=a.dtype)
    f = f.at[..., 0].set(a0**alpha)
    f, _ = jax.lax.scan(step, f, jnp.arange(1, n))
    return f


def _binomial(alpha: float, dtype: np.dtype) -> Array:
    """The series of ``(1 - u) ** alpha``"""
    c = np.ones(NS + 1)
    for i in range(1, NS + 1):
        c[i] = c[i - 1] * (i - 1 - alpha) / i
    return jnp.asarray(c, dtype=dtype)


# ------------------------------------------------------------------- the build
def _tables(gs: tuple[float, ...], jmax: int) -> np.ndarray:
    return np.stack([table(float(g), JC)[:, : jmax + 1] for g in gs])


def _shift_matrix(c: Array) -> Array:
    """``out[j, k] = c[j, k - j]`` (zero for ``k < j``): the series times ``u^j``"""
    jj = np.arange(c.shape[0])[:, None]
    kk = np.arange(NS + 1)[None, :]
    idx = np.clip(kk - jj, 0, None)
    return jnp.where(kk >= jj, jnp.take_along_axis(c, jnp.asarray(idx), axis=1), 0.0)


def _sum_conv(cj: Array, tab: Array) -> Array:
    """``sum_j cj[j] * tab[i, j]`` (power-series product), for every exponent ``i``"""
    m = jnp.einsum("jk,ijl->ikl", cj, tab)
    sel = (_I[:, None, None] + _I[None, :, None]) == _I[None, None, :]
    return jnp.einsum("ikl,kln->in", m, jnp.asarray(sel, dtype=m.dtype))


class Contact(NamedTuple):
    """Power series of the three pieces, summed over the terms of the law"""

    zh: Array  # boundary between A and B (k = 1/2)
    z_lo: Array  # lower end of the piece C
    a: Array  # (nt, n): k^m * RA(k)
    reg: Array  # (2, n): regular parts of B and C
    sing: Array  # (2, nt, 2, n): log and plain parts of B and C
    m: Array  # (nt,) exponents g + 3/2


def build(gs: tuple[float, ...], cs: Array, p: Array, z_c: Array, jmax: int) -> Contact:
    """Series of the contact pieces for ``I = sum_i cs[i] (1 - r^2)^gs[i]``

    ``gs`` are the (static) exponents, ``p`` the radius ratio, ``z_c`` the lower limit
    of the region where the contact pieces are wanted (the cut of the interior series)
    and ``jmax`` the number of terms of the expansion of ``1/s``.
    """
    dtype = cs.dtype
    g = np.asarray(gs, dtype=float)
    nt = g.shape[0]
    tab = jnp.asarray(_tables(gs, jmax), dtype=dtype)  # (nt, 2, jmax + 1, 4, n)
    one = jnp.zeros(NS + 1, dtype=dtype).at[0].set(1.0)
    u = jnp.zeros(NS + 1, dtype=dtype).at[1].set(1.0)
    p2 = p * p
    jj = np.arange(jmax + 1.0)

    # z as a series in the argument of each piece (A: k, B: y = 1 - k, C: y' = 1 - kappa)
    a1 = jnp.stack([one - 2.0 * u, 2.0 * u - one, one + u])
    kap = one - u
    rad = p2 * jax.vmap(_smul)(a1, a1) + (1.0 - p2) * jnp.stack(
        [one, one, _smul(kap, kap)]
    )
    root = _spow(rad, jnp.full((3, 1), 0.5, dtype=dtype))[:, 0]
    z_a = p * a1[0] + root[0]
    z_b = p * a1[1] + root[1]
    z_c_ = _smul(root[2] - p * a1[2], _sinv(kap))
    k_b = one - u
    x_b = 4.0 * p * _smul(z_b, k_b)
    x_c = 4.0 * p * _smul(z_c_, _sinv(kap))
    w_a = 4.0 * p * z_a

    # all the powers in one recurrence: A: z^(g+1), w^j; B, C: x^(g+1), x^j
    m_alpha = max(nt, NS + 1, nt + jmax + 1)

    def pad(v: np.ndarray) -> np.ndarray:
        return np.pad(v, (0, m_alpha - v.shape[0]))

    alphas = jnp.asarray(
        np.stack(
            [
                pad(np.concatenate([g + 1.0])),
                pad(np.arange(NS + 1.0)),
                pad(np.concatenate([g + 1.0, jj])),
                pad(np.concatenate([g + 1.0, jj])),
            ]
        ),
        dtype=dtype,
    )
    bases = jnp.stack([z_a, w_a, x_b, x_c])
    bases = bases.at[1].set(w_a)
    pw = _spow(bases, alphas)  # (4, m_alpha, n)
    z_pow, w_j, xb_pow, xc_pow = pw[0, :nt], pw[1, : NS + 1], pw[2], pw[3]

    # ---- piece A
    q = p2 * one - _smul(z_a, z_a)
    cj = jnp.concatenate([(one + q)[None], _smul(q, w_j[1:])], axis=0)
    cj = cj.at[1:].set(_shift_matrix(cj)[1:])
    r_a = _sum_conv(cj, tab[:, 0, : NS + 1, 0])
    pa = -((4.0 * p) ** (g + 1.0) / (2.0 * (g + 1.0)))[:, None] * z_pow
    sa = cs[:, None] * _smul(pa, r_a)

    # ---- pieces B and C
    def piece(fam: int, z: Array, x: Array, pwx: Array) -> Array:
        if fam == 0:
            pre = -_smul(pwx[:nt], _binomial(0.5, dtype)) / (2.0 * (g + 1.0))[:, None]
        else:
            pre = -pwx[:nt] / (2.0 * (g + 1.0))[:, None]
        q = p2 * one - _smul(z, z)
        xj = pwx[nt : nt + jmax + 1]
        cj = jnp.concatenate([(one + q)[None], _smul(q, xj[1:])], axis=0)
        sj = _shift_matrix(cj)
        r = _sum_conv(cj, tab[:, fam, :, 1])
        sl = _sum_conv(sj, tab[:, fam, :, 2])
        sc = _sum_conv(sj, tab[:, fam, :, 3])
        return jnp.stack([_smul(pre, r), _smul(pre, sl), _smul(pre, sc)])  # (3, nt, n)

    b = piece(0, z_b, x_b, xb_pow)
    c = piece(1, z_c_, x_c, xc_pow)
    reg = jnp.stack([jnp.einsum("i,in->n", cs, b[0]), jnp.einsum("i,in->n", cs, c[0])])
    sing = jnp.stack(
        [
            cs[:, None, None] * jnp.moveaxis(b[1:], 0, 1),
            cs[:, None, None] * jnp.moveaxis(c[1:], 0, 1),
        ]
    )
    zh = jnp.sqrt((1.0 - p) * (1.0 + p))
    # lower end of the piece C: where y' = Y_MAX, and at least the lower limit asked for
    y = Y_MAX
    z_y = (-p * (1 + y) + jnp.sqrt(p2 * (1 + y) ** 2 + (1 - y) ** 2 * (1 - p2))) / (
        1 - y
    )
    z_lo = jax.lax.stop_gradient(jnp.maximum(jnp.maximum(z_c, z_y), p + 1e-12))
    return Contact(zh, z_lo, sa, reg, sing, jnp.asarray(g + 1.5, dtype=dtype))


def _powers(x: Array) -> Array:
    """``x[..., None] ** arange(n)`` by cumulative products"""
    pw = jnp.concatenate(
        [jnp.ones_like(x)[..., None], jnp.broadcast_to(x[..., None], (*x.shape, NS))],
        axis=-1,
    )
    return jnp.cumprod(pw, axis=-1)


def _mpowers(x: Array, nt: int) -> Array:
    """``x ** (3/2 + i/2)`` for ``i = 0, ..., nt - 1`` (the exponents ``g + 3/2``, with
    ``g = i / 2`` an odd or even power of ``mu``), by cumulative products"""
    sx = jnp.sqrt(x)
    first = x * sx
    ratio = jnp.broadcast_to(sx[..., None], (*x.shape, nt - 1))
    return jnp.cumprod(jnp.concatenate([first[..., None], ratio], axis=-1), axis=-1)


def evaluate(contact: Contact, p: Array, z: Array) -> Array:
    """Blocked flux ``sum_i cs[i] P_{i/2}(z)`` for ``z`` in ``[z_lo, 1 + p)``

    The result is only meaningful in that range.
    """
    zh, z_lo, sa, reg, sing, _ = contact
    nt = sa.shape[0]
    ones = jnp.ones_like(z)
    zs = jnp.maximum(z, z_lo)

    # A: first contact
    k = (1.0 - (zs - p)) * (1.0 + (zs - p)) / (4.0 * zs * p)
    in_a = z >= zh
    ka = jnp.where(in_a, jnp.clip(k, 1e-300, 0.5), 0.5 * ones)
    va = jnp.sum(_mpowers(ka, nt) * (_powers(ka) @ sa.T), axis=-1)

    # B: second contact, from outside
    in_b = (z >= 1.0 - p) & ~in_a
    yb = jnp.where(
        in_b,
        jnp.clip(((zs + p) - 1.0) * ((zs + p) + 1.0) / (4.0 * zs * p), 1e-300, 0.5),
        0.25 * ones,
    )
    pb = _powers(yb)
    vb = pb @ reg[0] + jnp.sum(
        _mpowers(yb, nt)
        * (jnp.log(yb)[..., None] * (pb @ sing[0, :, 0].T) + pb @ sing[0, :, 1].T),
        axis=-1,
    )

    # C: second contact, from inside
    in_c = ~in_a & ~in_b
    zc_ = jnp.minimum(zs, 1.0 - p)
    yc = jnp.where(
        in_c,
        jnp.clip(
            (1.0 - (zc_ + p))
            * (1.0 + (zc_ + p))
            / ((1.0 - (zc_ - p)) * (1.0 + (zc_ - p))),
            1e-300,
            1.0,
        ),
        0.25 * ones,
    )
    pc = _powers(yc)
    vc = pc @ reg[1] + jnp.sum(
        _mpowers(yc, nt)
        * (jnp.log(yc)[..., None] * (pc @ sing[1, :, 0].T) + pc @ sing[1, :, 1].T),
        axis=-1,
    )

    return jnp.where(in_a, va, jnp.where(in_b, vb, vc))


def probe_points(p: Array, contact: Contact) -> Array:
    """Separations at which the series are most likely to be inaccurate: both ends of
    every piece and the middle of every piece"""
    z_lo, zh = contact.z_lo, contact.zh
    d = 1e-3 * p
    lo, hi = z_lo, 1.0 - p
    return jnp.stack(
        [
            z_lo,
            0.5 * (lo + hi),
            hi - d,
            hi + d,
            0.5 * (hi + zh),
            zh - d,
            zh + d,
            0.5 * (zh + 1.0 + p),
            1.0 + p - 10 * d,
        ]
    )
