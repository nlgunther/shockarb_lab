"""
Subspace dislocation geometry — pure linear algebra, zero I/O.

Implements the factor-subspace operations required for multi-regime signal
combination and selection.  Every result is a function of projection operators
H Hᵀ, which are invariant to any rotation or sign flip of the basis within
the subspace: (HR)(HR)ᵀ = H Hᵀ for R ∈ O(k).  That invariance is the design
contract; see the rotation-invariance test in tests/test_dislocation_geom.py.

Design notes
------------
- All I/O (loading model JSON, writing scores) lives in pipeline.py.
- This module mirrors engine.py: takes arrays in, returns arrays out.
- The factor basis stored in model.json is Vt, shape (k × d).  Transposed to
  (d × k) and re-orthonormalised via QR decomposition on load so the projector
  identity holds regardless of serialisation precision.

References
----------
See docs/SUBSPACE_DISLOCATION.md for the full design and a verified k=3
worked example.  §5 of docs/Opus_KT.md records the design invariants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Basis extraction
# =============================================================================

def factor_basis(model) -> "NDArray[np.floating]":
    """
    Extract a d×k orthonormal factor basis H from a frozen FactorModel.

    The model persists ``_Vt`` (shape k×d, the top-k rows of the SVD Vᵀ
    of the ETF return matrix).  Transposing gives a d×k matrix whose columns
    span the factor subspace.  Re-orthonormalisation via QR is applied so the
    projector identity H Hᵀ holds exactly regardless of any floating-point
    drift introduced during serialisation.

    Parameters
    ----------
    model : FactorModel
        A fitted model, either live or loaded from JSON via pipeline.load_model.
        Must expose ``_Vt``, a (k × d) ndarray.

    Returns
    -------
    H : ndarray, shape (d, k)
        Orthonormal factor basis: H.T @ H == I_k to machine precision.
    """
    Vt = np.asarray(model._Vt)          # (k, d)
    stored = Vt.T                        # (d, k) — columns are factor directions
    H, _ = np.linalg.qr(stored)         # re-orthonormalise; H has shape (d, k)
    return H


# =============================================================================
# Projection operator
# =============================================================================

def projector(H: "NDArray[np.floating]") -> "NDArray[np.floating]":
    """
    Compute the orthogonal projection matrix Π_H = H Hᵀ onto col(H).

    Π_H is invariant to any orthogonal transformation applied to H within
    the subspace: projector(H @ R) == projector(H) for R ∈ O(k).

    Parameters
    ----------
    H : ndarray, shape (d, k)
        Factor basis.  Columns need not be orthonormal — if they are not,
        QR is applied internally so the returned projector is correct.

    Returns
    -------
    Pi : ndarray, shape (d, d)
        Symmetric, idempotent orthogonal projector onto col(H).
    """
    H_orth, _ = np.linalg.qr(H)         # guarantee orthonormality
    return H_orth @ H_orth.T


# =============================================================================
# Principal angles
# =============================================================================

def principal_angles(
    H_u: "NDArray[np.floating]",
    H_i: "NDArray[np.floating]",
) -> "NDArray[np.floating]":
    """
    Compute the principal angles (in radians) between two factor subspaces.

    The cosines of the principal angles are the singular values of H_uᵀ H_i.
    A singular value near 1 means the corresponding pair of directions is
    (nearly) shared; a singular value near 0 means the directions are
    orthogonal (regime-specific).

    Parameters
    ----------
    H_u : ndarray, shape (d, k_u)
        Orthonormal basis for the ukraine_shock subspace.
    H_i : ndarray, shape (d, k_i)
        Orthonormal basis for the iran_shock subspace.

    Returns
    -------
    angles : ndarray, shape (min(k_u, k_i),)
        Principal angles in radians, in ascending order (smallest = most
        aligned pair).

    Raises
    ------
    ValueError
        If H_u and H_i have different ambient dimensions (d must match).

    Notes
    -----
    Both bases are re-orthonormalised via QR before the cross-gram is formed
    so that the result is numerically correct even if the inputs carry
    floating-point imprecision.
    """
    H_u_orth, _ = np.linalg.qr(H_u)
    H_i_orth, _ = np.linalg.qr(H_i)

    d_u, k_u = H_u_orth.shape
    d_i, k_i = H_i_orth.shape

    if d_u != d_i:
        raise ValueError(
            f"Ambient dimension mismatch: H_u has d={d_u}, H_i has d={d_i}. "
            "Both bases must live in the same return space. "
            "Check that both models were built with the same ETF basket."
        )

    # Cross-Gram matrix: cosines are its singular values
    cross_gram = H_u_orth.T @ H_i_orth    # (k_u, k_i)
    sv = np.linalg.svd(cross_gram, compute_uv=False)

    # Clip to [-1, 1] before arccos to guard against floating-point overshoot
    cosines = np.clip(sv, -1.0, 1.0)

    # arccos is ill-conditioned near cosine = +/-1: d/dx arccos(x) = -1/sqrt(1-x^2)
    # diverges there, so ordinary SVD roundoff (~1e-16) on a truly-shared
    # direction (cosine == 1 in exact arithmetic) gets amplified to angles on
    # the order of sqrt(eps) ~ 1e-8 rad (~1e-6 deg) instead of exactly 0.
    # 1e-9 is comfortably above float64 SVD roundoff for these matrix sizes
    # but far below the smallest genuine angle this module cares about (the
    # live ukraine_shock/iran_shock subspaces differ by 15 deg at minimum —
    # see docs/KT.md session 16), so this cannot mask a real angle.
    _NEAR_PARALLEL_TOL = 1e-9
    cosines = np.where(np.abs(1.0 - cosines) < _NEAR_PARALLEL_TOL, 1.0, cosines)
    cosines = np.where(np.abs(-1.0 - cosines) < _NEAR_PARALLEL_TOL, -1.0, cosines)

    return np.arccos(cosines)
