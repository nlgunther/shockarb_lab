"""
Tests for shockarb.dislocation_geom — subspace geometry.

Covers:
  - Rotation invariance of the projector (the primary design contract)
  - Sign-flip invariance of the projector
  - Basic projector properties: symmetry, idempotency
  - principal_angles correctness against the k=3 worked example from
    docs/SUBSPACE_DISLOCATION.md §5
  - principal_angles ambient-dimension mismatch raises ValueError
  - factor_basis returns orthonormal columns (H.T @ H ≈ I_k)
"""

from __future__ import annotations

import numpy as np
import pytest

from shockarb.dislocation_geom import factor_basis, principal_angles, projector


# =============================================================================
# Helpers
# =============================================================================

def _random_orthonormal(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly random orthonormal (d × k) matrix."""
    A = rng.standard_normal((d, k))
    H, _ = np.linalg.qr(A)
    return H


def _random_orthogonal(k: int, rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly random orthogonal (k × k) matrix via QR."""
    A = rng.standard_normal((k, k))
    R, _ = np.linalg.qr(A)
    return R


# =============================================================================
# Rotation-invariance — the primary design contract
# =============================================================================

class TestRotationInvariance:
    """
    Formal guarantee that the projector is basis-free.

    Π(H) must equal Π(HR) for any R ∈ O(k), because ShockArb's signal
    (dislocation, r², fit) is derived from Π_H, and the in-subspace
    rotation that occurs on model refit must never affect the score.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    def test_rotation_invariance(self, seed: int) -> None:
        """‖Π(H) − Π(HR)‖_F < 1e-12 for a random R ∈ O(k)."""
        rng = np.random.default_rng(seed)
        d, k = 19, 3
        H = _random_orthonormal(d, k, rng)
        R = _random_orthogonal(k, rng)
        HR = H @ R

        Pi_H  = projector(H)
        Pi_HR = projector(HR)

        err = np.linalg.norm(Pi_H - Pi_HR, ord="fro")
        assert err < 1e-12, (
            f"Projector changed under rotation: ‖Π(H) − Π(HR)‖_F = {err:.2e}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 5])
    def test_sign_flip_invariance(self, seed: int) -> None:
        """‖Π(H) − Π(H diag(±1))‖_F < 1e-12 for random sign flips."""
        rng = np.random.default_rng(seed)
        d, k = 19, 3
        H = _random_orthonormal(d, k, rng)

        # Random sign-flip diagonal matrix: a special case of O(k)
        signs = rng.choice([-1.0, 1.0], size=k)
        H_flipped = H * signs[np.newaxis, :]   # broadcasts sign onto each column

        Pi_H = projector(H)
        Pi_F = projector(H_flipped)

        err = np.linalg.norm(Pi_H - Pi_F, ord="fro")
        assert err < 1e-12, (
            f"Projector changed under sign flip: ‖Π(H) − Π(H diag(±1))‖_F = {err:.2e}"
        )


# =============================================================================
# Projector properties
# =============================================================================

class TestProjectorProperties:
    """Projector must be symmetric and idempotent."""

    def test_symmetry(self) -> None:
        rng = np.random.default_rng(99)
        H = _random_orthonormal(10, 3, rng)
        Pi = projector(H)
        assert np.allclose(Pi, Pi.T, atol=1e-14), "Projector is not symmetric"

    def test_idempotency(self) -> None:
        rng = np.random.default_rng(99)
        H = _random_orthonormal(10, 3, rng)
        Pi = projector(H)
        assert np.allclose(Pi @ Pi, Pi, atol=1e-14), "Projector is not idempotent (Π² ≠ Π)"

    def test_rank(self) -> None:
        rng = np.random.default_rng(99)
        d, k = 10, 3
        H = _random_orthonormal(d, k, rng)
        Pi = projector(H)
        rank = np.linalg.matrix_rank(Pi, tol=1e-10)
        assert rank == k, f"Projector rank should be {k}, got {rank}"

    def test_non_orthonormal_input_still_correct(self) -> None:
        """projector() re-orthonormalises internally, so scaled columns are OK."""
        rng = np.random.default_rng(99)
        d, k = 8, 2
        H = _random_orthonormal(d, k, rng)
        # Scale columns — no longer orthonormal, but span is unchanged
        H_scaled = H * np.array([2.5, 0.4])
        Pi_orig   = projector(H)
        Pi_scaled = projector(H_scaled)
        assert np.allclose(Pi_orig, Pi_scaled, atol=1e-12), (
            "projector() should be insensitive to column scaling"
        )


# =============================================================================
# Principal angles — worked example (k=3, d=6 from design doc §5)
# =============================================================================

class TestPrincipalAngles:
    """
    Verify principal_angles against the exact k=3 worked example in
    docs/SUBSPACE_DISLOCATION.md §5.2.

    S_u = span{e1, e2, e3},  S_i = span{e1, e4, e5}.
    Exactly one shared direction (e1); the other two pairs are orthogonal.
    Expected cosines: (1, 0, 0) → angles: (0°, 90°, 90°).
    """

    def _bases(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct the exact bases from the design-doc worked example."""
        # Standard basis in R^6
        e = np.eye(6)
        H_u = e[:, [0, 1, 2]]   # (6 × 3): columns e1, e2, e3
        H_i = e[:, [0, 3, 4]]   # (6 × 3): columns e1, e4, e5
        return H_u, H_i

    def test_angles_match_worked_example(self) -> None:
        H_u, H_i = self._bases()
        angles = principal_angles(H_u, H_i)  # radians

        # Expected: one angle = 0, two angles = π/2
        expected = np.sort([0.0, np.pi / 2, np.pi / 2])
        angles_sorted = np.sort(angles)

        assert np.allclose(angles_sorted, expected, atol=1e-12), (
            f"Principal angles {np.degrees(angles_sorted)} deg, "
            f"expected {np.degrees(expected)} deg"
        )

    def test_cosines_match_worked_example(self) -> None:
        """Cosines of principal angles should be (1, 0, 0) as in §5.2."""
        H_u, H_i = self._bases()
        angles = principal_angles(H_u, H_i)
        cosines = np.sort(np.cos(angles))[::-1]  # descending

        expected_cosines = np.array([1.0, 0.0, 0.0])
        assert np.allclose(cosines, expected_cosines, atol=1e-12), (
            f"Cosines: {cosines}, expected: {expected_cosines}"
        )

    def test_identical_subspaces_give_zero_angles(self) -> None:
        """If both regimes have the same subspace, all angles are zero."""
        rng = np.random.default_rng(0)
        d, k = 12, 3
        H = _random_orthonormal(d, k, rng)
        # Rotate H to get a different basis for the same subspace
        R = _random_orthogonal(k, rng)
        angles = principal_angles(H, H @ R)
        assert np.allclose(angles, 0.0, atol=1e-12), (
            f"Identical subspace should yield zero angles; got {np.degrees(angles)} deg"
        )

    def test_orthogonal_subspaces_give_ninety_degree_angles(self) -> None:
        """Completely orthogonal subspaces should give angles of π/2."""
        e = np.eye(6)
        H_a = e[:, :3]    # span{e1, e2, e3}
        H_b = e[:, 3:]    # span{e4, e5, e6}
        angles = principal_angles(H_a, H_b)
        assert np.allclose(angles, np.pi / 2, atol=1e-12), (
            f"Orthogonal subspaces should yield 90° angles; got {np.degrees(angles)} deg"
        )

    def test_ambient_dimension_mismatch_raises(self) -> None:
        """Mismatched ambient dimensions must raise ValueError."""
        rng = np.random.default_rng(0)
        H_u = _random_orthonormal(10, 3, rng)
        H_i = _random_orthonormal(12, 3, rng)   # different d
        with pytest.raises(ValueError, match="Ambient dimension mismatch"):
            principal_angles(H_u, H_i)


# =============================================================================
# factor_basis — orthonormality of extracted basis
# =============================================================================

class TestFactorBasis:
    """factor_basis() must return a matrix with orthonormal columns."""

    def test_orthonormality(self) -> None:
        """H.T @ H should be I_k to machine precision after QR."""

        # Build a minimal mock FactorModel with a realistic _Vt
        class _MockModel:
            _Vt: np.ndarray

        rng = np.random.default_rng(7)
        k, d = 3, 19
        # Simulate a raw Vt from SVD: rows are unit vectors but not
        # necessarily exactly orthonormal due to float noise
        raw = rng.standard_normal((d, k))
        Q, _ = np.linalg.qr(raw)
        Vt_clean = Q.T                # (k × d) — orthonormal rows

        # Add small noise to simulate float drift
        noise = rng.standard_normal((k, d)) * 1e-10
        model = _MockModel()
        model._Vt = Vt_clean + noise

        H = factor_basis(model)
        gram = H.T @ H

        assert H.shape == (d, k), f"Expected shape ({d}, {k}), got {H.shape}"
        assert np.allclose(gram, np.eye(k), atol=1e-12), (
            f"H.T @ H deviates from I_k:\n{gram - np.eye(k)}"
        )
