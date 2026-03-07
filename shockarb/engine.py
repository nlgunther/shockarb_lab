"""
Core factor model engine — pure mathematics, zero I/O.

Two-stage structural decomposition:

  Stage 1 — Factor extraction (ETF SVD)
    Let R_E be the (T × N_etf) matrix of mean-centred daily ETF returns.
    SVD: R_E = U Σ Vᵀ
    Retain top k rows of Vᵀ as factor directions.
    Factor return series: F = R_E @ Vᵀ[:k]ᵀ  →  (T × k)

  Stage 2 — Stock projection (OLS)
    Each stock's mean-centred returns R_S are regressed on F via OLS.
    Loadings B solve: min ‖R_S − F @ Bᵀ‖

  Scoring
    Given new ETF returns, compute factor scores and expected stock returns.
    delta = expected − actual.  Positive delta → stock undersold → buy signal.
    confidence_delta = delta_rel × R²  (down-weights low-fit stocks).

This module takes DataFrames in and returns DataFrames out.
All file and network interaction lives in pipeline.py.

Example
-------
    from shockarb.engine import FactorModel

    model = FactorModel(etf_returns, stock_returns)
    model.fit(n_components=3)

    scores = model.score(today_etfs, today_stocks)
    print(scores.nlargest(10, "confidence_delta"))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# FactorDiagnostics
# =============================================================================

@dataclass
class FactorDiagnostics:
    """
    Quality metrics from a completed model fit.

    A mispricing signal from a stock with low R² should be discounted
    heavily; the factor model cannot explain that stock's movements and
    the delta may reflect sector-specific noise rather than mispricing.

    Attributes
    ----------
    n_observations : int
        Trading days in the calibration window.
    n_etfs : int
        Number of ETFs in the factor basis.
    n_stocks : int
        Number of target stocks projected onto the basis.
    n_factors : int
        Number of SVD components retained.
    explained_variance_ratio : ndarray, shape (n_factors,)
        Fraction of total ETF variance captured by each factor.
        Cumulative sum should exceed 0.70 for a well-specified model.
    cumulative_variance : float
        Sum of explained_variance_ratio.
    stock_r_squared : Series, index = stock tickers
        Per-stock OLS R².  Values below 0.30 indicate the factor basis
        poorly explains that stock — treat its delta with caution.
    residual_vol : Series, index = stock tickers
        Per-stock annualised volatility of unexplained returns (√252 × σ).
    """
    n_observations: int
    n_etfs: int
    n_stocks: int
    n_factors: int
    explained_variance_ratio: "NDArray[np.floating]"
    cumulative_variance: float
    stock_r_squared: pd.Series
    residual_vol: pd.Series

    def summary(self) -> str:
        """Return a compact human-readable summary string."""
        return "\n".join([
            "FactorModel Diagnostics",
            f"  Window:    {self.n_observations} days",
            f"  ETFs:      {self.n_etfs}  |  Stocks: {self.n_stocks}  |  Factors: {self.n_factors}",
            f"  Cumulative variance explained: {self.cumulative_variance:.1%}",
            f"  Per-factor: {[f'{v:.1%}' for v in self.explained_variance_ratio]}",
            f"  Stock R² range: [{self.stock_r_squared.min():.2f}, {self.stock_r_squared.max():.2f}]",
        ])


# =============================================================================
# FactorModel
# =============================================================================

class FactorModel:
    """
    SVD-based factor model for geopolitical crisis mispricing detection.

    Stateless until fit() is called.  After fitting it can score new returns,
    project out-of-sample securities, and serialise/deserialise to JSON.

    Parameters
    ----------
    etf_returns : DataFrame, shape (T, N_etf)
        Daily returns for the macro factor ETFs.  Index must be a DatetimeIndex.
    stock_returns : DataFrame, shape (T, N_stock)
        Daily returns for target stocks.  Must share the same index.

    Raises
    ------
    ValueError
        If the two DataFrames have mismatched indices.

    Example
    -------
        model = FactorModel(etf_returns, stock_returns)
        model.fit(n_components=3)

        print(model.diagnostics.summary())
        scores = model.score(today_etf_returns, today_stock_returns)
    """

    def __init__(self, etf_returns: pd.DataFrame, stock_returns: pd.DataFrame):
        if not etf_returns.index.equals(stock_returns.index):
            raise ValueError(
                "Index mismatch: etf_returns and stock_returns must share an identical "
                "DatetimeIndex. Intersect the indices before constructing the model."
            )
        self.etf_returns = etf_returns
        self.stock_returns = stock_returns

        # Set by fit(); all None until then
        self._Vt: Optional[NDArray] = None          # (k × N_etf) factor directions
        self._F: Optional[pd.DataFrame] = None      # (T × k) factor return series
        self._etf_mean: Optional[pd.Series] = None  # per-ETF calibration mean
        self._stock_mean: Optional[pd.Series] = None  # per-stock calibration mean
        self.loadings: Optional[pd.DataFrame] = None   # (N_stock × k) OLS betas
        self.diagnostics: Optional[FactorDiagnostics] = None
        self._fitted = False

    def _require_fitted(self) -> None:
        """Raise if called before fit()."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call model.fit() first.")

    # -------------------------------------------------------------------------
    # Fitting
    # -------------------------------------------------------------------------

    def fit(self, n_components: int = 3) -> "FactorModel":
        """
        Fit the model: extract macro factors from ETFs, project stocks onto them.

        Parameters
        ----------
        n_components : int
            SVD factors to retain.  Must be < min(T, N_etf).
            Recommended: 3 for geopolitical events (market / energy-shock /
            sector-rotation axes).

        Returns
        -------
        self
            For method chaining: ``model.fit(3).score(...)``.
        """
        T, N_etf = self.etf_returns.shape
        N_stock = self.stock_returns.shape[1]

        if n_components >= min(T, N_etf):
            raise ValueError(
                f"n_components={n_components} must be < min(T={T}, N_etf={N_etf})"
            )

        logger.info(
            f"Fitting {n_components}-factor model: "
            f"{T} days × {N_etf} ETFs → {N_stock} stocks"
        )

        # Store calibration means for use in score()
        self._etf_mean = self.etf_returns.mean()
        self._stock_mean = self.stock_returns.mean()

        # ------ Stage 1: Factor extraction via SVD ------
        # Mean-centre so Factor 1 captures covariance, not drift
        R_E = self.etf_returns.values - self._etf_mean.values
        _, Sigma, Vt = np.linalg.svd(R_E, full_matrices=False)

        self._Vt = Vt[:n_components]               # (k × N_etf)
        F = R_E @ self._Vt.T                       # (T × k) factor returns
        self._F = pd.DataFrame(
            F,
            index=self.etf_returns.index,
            columns=[f"Factor_{i+1}" for i in range(n_components)],
        )

        # Explained variance from singular values
        var_ratio = (Sigma[:n_components] ** 2) / (Sigma ** 2).sum()

        # ------ Stage 2: Stock projection via OLS ------
        R_S = self.stock_returns.values - self._stock_mean.values
        # lstsq is numerically more stable than explicit pseudoinverse
        B_T, *_ = np.linalg.lstsq(F, R_S, rcond=None)

        self.loadings = pd.DataFrame(
            B_T.T,
            index=self.stock_returns.columns,
            columns=[f"Factor_{i+1}" for i in range(n_components)],
        )

        # ------ Stage 3: Diagnostics ------
        R_S_hat = F @ B_T
        resid = R_S - R_S_hat
        ss_res = (resid ** 2).sum(axis=0)
        ss_tot = (R_S ** 2).sum(axis=0)

        r_squared = np.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, 0.0)
        resid_vol = resid.std(axis=0) * np.sqrt(252)

        self.diagnostics = FactorDiagnostics(
            n_observations=T,
            n_etfs=N_etf,
            n_stocks=N_stock,
            n_factors=n_components,
            explained_variance_ratio=var_ratio,
            cumulative_variance=float(var_ratio.sum()),
            stock_r_squared=pd.Series(r_squared, index=self.stock_returns.columns),
            residual_vol=pd.Series(resid_vol, index=self.stock_returns.columns),
        )

        self._fitted = True
        logger.success(f"Model fitted. {var_ratio.sum():.1%} ETF variance explained.")

        if var_ratio.sum() < 0.70:
            logger.warning(
                f"Cumulative variance ({var_ratio.sum():.1%}) is below 70%. "
                "Consider increasing n_components or broadening ETF selection."
            )

        return self

    # -------------------------------------------------------------------------
    # Properties (read-only access to fitted state)
    # -------------------------------------------------------------------------

    @property
    def etf_basis(self) -> pd.DataFrame:
        """
        Factor basis directions as a (N_etf × k) DataFrame.

        Vᵀ transposed so rows are ETFs and columns are factors — easier to read
        than the raw (k × N_etf) matrix from SVD.  High absolute values indicate
        that ETF is heavily weighted in that factor.
        """
        self._require_fitted()
        return pd.DataFrame(
            self._Vt.T,
            index=self.etf_returns.columns,
            columns=[f"Factor_{i+1}" for i in range(self._Vt.shape[0])],
        )

    @property
    def factor_returns(self) -> pd.DataFrame:
        """Historical factor return series, shape (T × k)."""
        self._require_fitted()
        return self._F

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def score(
        self,
        today_etf_returns: pd.Series,
        today_stock_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Score today's tape against the fitted factor model.

        Parameters
        ----------
        today_etf_returns : Series
            Live ETF returns, indexed by ticker.  Missing tickers are filled
            with zero (treated as no movement in that factor direction).
        today_stock_returns : Series
            Live stock returns, indexed by ticker.  Only tickers present in
            both this Series and the fitted loadings are scored.

        Returns
        -------
        DataFrame
            One row per scored stock, sorted descending by confidence_delta.
            Columns:

            actual_return   — observed return today
            expected_rel    — factor-implied return (relative, no drift)
            expected_abs    — factor-implied return + calibration daily drift
            delta_rel       — expected_rel − actual (pure mispricing signal)
            delta_abs       — expected_abs − actual (drift-adjusted signal)
            r_squared       — from calibration fit; weight for confidence
            residual_vol    — annualised unexplained vol (risk sizing input)
            confidence_delta — delta_rel × r_squared (primary ranking signal)

        Notes
        -----
        Use delta_rel for ranking (ignores calibration-window drift).
        Use delta_abs for position sizing when drift is meaningful.
        Threshold guidance: confidence_delta > +0.005 with r_squared > 0.50
        is the conservative actionable threshold.
        """
        self._require_fitted()

        # Project today's ETF returns into factor space using stored means
        r_e = today_etf_returns.reindex(self._etf_mean.index).fillna(0) - self._etf_mean
        f_today = self._Vt @ r_e.values                     # (k,)

        # Factor-implied returns (relative: no drift; absolute: + daily drift)
        expected_rel = self.loadings.values @ f_today        # (N_stock,)
        expected_abs = expected_rel + self._stock_mean.values

        # Restrict to tickers present in both loadings and today's tape
        common = self.loadings.index.intersection(today_stock_returns.index)

        idx = self.loadings.index.get_indexer(common)
        result = pd.DataFrame(
            {
                "actual_return":  today_stock_returns.loc[common].values,
                "expected_rel":   expected_rel[idx],
                "expected_abs":   expected_abs[idx],
                "delta_rel":      expected_rel[idx] - today_stock_returns.loc[common].values,
                "delta_abs":      expected_abs[idx] - today_stock_returns.loc[common].values,
                "r_squared":      self.diagnostics.stock_r_squared.loc[common].values,
                "residual_vol":   self.diagnostics.residual_vol.loc[common].values,
            },
            index=common,
        )
        result["confidence_delta"] = result["delta_rel"] * result["r_squared"]

        return result.sort_values("confidence_delta", ascending=False)

    # -------------------------------------------------------------------------
    # Out-of-sample projection
    # -------------------------------------------------------------------------

    def project_security(
        self,
        ticker: str,
        returns: pd.Series,
        min_overlap: float = 0.8,
    ) -> pd.Series:
        """
        Project a new ticker onto the existing factor basis without refitting.

        Useful for quickly evaluating whether a ticker not in the original
        universe belongs in the next calibration run.

        Parameters
        ----------
        ticker : str
            Display name for the security.
        returns : Series
            Daily returns with a DatetimeIndex overlapping the calibration window.
        min_overlap : float
            Minimum fraction of calibration dates that must have data (default 0.8).

        Returns
        -------
        Series
            Factor loadings (length k) for the new security.

        Raises
        ------
        ValueError
            If data coverage is below min_overlap.

        Example
        -------
            prices = yf.download("SHOP", start="2022-02-10", end="2022-03-31")
            rets = prices["Adj Close"].pct_change().dropna()
            loadings = model.project_security("SHOP", rets)
        """
        self._require_fitted()

        aligned = returns.reindex(self.etf_returns.index)
        coverage = aligned.notna().mean()

        if coverage < min_overlap:
            raise ValueError(
                f"{ticker} has only {coverage:.0%} overlap with the calibration window "
                f"(minimum required: {min_overlap:.0%})."
            )

        mask = aligned.notna()
        r = (aligned[mask] - aligned[mask].mean()).values
        loadings, *_ = np.linalg.lstsq(self._F.loc[mask].values, r, rcond=None)

        return pd.Series(
            loadings,
            index=[f"Factor_{i+1}" for i in range(len(loadings))],
            name=ticker,
        )

    # -------------------------------------------------------------------------
    # Serialisation
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise fitted model state to a JSON-compatible dict.

        Only the data required to reconstruct scoring is stored:
        _Vt, loadings, _etf_mean, _stock_mean, diagnostics, and the
        column names needed to rebuild index labels.

        The raw return matrices (etf_returns, stock_returns) are NOT stored —
        they add megabytes to every file and are not needed for score().
        If you need them for project_security() after a load, re-run build().

        JSON is preferred over pickle for readability, cross-version
        stability, and security (pickle executes arbitrary code on load).
        """
        self._require_fitted()
        return {
            "metadata": {
                "n_factors": self.diagnostics.n_factors,
                "n_observations": self.diagnostics.n_observations,
                "cumulative_variance": self.diagnostics.cumulative_variance,
            },
            "Vt": self._Vt.tolist(),
            "etf_columns": list(self.etf_returns.columns),
            "etf_mean": self._etf_mean.tolist(),
            "loadings": self.loadings.values.tolist(),
            "stock_columns": list(self.stock_returns.columns),
            "stock_mean": self._stock_mean.tolist(),
            "explained_variance_ratio": self.diagnostics.explained_variance_ratio.tolist(),
            "stock_r_squared": self.diagnostics.stock_r_squared.to_dict(),
            "residual_vol": self.diagnostics.residual_vol.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FactorModel":
        """
        Reconstruct a scored-ready model from a dict produced by to_dict().

        The returned model can call score() and project_security() immediately.
        etf_returns and stock_returns are set to empty stubs (zero rows) since
        the raw calibration data is not persisted.  If you need those matrices
        (e.g. to refit), call Pipeline.build() instead.
        """
        n_factors = d["metadata"]["n_factors"]
        etf_cols = d["etf_columns"]
        stock_cols = d["stock_columns"]

        # Stubs satisfy the constructor's index-equality check without storing data
        empty_etf = pd.DataFrame(columns=etf_cols)
        empty_stock = pd.DataFrame(columns=stock_cols)

        model = cls(empty_etf, empty_stock)

        model._Vt = np.array(d["Vt"])
        model._etf_mean = pd.Series(d["etf_mean"], index=etf_cols)
        model._stock_mean = pd.Series(d["stock_mean"], index=stock_cols)
        model._F = pd.DataFrame(columns=[f"Factor_{i+1}" for i in range(n_factors)])

        model.loadings = pd.DataFrame(
            d["loadings"],
            index=stock_cols,
            columns=[f"Factor_{i+1}" for i in range(n_factors)],
        )
        model.diagnostics = FactorDiagnostics(
            n_observations=d["metadata"]["n_observations"],
            n_etfs=len(etf_cols),
            n_stocks=len(stock_cols),
            n_factors=n_factors,
            explained_variance_ratio=np.array(d["explained_variance_ratio"]),
            cumulative_variance=d["metadata"]["cumulative_variance"],
            stock_r_squared=pd.Series(d["stock_r_squared"]),
            residual_vol=pd.Series(d["residual_vol"]),
        )
        model._fitted = True
        return model
