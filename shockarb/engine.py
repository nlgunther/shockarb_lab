"""
Core factor model engine - pure mathematical operations, zero I/O.

This module implements a two-stage structural decomposition:
  1. Extract orthogonal macro factors from ETF returns using SVD
  2. Project individual stocks onto those factors via OLS regression

The output is per-stock factor loadings (betas) describing how much of each
stock's variance is explained by macro factors. When scored against live data,
stocks deviating from factor-implied returns are flagged as mispriced.

Mathematical Foundation
-----------------------
Stage 1 (Factor Extraction):
    Let R_E be the (T × N_etf) matrix of mean-centered daily ETF returns.
    SVD: R_E = U Σ Vᵀ
    We retain top k rows of Vᵀ as the factor basis directions.
    Factor returns: F = R_E @ Vᵀ[:k]ᵀ  →  (T × k) matrix

Stage 2 (Stock Projection):
    Each stock's returns R_S are regressed on F via OLS.
    Loadings B solve: min ||R_S - F @ Bᵀ||

Scoring:
    Given new ETF returns, compute factor scores and expected stock returns.
    Delta = expected - actual. Positive delta = stock oversold = potential buy.

Example
-------
    from shockarb.engine import FactorModel
    
    model = FactorModel(etf_returns, stock_returns)
    model.fit(n_components=3)
    
    # Score today's tape
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
# Diagnostics Container
# =============================================================================

@dataclass
class FactorDiagnostics:
    """
    Diagnostics and confidence metrics from the factor model fit.
    
    These metrics are essential for interpreting model output. A mispricing
    signal from a stock with low R² should be heavily discounted.
    
    Attributes
    ----------
    n_observations : int
        Trading days in calibration window.
    
    n_etfs : int
        Number of ETFs in the factor basis.
    
    n_stocks : int
        Number of stocks projected onto the basis.
    
    n_factors : int
        Number of SVD components retained.
    
    explained_variance_ratio : ndarray
        Fraction of ETF variance captured by each factor.
        Cumulative should be >70% for a well-specified model.
    
    cumulative_variance : float
        Sum of explained_variance_ratio.
    
    stock_r_squared : Series
        Per-stock R². Low R² (<0.30) means the model poorly explains that
        stock's movements - its mispricing delta should not be trusted.
    
    residual_vol : Series
        Per-stock annualized volatility of unexplained returns.
    """
    n_observations: int
    n_etfs: int
    n_stocks: int
    n_factors: int
    explained_variance_ratio: NDArray[np.floating]
    cumulative_variance: float
    stock_r_squared: pd.Series
    residual_vol: pd.Series
    
    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"FactorModel Diagnostics",
            f"  Window: {self.n_observations} days",
            f"  ETFs: {self.n_etfs}, Stocks: {self.n_stocks}, Factors: {self.n_factors}",
            f"  Cumulative Variance Explained: {self.cumulative_variance:.1%}",
            f"  Per-factor variance: {[f'{v:.1%}' for v in self.explained_variance_ratio]}",
            f"  Stock R² range: [{self.stock_r_squared.min():.2f}, {self.stock_r_squared.max():.2f}]",
        ]
        return "\n".join(lines)


# =============================================================================
# Factor Model
# =============================================================================

class FactorModel:
    """
    SVD-based factor model for crisis mispricing detection.
    
    This class is stateless until fit() is called. After fitting, it can:
      - Report factor loadings and diagnostics
      - Score new returns against the fitted model
      - Project out-of-sample securities onto the existing basis
    
    Parameters
    ----------
    etf_returns : DataFrame
        (T × N_etf) daily returns for factor basis ETFs. Index must be DatetimeIndex.
    
    stock_returns : DataFrame
        (T × N_stock) daily returns for target stocks. Must share index with etf_returns.
    
    Raises
    ------
    ValueError
        If DataFrames have mismatched indices.
    
    Example
    -------
        model = FactorModel(etf_returns, stock_returns)
        model.fit(n_components=3)
        
        # Access fitted state
        print(model.diagnostics.summary())
        print(model.etf_basis)  # Factor loadings for ETFs
        print(model.loadings)   # Factor loadings for stocks
    """
    
    def __init__(self, etf_returns: pd.DataFrame, stock_returns: pd.DataFrame):
        if not etf_returns.index.equals(stock_returns.index):
            raise ValueError(
                "Index mismatch: etf_returns and stock_returns must have identical DatetimeIndex. "
                "Align temporal indices before constructing the model."
            )
        
        self.etf_returns = etf_returns
        self.stock_returns = stock_returns
        
        # Internal state (populated by fit())
        self._Vt: Optional[NDArray] = None           # (k × N_etf) factor directions
        self._F: Optional[pd.DataFrame] = None       # (T × k) factor return series
        self._etf_mean: Optional[pd.Series] = None   # ETF calibration means
        self._stock_mean: Optional[pd.Series] = None # Stock calibration means
        self.loadings: Optional[pd.DataFrame] = None # (N_stock × k) factor betas
        self.diagnostics: Optional[FactorDiagnostics] = None
        self._fitted = False

    def _require_fitted(self) -> None:
        """Guard against operations on unfitted model."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call model.fit() first.")

    # -------------------------------------------------------------------------
    # Fitting
    # -------------------------------------------------------------------------

    def fit(self, n_components: int = 3) -> FactorModel:
        """
        Fit the factor model: extract factors from ETFs, project stocks.
        
        Parameters
        ----------
        n_components : int
            Number of SVD factors to retain. Typical values:
              - 2: Market + one sector rotation axis
              - 3: Market + sector + shock (recommended)
              - 4+: Risks overfitting with short calibration windows
        
        Returns
        -------
        self : FactorModel
            Returns self for method chaining.
        
        Notes
        -----
        Factor interpretation (typical for geopolitical crises):
          - Factor 1: Broad market direction (risk-on/off)
          - Factor 2: Energy vs. everything else (geopolitical shock axis)
          - Factor 3: Often growth-vs-value or sector rotation noise
        """
        T, N_etf = self.etf_returns.shape
        N_stock = self.stock_returns.shape[1]
        
        if n_components >= min(T, N_etf):
            raise ValueError(
                f"n_components={n_components} must be < min(T={T}, N_etf={N_etf})"
            )
        
        logger.info(f"Fitting {n_components}-factor model: {T} days × {N_etf} ETFs → {N_stock} stocks")
        
        # Store means for scoring
        self._etf_mean = self.etf_returns.mean()
        self._stock_mean = self.stock_returns.mean()
        
        # ---------- Stage 1: Factor Extraction via SVD ----------
        # Mean-center to ensure Factor 1 captures variance, not drift
        R_E = self.etf_returns.values - self._etf_mean.values
        U, Sigma, Vt = np.linalg.svd(R_E, full_matrices=False)
        
        # Retain top k factor directions
        self._Vt = Vt[:n_components]
        
        # Compute factor returns: F = R_E @ Vᵀ[:k]ᵀ
        F = R_E @ self._Vt.T
        self._F = pd.DataFrame(
            F, 
            index=self.etf_returns.index,
            columns=[f"Factor_{i+1}" for i in range(n_components)]
        )
        
        # Explained variance
        total_var = (Sigma ** 2).sum()
        var_ratio = (Sigma[:n_components] ** 2) / total_var
        
        # ---------- Stage 2: Stock Projection via OLS ----------
        R_S = self.stock_returns.values - self._stock_mean.values
        
        # lstsq is more numerically stable than explicit pseudoinverse
        B_T, residuals, rank, s = np.linalg.lstsq(F, R_S, rcond=None)
        
        self.loadings = pd.DataFrame(
            B_T.T,
            index=self.stock_returns.columns,
            columns=[f"Factor_{i+1}" for i in range(n_components)]
        )
        
        # ---------- Stage 3: Compute Diagnostics ----------
        R_S_hat = F @ B_T
        resid = R_S - R_S_hat
        
        ss_res = (resid ** 2).sum(axis=0)
        ss_tot = (R_S ** 2).sum(axis=0)
        
        # R² with safe division
        r_squared = np.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, 0.0)
        resid_vol = resid.std(axis=0) * np.sqrt(252)  # Annualized
        
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
                "Consider increasing n_components or checking ETF selection."
            )
        
        return self

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def etf_basis(self) -> pd.DataFrame:
        """
        Factor basis directions (Vᵀ transposed for readability).
        
        Returns DataFrame with ETFs as rows, Factors as columns. High absolute
        values indicate heavy weighting of that ETF within that factor.
        """
        self._require_fitted()
        return pd.DataFrame(
            self._Vt.T,
            index=self.etf_returns.columns,
            columns=[f"Factor_{i+1}" for i in range(self._Vt.shape[0])]
        )

    @property
    def factor_returns(self) -> pd.DataFrame:
        """Historical factor return series (T × k)."""
        self._require_fitted()
        return self._F

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def score(
        self, 
        today_etf_returns: pd.Series, 
        today_stock_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Score today's returns against the factor model.
        
        Computes expected returns from factors and flags deviations as mispricing.
        
        Parameters
        ----------
        today_etf_returns : Series
            Today's returns for basis ETFs. Index = ticker symbols.
        
        today_stock_returns : Series
            Today's returns for target stocks. Index = ticker symbols.
        
        Returns
        -------
        DataFrame
            Sorted by confidence_delta (descending). Columns:
              - actual_return: What happened
              - expected_return: Factor-implied return (zero-mean)
              - delta: expected - actual. Positive = oversold = potential buy
              - r_squared: Historical model fit quality for this stock
              - residual_vol: Typical unexplained daily volatility (annualized)
              - confidence_delta: delta × r_squared (primary signal)
        
        Notes
        -----
        The expected return is computed WITHOUT adding back the calibration mean.
        This compares factor-implied moves to actual moves, which is the
        economically meaningful comparison during a crisis.
        
        Example
        -------
            scores = model.score(today_etfs, today_stocks)
            
            # Top 5 mispriced stocks with R² > 0.4
            actionable = scores[scores["r_squared"] > 0.4].head(5)
        """
        self._require_fitted()
        
        # Align ETF returns to model's expected tickers, fill missing with 0
        r_e = today_etf_returns.reindex(self._etf_mean.index).fillna(0)
        
        # Map to factor space (deviation from calibration mean)
        f_today = self._Vt @ (r_e.values - self._etf_mean.values)
        
        # Factor-implied stock returns (zero-mean basis)
        expected = self.loadings.values @ f_today
        
        # Align stock returns
        common = self.loadings.index.intersection(today_stock_returns.index)
        expected_s = pd.Series(expected, index=self.loadings.index).loc[common]
        actual_s = today_stock_returns.loc[common]
        
        result = pd.DataFrame({
            "actual_return": actual_s,
            "expected_return": expected_s,
            "delta": expected_s - actual_s,
            "r_squared": self.diagnostics.stock_r_squared.loc[common],
            "residual_vol": self.diagnostics.residual_vol.loc[common],
        })
        
        # Primary signal: confidence-weighted delta
        result["confidence_delta"] = result["delta"] * result["r_squared"]
        
        return result.sort_values("confidence_delta", ascending=False)

    # -------------------------------------------------------------------------
    # Out-of-sample projection
    # -------------------------------------------------------------------------

    def project_security(
        self, 
        ticker: str, 
        returns: pd.Series, 
        min_overlap: float = 0.8
    ) -> pd.Series:
        """
        Project an out-of-sample security onto the existing factor basis.
        
        Useful for quickly evaluating a new ticker without refitting the entire model.
        
        Parameters
        ----------
        ticker : str
            Name/symbol for the new security.
        
        returns : Series
            Daily returns with DatetimeIndex overlapping calibration window.
        
        min_overlap : float
            Minimum fraction of calibration dates that must have data.
            Default 0.8 (80%). Prevents garbage projections from thin data.
        
        Returns
        -------
        Series
            Factor loadings for the new security.
        
        Raises
        ------
        ValueError
            If overlap is insufficient.
        
        Example
        -------
            import yfinance as yf
            prices = yf.download("SHOP", start="2022-02-10", end="2022-03-31")
            returns = prices["Adj Close"].pct_change().dropna()
            loadings = model.project_security("SHOP", returns)
        """
        self._require_fitted()
        
        aligned = returns.reindex(self.etf_returns.index)
        coverage = aligned.notna().mean()
        
        if coverage < min_overlap:
            raise ValueError(
                f"{ticker} has only {coverage:.0%} overlap with calibration window. "
                f"Minimum required: {min_overlap:.0%}"
            )
        
        mask = aligned.notna()
        r = (aligned[mask] - aligned[mask].mean()).values
        F_masked = self._F.loc[mask].values
        
        loadings, *_ = np.linalg.lstsq(F_masked, r, rcond=None)
        
        return pd.Series(
            loadings, 
            index=[f"Factor_{i+1}" for i in range(len(loadings))],
            name=ticker
        )

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialize model state to a dictionary (JSON-compatible).
        
        Used by Pipeline.save_model(). Preserves all state needed to reconstruct
        the model without refitting.
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
            "etf_index": [d.isoformat() for d in self.etf_returns.index],
            "etf_returns": self.etf_returns.values.tolist(),
            "etf_mean": self._etf_mean.tolist(),
            "factor_returns": self._F.values.tolist(),
            "loadings": self.loadings.values.tolist(),
            "stock_columns": list(self.stock_returns.columns),
            "stock_index": [d.isoformat() for d in self.stock_returns.index],
            "stock_returns": self.stock_returns.values.tolist(),
            "stock_mean": self._stock_mean.tolist(),
            "explained_variance_ratio": self.diagnostics.explained_variance_ratio.tolist(),
            "stock_r_squared": self.diagnostics.stock_r_squared.to_dict(),
            "residual_vol": self.diagnostics.residual_vol.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FactorModel:
        """
        Reconstruct a fitted model from a dictionary.
        
        Used by Pipeline.load_model().
        """
        etf_index = pd.to_datetime(d["etf_index"])
        stock_index = pd.to_datetime(d["stock_index"])
        
        etf_returns = pd.DataFrame(
            d["etf_returns"], index=etf_index, columns=d["etf_columns"]
        )
        stock_returns = pd.DataFrame(
            d["stock_returns"], index=stock_index, columns=d["stock_columns"]
        )
        
        model = cls(etf_returns, stock_returns)
        
        n_factors = d["metadata"]["n_factors"]
        model._Vt = np.array(d["Vt"])
        model._F = pd.DataFrame(
            d["factor_returns"], 
            index=etf_index,
            columns=[f"Factor_{i+1}" for i in range(n_factors)]
        )
        model._etf_mean = pd.Series(d["etf_mean"], index=d["etf_columns"])
        model._stock_mean = pd.Series(d["stock_mean"], index=d["stock_columns"])
        model.loadings = pd.DataFrame(
            d["loadings"],
            index=d["stock_columns"],
            columns=[f"Factor_{i+1}" for i in range(n_factors)]
        )
        model.diagnostics = FactorDiagnostics(
            n_observations=d["metadata"]["n_observations"],
            n_etfs=len(d["etf_columns"]),
            n_stocks=len(d["stock_columns"]),
            n_factors=n_factors,
            explained_variance_ratio=np.array(d["explained_variance_ratio"]),
            cumulative_variance=d["metadata"]["cumulative_variance"],
            stock_r_squared=pd.Series(d["stock_r_squared"]),
            residual_vol=pd.Series(d["residual_vol"]),
        )
        model._fitted = True
        
        return model
