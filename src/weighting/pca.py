# -*- coding: utf-8 -*-
"""
PCA Weight Calculator

Principal Component Analysis-based objective weight calculation method.
Derives criteria weights from the multivariate variance-covariance structure,
capturing each criterion's contribution to the overall data variability.

Mathematical Formula:
    w_j = Σ_k (λ_k / Σλ) × v_jk²
    
where:
    λ_k = eigenvalue of k-th principal component
    v_jk = loading of criterion j on component k
    
Only components explaining ≥ variance_threshold of cumulative variance are retained.

References:
    Deng, H., Yeh, C.H., & Willis, R.J. (2000). Inter-company comparison
    using modified TOPSIS with objective weights. Computers & Operations Research.
    
    Zhu, Y. et al. (2020). Comprehensive evaluation of regional energy Internet
    using PCA-entropy-TOPSIS. Energy Reports.
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .base import WeightResult


class PCAWeightCalculator:
    """
    PCA-based objective weight calculation for MCDM.
    
    Derives criterion weights from the eigenstructure of the standardized
    decision matrix. Criteria that load heavily on principal components
    explaining large proportions of variance receive higher weights.
    
    This captures the full multivariate variance-covariance structure,
    complementing Entropy (univariate) and CRITIC (bivariate) methods.
    
    Parameters
    ----------
    variance_threshold : float
        Cumulative variance ratio threshold for retaining principal
        components. Default 0.85 (retain components explaining ≥ 85%
        of total variance).
    epsilon : float
        Small constant for numerical stability.
    
    Attributes
    ----------
    variance_threshold : float
        Cumulative variance threshold
    epsilon : float
        Numerical stability constant
    
    Examples
    --------
    >>> import pandas as pd
    >>> from src.weighting import PCAWeightCalculator
    >>> 
    >>> data = pd.DataFrame({
    ...     'C1': [0.8, 0.6, 0.9, 0.7, 0.5],
    ...     'C2': [0.75, 0.55, 0.85, 0.65, 0.45],  # Correlated with C1
    ...     'C3': [0.3, 0.9, 0.1, 0.7, 0.4]         # Independent
    ... })
    >>> 
    >>> calc = PCAWeightCalculator(variance_threshold=0.85)
    >>> result = calc.calculate(data)
    >>> print(result.weights)
    
    Notes
    -----
    PCA weights naturally handle redundancy: if multiple criteria express
    the same latent factor, the weight is distributed across them rather
    than being multiplied. This avoids the over-weighting problem that
    can occur with Entropy when correlated criteria all show high variation.
    
    Requires m > n (more alternatives than criteria) for stable estimates.
    With the project's 64 provinces × 20 components, this is satisfied.
    
    References
    ----------
    Deng, H., Yeh, C.H., & Willis, R.J. (2000). Inter-company comparison
    using modified TOPSIS with objective weights. Computers & Operations
    Research, 27(10), 963-973.
    
    Li, X. et al. (2011). Integrated weight method for attribute based on
    PCA. Systems Engineering and Electronics.
    """
    
    def __init__(self, variance_threshold: float = 0.85, epsilon: float = 1e-10):
        self.variance_threshold = variance_threshold
        self.epsilon = epsilon
    
    def calculate(self, data: pd.DataFrame) -> WeightResult:
        """
        Calculate PCA-based criterion weights.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria).
            Each column is a criterion, each row an alternative.
        
        Returns
        -------
        WeightResult
            PCA-derived weights with eigenvalue and loading details.
        
        Raises
        ------
        ValueError
            If data has fewer than 2 rows or 2 columns.
        """
        m, n = data.shape
        if m < 2 or n < 2:
            raise ValueError(f"PCA requires at least 2 alternatives and 2 criteria, "
                           f"got {m} alternatives and {n} criteria.")
        
        columns = data.columns.tolist()
        
        # Step 1: Standardize (z-score normalization)
        scaler = StandardScaler()
        Z = scaler.fit_transform(data.values)
        
        # Step 2: Fit PCA with all components
        pca = PCA(n_components=min(m, n))
        pca.fit(Z)
        
        eigenvalues = pca.explained_variance_
        explained_ratio = pca.explained_variance_ratio_
        cumulative_ratio = np.cumsum(explained_ratio)
        
        # Step 3: Determine number of components to retain
        # At minimum retain 1 component, at most retain all
        n_retained = max(1, int(np.searchsorted(cumulative_ratio, 
                                                 self.variance_threshold) + 1))
        n_retained = min(n_retained, len(eigenvalues))
        
        # Step 4: Get loadings matrix (eigenvectors × sqrt(eigenvalues))
        # components_ shape: (n_components, n_features)
        components = pca.components_[:n_retained]  # (K, n)
        retained_eigenvalues = eigenvalues[:n_retained]
        retained_ratios = explained_ratio[:n_retained]
        
        # Step 5: Compute criterion weights
        # w_j* = Σ_k (λ_k / Σλ_retained) × v_jk²
        proportion = retained_eigenvalues / (retained_eigenvalues.sum() + self.epsilon)
        
        raw_weights = np.zeros(n)
        for k in range(n_retained):
            raw_weights += proportion[k] * (components[k, :] ** 2)
        
        # Clip and normalize to unit sum
        raw_weights = np.clip(raw_weights, self.epsilon, None)
        weights = raw_weights / raw_weights.sum()
        
        # Build loadings matrix for details
        loadings = np.zeros((n, n_retained))
        for k in range(n_retained):
            loadings[:, k] = components[k, :] * np.sqrt(retained_eigenvalues[k])
        
        loadings_dict = {}
        for j, col in enumerate(columns):
            loadings_dict[col] = {f"PC{k+1}": float(loadings[j, k]) 
                                  for k in range(n_retained)}
        
        return WeightResult(
            weights={col: float(weights[j]) for j, col in enumerate(columns)},
            method="pca",
            details={
                "eigenvalues": {f"PC{k+1}": float(eigenvalues[k]) 
                               for k in range(len(eigenvalues))},
                "explained_variance_ratio": {f"PC{k+1}": float(explained_ratio[k]) 
                                            for k in range(len(explained_ratio))},
                "cumulative_variance": {f"PC{k+1}": float(cumulative_ratio[k]) 
                                       for k in range(len(cumulative_ratio))},
                "n_components_retained": n_retained,
                "variance_threshold": self.variance_threshold,
                "total_variance_explained": float(cumulative_ratio[n_retained - 1]),
                "loadings": loadings_dict,
                "n_samples": m,
                "n_criteria": n
            }
        )
    
    def get_residual_correlation(self, data: pd.DataFrame, 
                                 n_components_remove: Optional[int] = None
                                 ) -> pd.DataFrame:
        """
        Compute correlation of PCA-residualized data.
        
        Removes the top-K principal components and returns the correlation
        matrix of the residuals. Used by the integrated hybrid ensemble
        to provide PCA-informed correlation to the CRITIC method.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
        n_components_remove : int, optional
            Number of top principal components to remove.
            If None, removes components explaining ≥ variance_threshold.
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix of PCA-residualized data.
        """
        m, n = data.shape
        columns = data.columns.tolist()
        
        # Standardize
        scaler = StandardScaler()
        Z = scaler.fit_transform(data.values)
        
        # Fit PCA
        pca = PCA(n_components=min(m, n))
        pca.fit(Z)
        
        # Determine how many components to remove
        if n_components_remove is None:
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            n_components_remove = max(1, int(np.searchsorted(
                cumulative, self.variance_threshold) + 1))
            n_components_remove = min(n_components_remove, n - 1)
        
        n_components_remove = min(n_components_remove, n - 1)
        
        # Project data onto top-K components, then subtract to get residuals
        # Z_approx = Z @ V_K^T @ V_K  (reconstruction from top-K PCs)
        V_K = pca.components_[:n_components_remove]  # (K, n)
        Z_approx = Z @ V_K.T @ V_K  # (m, n)
        Z_residual = Z - Z_approx  # (m, n)
        
        # Compute correlation of residuals
        residual_df = pd.DataFrame(Z_residual, columns=columns)
        
        # Handle near-zero variance columns in residuals
        std = residual_df.std()
        zero_std_cols = std[std < self.epsilon].index.tolist()
        if zero_std_cols:
            # Add tiny noise to avoid division by zero in correlation
            for col in zero_std_cols:
                residual_df[col] += np.random.normal(0, self.epsilon, size=m)
        
        corr_matrix = residual_df.corr()
        
        # Fill any NaN with 0 (uncorrelated assumption for degenerate cases)
        corr_matrix = corr_matrix.fillna(0.0)
        
        return corr_matrix
