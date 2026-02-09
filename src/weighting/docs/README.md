# Weighting Methods Documentation

This module provides objective weight calculation methods for Multi-Criteria Decision Making (MCDM).

## Overview

Determining criterion weights is a critical step in MCDM as weights significantly impact the final rankings. This module implements objective (data-driven) weighting methods that derive weights from the decision matrix itself, eliminating subjective bias.

The three individual methods form a **complementary triad**:
- **Entropy** → univariate information (order 1)
- **CRITIC** → bivariate information (order 2, pairwise)
- **PCA** → multivariate information (order n, full covariance structure)

These are combined via an advanced ensemble with 6 strategies (including a deeply integrated hybrid).

## Methods

### 1. Entropy Weights (`entropy.py`)

**Based on Shannon's Information Theory**

The entropy method assigns higher weights to criteria with more variation across alternatives, as these criteria provide more information for distinguishing between options.

#### Mathematical Formulation

1. **Normalize to proportions:**
   $$p_{ij} = \frac{x_{ij}}{\sum_{i=1}^{m} x_{ij}}$$

2. **Calculate entropy:**
   $$E_j = -k \sum_{i=1}^{m} p_{ij} \ln(p_{ij})$$
   where $k = \frac{1}{\ln(m)}$

3. **Calculate divergence:**
   $$D_j = 1 - E_j$$

4. **Calculate weights:**
   $$w_j = \frac{D_j}{\sum_{k=1}^{n} D_k}$$

#### When to Use
- When you want criteria with more variation to have more influence
- When all alternatives have valid positive values
- When information content is the primary concern

---

### 2. CRITIC Weights (`critic.py`)

**Criteria Importance Through Inter-criteria Correlation**

CRITIC considers both the contrast intensity (standard deviation) and conflicting character (correlation) of criteria.

#### Mathematical Formulation

1. **Calculate standard deviation:**
   $$\sigma_j = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(x_{ij} - \bar{x}_j)^2}$$

2. **Calculate correlation conflict:**
   $$\text{Conflict}_j = \sum_{k=1}^{n}(1 - r_{jk})$$
   where $r_{jk}$ is the Pearson correlation between criteria j and k

3. **Calculate information content:**
   $$C_j = \sigma_j \times \text{Conflict}_j$$

4. **Calculate weights:**
   $$w_j = \frac{C_j}{\sum_{k=1}^{n} C_k}$$

#### When to Use
- When you want to consider inter-criteria relationships
- When correlated criteria should share importance
- When both variation and uniqueness matter

---

### 3. PCA Weights (`pca.py`)

**Principal Component Analysis-Based Multivariate Weighting**

PCA derives criterion weights from the eigenstructure of the standardized decision matrix, capturing each criterion's contribution to the overall multivariate data structure.

#### Mathematical Formulation

1. **Standardize the decision matrix:**
   $$z_{ij} = \frac{x_{ij} - \bar{x}_j}{\sigma_j}$$

2. **Eigendecompose the correlation matrix:**
   $$R \mathbf{v}_k = \lambda_k \mathbf{v}_k$$

3. **Retain K components** explaining ≥ 85% cumulative variance:
   $$\frac{\sum_{k=1}^{K} \lambda_k}{\sum_{k=1}^{n} \lambda_k} \geq 0.85$$

4. **Compute criterion weights (variance-weighted loadings):**
   $$w_j^* = \sum_{k=1}^{K} \frac{\lambda_k}{\sum_{l=1}^{K} \lambda_l} \cdot v_{jk}^2$$

5. **Normalize:**
   $$w_j = \frac{w_j^*}{\sum_{j=1}^{n} w_j^*}$$

#### Key Advantages
- Captures the **full multivariate structure** (not just pairwise correlations)
- Naturally handles **redundancy** — correlated criteria sharing a latent factor don't get over-weighted
- Provides **dimensionality reduction** perspective unique among the three methods

#### When to Use
- When criteria have complex correlation structures (clusters of correlated criteria)
- When you suspect redundancy among criteria
- When latent factor structure should influence weighting

---

### 4. Ensemble Weights (`ensemble.py`)

**Advanced Combined Weighting with 6 Strategies**

Combines individual weight vectors using configurable aggregation strategies, from simple statistical means to deeply integrated hybrid approaches.

#### Strategy A: Geometric Mean (Legacy)
$$w_j^{ens} = \frac{\left(\prod_{m=1}^{M} w_j^{(m)}\right)^{1/M}}{\sum_k \left(\prod_{m=1}^{M} w_k^{(m)}\right)^{1/M}}$$

Equivalent to the minimum Kullback-Leibler divergence solution for equal-confidence methods.

#### Strategy B: Game Theory (Min-Deviation Optimization)

Each method's contribution is weighted by entropy-based confidence:

$$\alpha_m = \frac{1 - H(\mathbf{w}^{(m)})}{\sum_l (1 - H(\mathbf{w}^{(l)}))}$$

where $H(\mathbf{w}) = -\sum_j w_j \ln(w_j) / \ln(n)$ is the normalized Shannon entropy.

$$w_j^{game} = \sum_{m=1}^{M} \alpha_m \cdot w_j^{(m)}$$

Methods producing more differentiated (lower entropy) weight vectors get higher influence.

#### Strategy C: Bayesian Bootstrap

1. Bootstrap-resample the decision matrix B times
2. Recompute each method's weights on each resample → estimate variance $\sigma_{mj}^2$
3. Inverse-variance weighted combination:
   $$w_j = \frac{\sum_m \frac{w_{mj}}{\sigma_{mj}^2}}{\sum_m \frac{1}{\sigma_{mj}^2}}$$

Methods producing stable weights across resamples get higher effective influence. Provides 95% confidence intervals.

#### Strategy D: Integrated Hybrid (Default — Recommended)

A deeply integrated three-stage approach where the methods structurally inform each other:

**Stage 1 — PCA Structural Analysis:**
- Run PCA to extract factor structure
- Compute PCA-residualized correlation matrix (remove top-K principal components)

**Stage 2 — Modified CRITIC with PCA-Informed Correlation:**
$$C_j^{hybrid} = \sigma_j \times \sum_{k} (1 - r_{jk}^{residual})$$

Uses PCA-residualized correlations instead of raw Pearson correlations, focusing the conflict measure on **unique information** not captured by dominant latent factors.

**Stage 3 — Entropy-Weighted Integration:**
$$\alpha_m = \frac{1 - H(\mathbf{w}^{(m)})}{\sum_l (1 - H(\mathbf{w}^{(l)}))}$$
$$w_j^{final} = \sum_m \alpha_m \cdot w_j^{(m)}$$

More decisive methods (lower entropy weight vectors) contribute more to the final weights.

#### When to Use
- **Integrated Hybrid**: Default choice — leverages all three methods synergistically
- **Game Theory**: When you want automatic confidence-based weighting without bootstrap cost
- **Bayesian Bootstrap**: When method stability/reliability is a primary concern
- **Geometric/Arithmetic/Harmonic**: For simplicity or backward compatibility

---

## Usage Examples

```python
import pandas as pd
from src.weighting import (
    EntropyWeightCalculator,
    CRITICWeightCalculator,
    PCAWeightCalculator,
    EnsembleWeightCalculator,
    calculate_weights
)

# Sample decision matrix
data = pd.DataFrame({
    'Cost': [100, 150, 120, 180, 90],
    'Quality': [0.8, 0.6, 0.9, 0.7, 0.5],
    'Speed': [5, 3, 4, 2, 6]
})

# Individual methods
entropy_result = EntropyWeightCalculator().calculate(data)
critic_result = CRITICWeightCalculator().calculate(data)
pca_result = PCAWeightCalculator(variance_threshold=0.85).calculate(data)

# Ensemble: integrated hybrid (default, recommended)
hybrid_result = EnsembleWeightCalculator().calculate(data)

# Ensemble: game theory
gt_result = EnsembleWeightCalculator(aggregation='game_theory').calculate(data)

# Ensemble: bayesian bootstrap
bb_result = EnsembleWeightCalculator(aggregation='bayesian_bootstrap').calculate(data)

# Ensemble: legacy geometric mean (entropy + CRITIC only)
legacy_result = EnsembleWeightCalculator(
    methods=['entropy', 'critic'], aggregation='geometric'
).calculate(data)

# Quick convenience function
weights = calculate_weights(data, method='pca')
```

---

## Comparison of Methods

| Aspect | Entropy | CRITIC | PCA | Ensemble (Hybrid) |
|--------|---------|--------|-----|-------------------|
| **Considers variation** | ✓ | ✓ | ✓ | ✓ |
| **Considers correlation** | ✗ | ✓ (pairwise) | ✓ (full structure) | ✓ |
| **Handles redundancy** | ✗ | Partially | ✓ | ✓ |
| **Information level** | Univariate | Bivariate | Multivariate | All levels |
| **Robustness** | Medium | Medium | Medium | High |
| **Complexity** | Low | Medium | Medium | High |
| **Best for** | Information content | Unique information | Latent structure | General use |

---

## Configuration

Configure via `WeightingConfig` in `src/config.py`:

```python
from src.config import Config

config = Config()
config.weighting.methods = ["entropy", "critic", "pca"]  # Individual methods
config.weighting.ensemble_strategy = "integrated_hybrid"   # Combination strategy
config.weighting.pca_variance_threshold = 0.85             # PCA component retention
config.weighting.bootstrap_samples = 500                   # For bayesian_bootstrap
```

Available ensemble strategies: `'geometric'`, `'arithmetic'`, `'harmonic'`, `'game_theory'`, `'bayesian_bootstrap'`, `'integrated_hybrid'`

---

## References

1. Shannon, C.E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.

2. Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining objective weights in multiple criteria problems: The CRITIC method. Computers & Operations Research.

3. Deng, H., Yeh, C.H., & Willis, R.J. (2000). Inter-company comparison using modified TOPSIS with objective weights. Computers & Operations Research, 27(10), 963-973.

4. Wang, Y.M., & Luo, Y. (2010). Integration of correlations with standard deviations for determining attribute weights in multiple attribute decision making. Mathematical and Computer Modelling.

5. Yan, H.B., & Ma, T. (2015). A game theory-based approach for combining multiple sets of weights. Expert Systems with Applications.

6. Zhu, Y. et al. (2020). Comprehensive evaluation of regional energy Internet using PCA-entropy-TOPSIS. Energy Reports.
