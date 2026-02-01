# Weighting Methods Documentation

This module provides objective weight calculation methods for Multi-Criteria Decision Making (MCDM).

## Overview

Determining criterion weights is a critical step in MCDM as weights significantly impact the final rankings. This module implements objective (data-driven) weighting methods that derive weights from the decision matrix itself, eliminating subjective bias.

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

### 3. Ensemble Weights (`ensemble.py`)

**Combined Weighting Approach**

Combines multiple weighting methods for more robust results.

#### Aggregation Methods

**Arithmetic Mean:**
$$w_j^{ens} = \sum_{m=1}^{M} \alpha_m \cdot w_j^{(m)}$$

**Geometric Mean:**
$$w_j^{ens} = \left(\prod_{m=1}^{M} w_j^{(m)}\right)^{1/M}$$

**Harmonic Mean:**
$$w_j^{ens} = \frac{M}{\sum_{m=1}^{M} \frac{1}{w_j^{(m)}}}$$

#### When to Use
- When you want robust weights that are less sensitive to method choice
- When individual methods give conflicting results
- As a default choice when uncertain about which method to use

---

## Usage Examples

```python
import pandas as pd
from src.weighting import (
    EntropyWeightCalculator,
    CRITICWeightCalculator,
    EnsembleWeightCalculator,
    calculate_weights
)

# Sample decision matrix
data = pd.DataFrame({
    'Cost': [100, 150, 120, 180],
    'Quality': [0.8, 0.6, 0.9, 0.7],
    'Speed': [5, 3, 4, 2]
})

# Method 1: Entropy weights
entropy_calc = EntropyWeightCalculator()
entropy_result = entropy_calc.calculate(data)
print("Entropy weights:", entropy_result.weights)

# Method 2: CRITIC weights
critic_calc = CRITICWeightCalculator()
critic_result = critic_calc.calculate(data)
print("CRITIC weights:", critic_result.weights)

# Method 3: Ensemble weights (recommended)
ensemble_calc = EnsembleWeightCalculator(aggregation='geometric')
ensemble_result = ensemble_calc.calculate(data)
print("Ensemble weights:", ensemble_result.weights)

# Quick convenience function
weights = calculate_weights(data, method='ensemble')
```

---

## Comparison of Methods

| Aspect | Entropy | CRITIC | Ensemble |
|--------|---------|--------|----------|
| **Considers variation** | ✓ | ✓ | ✓ |
| **Considers correlation** | ✗ | ✓ | ✓ |
| **Robustness** | Medium | Medium | High |
| **Complexity** | Low | Medium | Medium |
| **Best for** | Information content | Unique information | General use |

---

## References

1. Shannon, C.E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.

2. Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining objective weights in multiple criteria problems: The CRITIC method. Computers & Operations Research.

3. Wang, Y.M., & Luo, Y. (2010). Integration of correlations with standard deviations for determining attribute weights in multiple attribute decision making. Mathematical and Computer Modelling.
