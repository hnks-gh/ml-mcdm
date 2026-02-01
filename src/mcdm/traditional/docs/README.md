# Traditional MCDM Methods Documentation

This module provides crisp (non-fuzzy) Multi-Criteria Decision Making methods for ranking alternatives.

## Overview

Traditional MCDM methods operate on precise numerical values and provide deterministic rankings. Each method has unique characteristics making it suitable for different decision scenarios.

## Methods

### 1. TOPSIS (`topsis.py`)

**Technique for Order Preference by Similarity to Ideal Solution**

TOPSIS ranks alternatives by measuring their geometric distance from ideal and anti-ideal solutions.

#### Mathematical Formulation

1. **Normalize** (vector normalization):
   $$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{m} x_{ij}^2}}$$

2. **Weight** the normalized matrix:
   $$v_{ij} = w_j \times r_{ij}$$

3. **Identify ideal solutions**:
   - $A^+ = \{v_1^+, ..., v_n^+\}$ (best values)
   - $A^- = \{v_1^-, ..., v_n^-\}$ (worst values)

4. **Calculate distances**:
   $$D_i^+ = \sqrt{\sum_{j=1}^{n}(v_{ij} - v_j^+)^2}$$
   $$D_i^- = \sqrt{\sum_{j=1}^{n}(v_{ij} - v_j^-)^2}$$

5. **Closeness coefficient**:
   $$C_i = \frac{D_i^-}{D_i^+ + D_i^-}$$

#### When to Use
- When you need distance-based ranking
- When both benefit and cost criteria exist
- When you want intuitive "closer to ideal" interpretation

---

### 2. VIKOR (`vikor.py`)

**Multi-criteria Optimization and Compromise Solution**

VIKOR focuses on finding a compromise solution that provides maximum group utility while minimizing individual regret.

#### Mathematical Formulation

1. **Determine ideal values**:
   - $f_j^* = \max_i f_{ij}$ (best)
   - $f_j^- = \min_i f_{ij}$ (worst)

2. **Calculate utility measures**:
   $$S_i = \sum_{j=1}^{n} w_j \frac{f_j^* - f_{ij}}{f_j^* - f_j^-}$$ (group utility)
   $$R_i = \max_j \left[ w_j \frac{f_j^* - f_{ij}}{f_j^* - f_j^-} \right]$$ (individual regret)

3. **Compromise measure**:
   $$Q_i = v \frac{S_i - S^*}{S^- - S^*} + (1-v) \frac{R_i - R^*}{R^- - R^*}$$
   where $v$ controls the group utility weight (typically 0.5)

#### When to Use
- When compromise between group utility and individual regret matters
- When checking solution stability is important
- When conflicting criteria need balanced consideration

---

### 3. PROMETHEE (`promethee.py`)

**Preference Ranking Organization Method for Enrichment Evaluations**

PROMETHEE uses pairwise comparisons with preference functions to model decision maker's preferences.

#### Preference Functions

| Type | Name | Formula |
|------|------|---------|
| I | Usual | $P(d) = 0$ if $d \leq 0$, else $1$ |
| II | U-shape | $P(d) = 0$ if $d \leq q$, else $1$ |
| III | V-shape | $P(d) = d/p$ if $0 < d < p$, else $1$ |
| IV | Level | $P(d) = 0.5$ if $q < d \leq p$ |
| V | V-shape+I | Linear between $q$ and $p$ |
| VI | Gaussian | $P(d) = 1 - e^{-d^2/(2\sigma^2)}$ |

#### Mathematical Formulation

1. **Preference degree** for criterion $j$:
   $$P_j(a, b) = F_j(d_j(a, b))$$

2. **Aggregated preference index**:
   $$\pi(a, b) = \sum_{j=1}^{n} w_j P_j(a, b)$$

3. **Outranking flows**:
   $$\Phi^+(a) = \frac{1}{m-1} \sum_{b} \pi(a, b)$$ (leaving flow)
   $$\Phi^-(a) = \frac{1}{m-1} \sum_{b} \pi(b, a)$$ (entering flow)

4. **Net flow** (PROMETHEE II):
   $$\Phi(a) = \Phi^+(a) - \Phi^-(a)$$

#### When to Use
- When modeling gradual preferences is important
- When incomparability between alternatives should be detected
- When criterion-specific preference modeling is needed

---

### 4. COPRAS (`copras.py`)

**Complex Proportional Assessment**

COPRAS assumes direct and proportional dependence of utility on criterion values and weights.

#### Mathematical Formulation

1. **Sum normalization**:
   $$r_{ij} = \frac{x_{ij}}{\sum_{i=1}^{m} x_{ij}}$$

2. **Weighted normalized values**:
   $$d_{ij} = r_{ij} \times w_j$$

3. **Sums for each alternative**:
   $$S_i^+ = \sum_{j \in J_{max}} d_{ij}$$ (benefit criteria)
   $$S_i^- = \sum_{j \in J_{min}} d_{ij}$$ (cost criteria)

4. **Relative significance**:
   $$Q_i = S_i^+ + \frac{S_{min}^- \times \sum S_i^-}{S_i^- \times \sum (1/S_i^-)}$$

5. **Utility degree**:
   $$N_i = \frac{Q_i}{Q_{max}} \times 100\%$$

#### When to Use
- When dealing with both maximizing and minimizing criteria
- When proportional assessment is appropriate
- When you want percentage-based utility interpretation

---

### 5. EDAS (`edas.py`)

**Evaluation based on Distance from Average Solution**

EDAS uses the average solution as reference instead of ideal solutions, making it more robust to outliers.

#### Mathematical Formulation

1. **Average solution**:
   $$AV_j = \frac{1}{m} \sum_{i=1}^{m} x_{ij}$$

2. **Distance measures** (for benefit criteria):
   $$PDA_{ij} = \frac{\max(0, x_{ij} - AV_j)}{AV_j}$$
   $$NDA_{ij} = \frac{\max(0, AV_j - x_{ij})}{AV_j}$$

3. **Weighted sums**:
   $$SP_i = \sum_{j=1}^{n} w_j \times PDA_{ij}$$
   $$SN_i = \sum_{j=1}^{n} w_j \times NDA_{ij}$$

4. **Appraisal score**:
   $$AS_i = \frac{1}{2}\left(\frac{SP_i}{SP_{max}} + 1 - \frac{SN_i}{SN_{max}}\right)$$

#### When to Use
- When outliers might distort ideal-based methods
- When average performance is a meaningful benchmark
- When you want robust rankings

---

## Method Comparison

| Method | Reference Point | Aggregation | Handles Incomparability | Complexity |
|--------|----------------|-------------|------------------------|------------|
| TOPSIS | Ideal/Anti-ideal | Distance | No | Low |
| VIKOR | Best/Worst | Utility+Regret | Via conditions | Medium |
| PROMETHEE | Pairwise | Preference flows | Yes (PROMETHEE I) | High |
| COPRAS | None | Proportional | No | Low |
| EDAS | Average | Distance | No | Low |

---

## Usage Examples

```python
import pandas as pd
from src.mcdm.traditional import (
    TOPSISCalculator,
    VIKORCalculator,
    PROMETHEECalculator,
    COPRASCalculator,
    EDASCalculator
)

# Sample decision matrix
data = pd.DataFrame({
    'Quality': [0.8, 0.6, 0.9, 0.7, 0.5],
    'Price': [100, 150, 120, 80, 200],    # Cost criterion
    'Speed': [5, 3, 4, 6, 2],
    'Reliability': [0.9, 0.7, 0.85, 0.8, 0.6]
}, index=['A', 'B', 'C', 'D', 'E'])

weights = {'Quality': 0.3, 'Price': 0.25, 'Speed': 0.25, 'Reliability': 0.2}

# TOPSIS
topsis = TOPSISCalculator(cost_criteria=['Price'])
topsis_result = topsis.calculate(data, weights)
print("TOPSIS Ranking:", topsis_result.ranks.sort_values())

# VIKOR
vikor = VIKORCalculator(v=0.5, cost_criteria=['Price'])
vikor_result = vikor.calculate(data, weights)
print("VIKOR Compromise:", vikor_result.compromise_solution)

# PROMETHEE
promethee = PROMETHEECalculator(preference_function='vshape', cost_criteria=['Price'])
promethee_result = promethee.calculate(data, weights)
print("PROMETHEE Net Flow:", promethee_result.phi_net.sort_values(ascending=False))

# COPRAS
copras = COPRASCalculator(cost_criteria=['Price'])
copras_result = copras.calculate(data, weights)
print("COPRAS Utility:", copras_result.utility_degree)

# EDAS
edas = EDASCalculator(cost_criteria=['Price'])
edas_result = edas.calculate(data, weights)
print("EDAS Appraisal Score:", edas_result.AS.sort_values(ascending=False))
```

---

## Dynamic/Multi-Period Variants

Each method has a multi-period variant for panel data:
- `DynamicTOPSIS`: Incorporates trajectory and stability analysis
- `MultiPeriodVIKOR`: Temporal aggregation of Q scores
- `MultiPeriodPROMETHEE`: Flow evolution over time
- `MultiPeriodCOPRAS`: Utility degree trends
- `MultiPeriodEDAS`: Distance from evolving averages

---

## References

1. Hwang, C.L., & Yoon, K. (1981). Multiple Attribute Decision Making. Springer.
2. Opricovic, S., & Tzeng, G.H. (2004). Compromise solution by MCDM methods. EJOR.
3. Brans, J.P., & Vincke, P. (1985). A preference ranking organisation method. Management Science.
4. Zavadskas, E.K., & Kaklauskas, A. (1996). Determination of a rational contractor. Statyba.
5. Ghorabaee, M.K., et al. (2015). Multi-criteria inventory classification using EDAS. Informatica.
