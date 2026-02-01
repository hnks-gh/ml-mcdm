# Fuzzy MCDM Methods

This module provides fuzzy extensions of traditional Multi-Criteria Decision Making (MCDM) methods using **Triangular Fuzzy Numbers (TFN)** to handle uncertainty in decision-making processes.

## Overview

Fuzzy MCDM methods extend classical approaches by representing uncertain or imprecise data as fuzzy numbers, providing more robust rankings under uncertainty.

### Core Concept: Triangular Fuzzy Number (TFN)

A TFN $\tilde{A} = (l, m, u)$ represents uncertain values where:
- $l$ = lower bound (minimum possible value)
- $m$ = modal value (most likely value)
- $u$ = upper bound (maximum possible value)

The membership function is:

$$\mu_{\tilde{A}}(x) = \begin{cases} \frac{x - l}{m - l} & \text{if } l \leq x \leq m \\ \frac{u - x}{u - m} & \text{if } m \leq x \leq u \\ 0 & \text{otherwise} \end{cases}$$

---

## Available Methods

### 1. Fuzzy TOPSIS
**File:** `topsis.py`

Fuzzy Technique for Order Preference by Similarity to Ideal Solution.

**Key Features:**
- Extends TOPSIS with fuzzy ideal/anti-ideal solutions
- Uses vertex distance for fuzzy comparisons
- Calculates closeness coefficient from fuzzy distances

**Formula:**
$$CC_i = \frac{d_i^-}{d_i^* + d_i^-}$$

where $d_i^*$ and $d_i^-$ are distances to fuzzy positive and negative ideal solutions.

**Usage:**
```python
from src.mcdm.fuzzy import FuzzyTOPSIS

calculator = FuzzyTOPSIS(cost_criteria=['Cost', 'Risk'])
result = calculator.calculate_from_panel(panel_data, weights=weights)
print(result.top_n(5))
```

---

### 2. Fuzzy VIKOR
**File:** `vikor.py`

Fuzzy VIseKriterijumska Optimizacija I Kompromisno Resenje.

**Key Features:**
- Finds compromise solutions under uncertainty
- Balances maximum group utility (S) and minimum regret (R)
- Parameter `v` controls trade-off (v=0.5 is balanced)

**Formulas:**
$$\tilde{S}_i = \sum_j w_j \times \frac{f_j^* - \tilde{f}_{ij}}{f_j^* - f_j^-}$$

$$\tilde{R}_i = \max_j \left[ w_j \times \frac{f_j^* - \tilde{f}_{ij}}{f_j^* - f_j^-} \right]$$

$$\tilde{Q}_i = v \times \frac{\tilde{S}_i - S^*}{S^- - S^*} + (1-v) \times \frac{\tilde{R}_i - R^*}{R^- - R^*}$$

**Compromise Conditions:**
- C1 (Advantage): $Q(a'') - Q(a') \geq 1/(m-1)$
- C2 (Stability): $a'$ is also best in S or R ranking

**Usage:**
```python
from src.mcdm.fuzzy import FuzzyVIKOR

calculator = FuzzyVIKOR(v=0.5, cost_criteria=['Cost'])
result = calculator.calculate_from_panel(panel_data)
print(f"Compromise solution: {result.compromise_solution}")
print(f"Advantage condition: {result.advantage_condition}")
```

---

### 3. Fuzzy PROMETHEE
**File:** `promethee.py`

Fuzzy Preference Ranking Organization Method for Enrichment Evaluation.

**Key Features:**
- Pairwise comparison with fuzzy preferences
- Multiple preference functions available
- Calculates outranking flows (leaving, entering, net)

**Preference Functions:**
| Type | Formula |
|------|---------|
| `usual` | $P(d) = 1$ if $d > 0$, else $0$ |
| `ushape` | $P(d) = 1$ if $d > q$, else $0$ |
| `vshape` | $P(d) = d/p$ if $d < p$, else $1$ |
| `level` | $P(d) = 0.5$ if $q < d \leq p$ |
| `vshape_i` | Linear with indifference threshold |

**Outranking Flows:**
$$\tilde{\phi}^+(a) = \frac{1}{n-1} \sum_{x \neq a} \tilde{\pi}(a, x)$$

$$\tilde{\phi}^-(a) = \frac{1}{n-1} \sum_{x \neq a} \tilde{\pi}(x, a)$$

$$\tilde{\phi}(a) = \tilde{\phi}^+(a) - \tilde{\phi}^-(a)$$

**Usage:**
```python
from src.mcdm.fuzzy import FuzzyPROMETHEE

calculator = FuzzyPROMETHEE(
    preference_function='vshape',
    preference_threshold=0.3,
    indifference_threshold=0.1
)
result = calculator.calculate_from_panel(panel_data)
```

---

### 4. Fuzzy COPRAS
**File:** `copras.py`

Fuzzy Complex Proportional Assessment method.

**Key Features:**
- Separates benefit and cost criteria explicitly
- Calculates relative significance Q
- Provides utility degree as percentage

**Formulas:**
$$\tilde{S}^+_i = \sum_{j \in B} \tilde{d}_{ij}$$ (sum of benefit criteria)

$$\tilde{S}^-_i = \sum_{j \in C} \tilde{d}_{ij}$$ (sum of cost criteria)

$$\tilde{Q}_i = \tilde{S}^+_i + \frac{S^-_{min} \times \sum \tilde{S}^-}{\tilde{S}^-_i \times \sum(S^-_{min}/\tilde{S}^-_k)}$$

$$N_i = \frac{Q_i}{Q_{max}} \times 100\%$$ (utility degree)

**Usage:**
```python
from src.mcdm.fuzzy import FuzzyCOPRAS

calculator = FuzzyCOPRAS(cost_criteria=['Cost', 'Risk'])
result = calculator.calculate_from_panel(panel_data)
print(f"Utility degrees: {result.utility_degree}")
```

---

### 5. Fuzzy EDAS
**File:** `edas.py`

Fuzzy Evaluation based on Distance from Average Solution.

**Key Features:**
- Uses average solution as reference (robust to outliers)
- Calculates positive (PDA) and negative (NDA) distances
- Combined appraisal score

**Formulas:**
$$\tilde{AV}_j = \frac{1}{n} \sum_i \tilde{x}_{ij}$$ (average solution)

For benefit criteria:
$$\tilde{PDA}_{ij} = \frac{\max(0, \tilde{x}_{ij} - \tilde{AV}_j)}{\tilde{AV}_j}$$

$$\tilde{NDA}_{ij} = \frac{\max(0, \tilde{AV}_j - \tilde{x}_{ij})}{\tilde{AV}_j}$$

$$AS_i = \frac{NSP_i + NSN_i}{2}$$ (appraisal score)

where $NSP_i = SP_i / \max(SP)$ and $NSN_i = 1 - SN_i / \max(SN)$

**Usage:**
```python
from src.mcdm.fuzzy import FuzzyEDAS

calculator = FuzzyEDAS(cost_criteria=['Cost'])
result = calculator.calculate_from_panel(panel_data)
print(result.top_n(5))
```

---

## Core Components

### TriangularFuzzyNumber Class
**File:** `base.py`

```python
from src.mcdm.fuzzy import TriangularFuzzyNumber

# Create from values
tfn = TriangularFuzzyNumber(0.2, 0.5, 0.8)

# Arithmetic operations
sum_tfn = tfn1 + tfn2
diff_tfn = tfn1 - tfn2
prod_tfn = tfn1 * tfn2
scaled = tfn * 2.0

# Defuzzification
crisp = tfn.defuzzify('centroid')  # (l + m + u) / 3
crisp = tfn.defuzzify('mom')       # m (mean of maximum)
crisp = tfn.defuzzify('bisector')  # (l + 2m + u) / 4

# Distance calculation
dist = tfn1.distance(tfn2)  # Vertex distance

# Create from crisp value
tfn = TriangularFuzzyNumber.from_crisp(0.5, spread=0.1)
```

### FuzzyDecisionMatrix Class

```python
from src.mcdm.fuzzy import FuzzyDecisionMatrix

# From crisp data with uncertainty
fuzzy_matrix = FuzzyDecisionMatrix.from_crisp_with_uncertainty(
    data, 
    uncertainty=std_matrix,  # Optional
    spread_ratio=0.1
)

# From panel data (uses temporal variance)
fuzzy_matrix = FuzzyDecisionMatrix.from_panel_temporal_variance(
    panel_data,
    spread_factor=1.0
)

# Convert back to crisp
crisp_df = fuzzy_matrix.to_crisp('centroid')
```

### Linguistic Scales

Predefined scales for converting linguistic terms to fuzzy numbers:

```python
from src.mcdm.fuzzy import LINGUISTIC_SCALE_5, LINGUISTIC_SCALE_7, IMPORTANCE_SCALE

# 5-point scale
LINGUISTIC_SCALE_5 = {
    'very_low':  TFN(0.0, 0.0, 0.25),
    'low':       TFN(0.0, 0.25, 0.50),
    'medium':    TFN(0.25, 0.50, 0.75),
    'high':      TFN(0.50, 0.75, 1.0),
    'very_high': TFN(0.75, 1.0, 1.0),
}

# 7-point scale available for finer granularity
# IMPORTANCE_SCALE for criteria weight assessments
```

---

## Common Usage Patterns

### Pattern 1: From Panel Data
```python
# Uses temporal variance for fuzzy spreads
result = calculator.calculate_from_panel(
    panel_data,
    weights=weights,
    spread_factor=1.0  # Multiplier for variance
)
```

### Pattern 2: From Crisp Data with Uncertainty
```python
# Explicit uncertainty matrix
result = calculator.calculate(
    data,
    weights=weights,
    uncertainty=std_matrix,  # Optional std matrix
    spread_ratio=0.1  # Used if no uncertainty matrix
)
```

### Pattern 3: Accessing Results
```python
# All results have consistent interface
result.final_ranks      # pd.Series of rankings
result.top_n(5)         # Top 5 alternatives
result.weights          # Weights used

# Method-specific attributes
result.fuzzy_*          # Fuzzy values before defuzzification
```

---

## Method Comparison

| Method | Approach | Best For | Key Output |
|--------|----------|----------|------------|
| **Fuzzy TOPSIS** | Distance-based | General ranking | Closeness coefficient |
| **Fuzzy VIKOR** | Compromise | Conflicting criteria | Compromise set |
| **Fuzzy PROMETHEE** | Outranking | Pairwise preferences | Net flows |
| **Fuzzy COPRAS** | Utility-based | Clear benefit/cost | Utility degree |
| **Fuzzy EDAS** | Average-based | Outlier resistance | Appraisal score |

---

## References

1. Chen, C.T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy environment.
2. Opricovic, S., & Tzeng, G.H. (2007). Extended VIKOR method comparison.
3. Brans, J.P., & Vincke, P. (1985). A Preference Ranking Organisation Method.
4. Zavadskas, E.K., et al. (1994). The new method of multicriteria complex proportional assessment.
5. Keshavarz Ghorabaee, M., et al. (2015). EDAS method for multi-criteria inventory classification.
