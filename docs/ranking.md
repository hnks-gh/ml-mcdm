# Ranking Methodology: IFS-MCDM with Evidential Reasoning

## Overview

This framework implements a **two-stage hierarchical ranking system** that combines:
1. **Intuitionistic Fuzzy Sets (IFS)** - Atanassov (1986) framework for handling uncertainty
2. **Multi-Criteria Decision Making (MCDM)** - 12 complementary ranking methods
3. **Evidential Reasoning (ER)** - Yang & Xu (2002) for rigorous belief aggregation

**Key Innovation:** Replaces traditional fuzzy triangular numbers (TFN) and voting-based aggregation with mathematically rigorous IFS representation and ER combination.

---

## System Architecture

```
Input: Panel Data (N provinces × p subcriteria × T years)
  ↓
Stage 0: Adaptive Data Preprocessing
  ├── Exclude provinces with all-zero data
  ├── Exclude subcriteria with all-zero values
  └── Min-max normalization to [0, 1]
  ↓
Stage 1: IFS Construction
  ├── Membership (μ): Normalized criterion value
  ├── Non-membership (ν): From temporal variance (hesitancy)
  └── Hesitancy (π): π = 1 - μ - ν
  ↓
Stage 2: Within-Criterion Ranking (8 criteria groups)
  │
  ├── Traditional MCDM (6 methods on normalized values)
  │   ├── TOPSIS (vector normalization)
  │   ├── VIKOR (compromise programming)
  │   ├── PROMETHEE (pairwise preferences)
  │   ├── COPRAS (ratio assessment)
  │   ├── EDAS (distance from average)
  │   └── SAW (weighted sum)
  │
  ├── IFS-MCDM (6 methods on IFN matrices)
  │   ├── IFS-TOPSIS (IFN ideal solutions)
  │   ├── IFS-VIKOR (IFN compromise)
  │   ├── IFS-PROMETHEE (IFN preferences)
  │   ├── IFS-COPRAS (IFN ratio assessment)
  │   ├── IFS-EDAS (IFN average deviation)
  │   └── IFS-SAW (IFN weighted aggregation)
  │
  └── Evidential Reasoning Combination (12 scores → 1 belief per criterion)
  ↓
Stage 3: Global Ranking (combine 8 criteria)
  ├── Apply criterion weights (from GTWC)
  ├── ER aggregation of 8 belief distributions
  └── Compute final scores and rankings
  ↓
Output: Final Rankings + Uncertainty Quantification
```

---

## Part I: Intuitionistic Fuzzy Sets (IFS)

### 1.1 Theoretical Foundation

**Reference:** Atanassov, K.T. (1986). "Intuitionistic fuzzy sets." *Fuzzy Sets and Systems*, 20(1), 87-96.

An **Intuitionistic Fuzzy Number (IFN)** extends classical fuzzy sets by introducing non-membership:

$$
\text{IFN} = (\mu, \nu, \pi)
$$

Where:
- **μ (mu)**: Membership degree — "to what extent the element belongs" ∈ [0, 1]
- **ν (nu)**: Non-membership degree — "to what extent the element does NOT belong" ∈ [0, 1]
- **π (pi)**: Hesitancy/Uncertainty — "the degree of indeterminacy"

**Constraint:**
$$
\mu + \nu \leq 1, \quad \pi = 1 - \mu - \nu
$$

**Key Advantage Over Classical Fuzzy Sets:**
- Classical fuzzy: ν = 1 - μ (forced complementarity)
- IFS: ν is independent, allowing for genuine hesitancy (π > 0)

### 1.2 Score Function

To convert IFN to crisp score for comparison:

$$
S(\text{IFN}) = \mu - \nu \in [-1, 1]
$$

**Reference:** Chen & Tan (1994)

**Interpretation:**
- S = 1: Perfect membership (μ=1, ν=0)
- S = 0: Complete hesitancy (μ=0, ν=0, π=1) or neutrality (μ=0.5, ν=0.5)
- S = -1: Perfect non-membership (μ=0, ν=1)

> **Note:** Some presentations rescale to [0,1] via $S' = (\mu - \nu + 1)/2$; the
> implementation uses the raw Chen–Tan form $S = \mu - \nu$ which preserves the
> same ranking order.

### 1.3 IFS Construction from Temporal Data

**Method:** `IFSDecisionMatrix.from_temporal_variance()`

Given:
- `current_data`: Normalized criterion values for target year (μ baseline)
- `historical_std`: Standard deviation across all years (captures variability)
- `global_range`: Global max - min for each criterion
- `spread_factor`: Hyperparameter controlling ν sensitivity (default: 0.5)

**Algorithm:**

```python
for each alternative i, criterion j:
    # Step 1: Membership from normalized current value
    μ_ij = current_data[i, j]  # Already in [0, 1]
    
    # Step 2: Hesitancy from temporal uncertainty
    relative_std = historical_std[i, j] / (global_range[j] + ε)
    π_ij = min(spread_factor × relative_std, 1 − μ_ij)
    π_ij = max(π_ij, 0)
    
    # Step 3: Non-membership as remainder
    ν_ij = 1 − μ_ij − π_ij
    
    IFN[i, j] = (μ_ij, ν_ij, π_ij)
```

**Rationale:**
- High temporal variability → high π (more hesitancy / indeterminacy)
- Stable performance → low π (confident in current value)
- Non-membership ν absorbs the leftover after μ and π are set

### 1.4 IFS Distance Metrics

All distances below are the **Szmidt–Kacprzyk normalized** forms
(factor $1/2$ ensures $d \in [0,1]$ for a single IFN pair).

#### Normalized Euclidean Distance
$$
d_E(A, B) = \sqrt{\frac{1}{2}\left[(\mu_A - \mu_B)^2 + (\nu_A - \nu_B)^2 + (\pi_A - \pi_B)^2\right]}
$$

#### Normalized Hamming Distance
$$
d_H(A, B) = \frac{1}{2}\left(|\mu_A - \mu_B| + |\nu_A - \nu_B| + |\pi_A - \pi_B|\right)
$$

#### Szmidt-Kacprzyk Normalized Distance

Same as the normalized Euclidean distance above (used in IFS-VIKOR and IFS-TOPSIS):
$$
d_{SK}(A, B) = \sqrt{\frac{1}{2}\left[(\mu_A - \mu_B)^2 + (\nu_A - \nu_B)^2 + (\pi_A - \pi_B)^2\right]}
$$

---

## Part II: MCDM Methods

### 2.1 Traditional MCDM Methods

All traditional methods operate on **min-max normalized crisp values** ∈ [0, 1].

#### 2.1.1 TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

**Reference:** Hwang & Yoon (1981)

**Steps:**
1. Normalize decision matrix (vector normalization: $x_{ij} / \sqrt{\sum x_{ik}^2}$)
2. Calculate weighted normalized matrix: $v_{ij} = w_j \times x_{ij}$
3. Determine ideal solutions:
   - Ideal: $A^+ = \{\max(v_{ij})\}$ for benefits, $\{\min(v_{ij})\}$ for costs
   - Anti-ideal: $A^- = \{\min(v_{ij})\}$ for benefits, $\{\max(v_{ij})\}$ for costs
4. Calculate distances:
   - $d_i^+ = \sqrt{\sum (v_{ij} - v_j^+)^2}$
   - $d_i^- = \sqrt{\sum (v_{ij} - v_j^-)^2}$
5. Closeness coefficient: $C_i = \frac{d_i^-}{d_i^+ + d_i^-}$ (higher is better)

**Pros:** Intuitive, considers both ideal and anti-ideal  
**Cons:** Sensitive to normalization method

---

#### 2.1.2 VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje)

**Reference:** Opricovic & Tzeng (2004)

**Compromise Programming Approach:**

1. Best/worst values per criterion:
   - $f_j^* = \max_i x_{ij}$ (benefit) or $\min_i x_{ij}$ (cost)
   - $f_j^- = \min_i x_{ij}$ (benefit) or $\max_i x_{ij}$ (cost)

2. Calculate utility (S) and regret (R):
   $$
   S_i = \sum_j w_j \frac{f_j^* - x_{ij}}{f_j^* - f_j^-}
   $$
   $$
   R_i = \max_j \left(w_j \frac{f_j^* - x_{ij}}{f_j^* - f_j^-}\right)
   $$

3. VIKOR index (Q):
   $$
   Q_i = v \frac{S_i - S^*}{S^- - S^*} + (1-v) \frac{R_i - R^*}{R^- - R^*}
   $$
   where $v = 0.5$ (strategy weight for majority)

**Pros:** Balances group utility and individual regret  
**Cons:** Requires acceptable advantage and stability conditions

---

#### 2.1.3 PROMETHEE (Preference Ranking Organization Method for Enrichment Evaluations)

**Reference:** Brans & Vincke (1985)

**Pairwise Preference Function:**

For each criterion j, define preference $P_j(a, b)$ based on difference $d = x_{aj} - x_{bj}$:

**V-shape preference function (Type III, default):**
$$
P_j(a, b) = \begin{cases}
0 & \text{if } d \leq 0 \\
\frac{d}{p} & \text{if } 0 < d < p \\
1 & \text{if } d \geq p
\end{cases}
$$

where:
- p: Preference threshold (default 0.3)

> **Note:** A V-shape with indifference threshold (Type V) is also
> available via $q$ (indifference threshold, default 0.1):
>
> $P_j = 0$ if $d \le q$; $P_j = (d-q)/(p-q)$ if $q < d < p$; $P_j = 1$ if $d \ge p$.

**Aggregated Preferences:**
$$
\pi(a, b) = \sum_j w_j P_j(a, b)
$$

**Net Flow (PROMETHEE II):**
$$
\Phi_{net}(a) = \Phi^+(a) - \Phi^-(a)
$$
where:
- $\Phi^+(a) = \frac{1}{n-1} \sum_{b \neq a} \pi(a, b)$ (outranking flow)
- $\Phi^-(a) = \frac{1}{n-1} \sum_{b \neq a} \pi(b, a)$ (outranked flow)

**Pros:** Rich preference modeling, transparent  
**Cons:** Requires threshold calibration

---

#### 2.1.4 COPRAS (Complex Proportional Assessment)

**Reference:** Zavadskas et al. (1994)

**Maximizing and Minimizing Indices:**

1. Weighted normalized sum for benefits:
   $$
   S_{+i} = \sum_{j \in J_{max}} w_j \cdot \frac{x_{ij}}{\sum_i x_{ij}}
   $$

2. Weighted normalized sum for costs:
   $$
   S_{-i} = \sum_{j \in J_{min}} w_j \cdot \frac{x_{ij}}{\sum_i x_{ij}}
   $$

3. Relative significance:
   $$
   Q_i = S_{+i} + \frac{\sum_k S_{-k}}{S_{-i} \cdot \sum_k \frac{1}{S_{-k}}}
   $$

4. Utility degree:
   $$
   U_i = \frac{Q_i}{Q_{max}} \times 100\%
   $$

**Pros:** Direct ratio comparisons, clear economic interpretation  
**Cons:** Complex formula, sensitive to zero values

---

#### 2.1.5 EDAS (Evaluation based on Distance from Average Solution)

**Reference:** Keshavarz Ghorabaee et al. (2015)

**Distance from Average:**

1. Average solution:
   $$
   AV_j = \frac{\sum_i x_{ij}}{n}
   $$

2. Positive distance from average (PDA):
   $$
   PDA_{ij} = \frac{\max(0, x_{ij} - AV_j)}{AV_j}
   $$

3. Negative distance from average (NDA):
   $$
   NDA_{ij} = \frac{\max(0, AV_j - x_{ij})}{AV_j}
   $$

4. Weighted sums:
   $$
   SP_i = \sum_j w_j \cdot PDA_{ij}, \quad SN_i = \sum_j w_j \cdot NDA_{ij}
   $$

5. Normalize:
   $$
   NSP_i = \frac{SP_i}{\max_k SP_k}, \quad NSN_i = 1 - \frac{SN_i}{\max_k SN_k}
   $$

6. Appraisal score:
   $$
   AS_i = \frac{NSP_i + NSN_i}{2}
   $$

**Pros:** Uses average as reference (robust to outliers)  
**Cons:** May not distinguish well if all alternatives are similar

---

#### 2.1.6 SAW (Simple Additive Weighting)

**Reference:** Fishburn (1967)

**Weighted Linear Combination:**

1. Normalize:
   - Benefit: $r_{ij} = \frac{x_{ij} - \min_i x_{ij}}{\max_i x_{ij} - \min_i x_{ij}}$
   - Cost: $r_{ij} = \frac{\max_i x_{ij} - x_{ij}}{\max_i x_{ij} - \min_i x_{ij}}$

2. Aggregate:
   $$
   S_i = \sum_j w_j \cdot r_{ij}
   $$

**Pros:** Simple, transparent, easy to interpret  
**Cons:** Assumes full compensability (high score in one criterion can compensate for low in another)

---

### 2.2 IFS-MCDM Methods

All IFS methods operate on **IFSDecisionMatrix** containing IFNs (μ, ν, π).

#### 2.2.1 IFS-TOPSIS

**Extension:** Ideal/anti-ideal solutions are IFNs, distances use IFS metrics.

**Steps:**
1. Weighted IFS matrix: $\tilde{v}_{ij} = w_j \otimes \text{IFN}_{ij}$
   - Scalar multiplication (Xu, 2007): $w \otimes (\mu, \nu, \pi) = (1-(1-\mu)^w,\; \nu^w,\; \text{remainder})$

2. IFS ideal solutions:
   - Benefit: $\tilde{A}^+ = (\max(\mu_{ij}), \min(\nu_{ij}), \pi)$
   - Cost: $\tilde{A}^+ = (\min(\mu_{ij}), \max(\nu_{ij}), \pi)$

3. IFS distances:
   $$
   d^+_i = \sqrt{\sum_j d_{SK}(\tilde{v}_{ij}, \tilde{A}^+_j)^2}
   $$

4. Closeness:
   $$
   C_i = \frac{d^-_i}{d^+_i + d^-_i}
   $$

---

#### 2.2.2 IFS-VIKOR

**Extension:** Best/worst are IFNs based on score function.

**Algorithm:**
1. IFS best/worst per criterion:
   - $\tilde{f}_j^* = \text{IFN with } \max(S(\text{IFN}_{ij}))$
   - $\tilde{f}_j^- = \text{IFN with } \min(S(\text{IFN}_{ij}))$

2. IFS utility (S) and regret (R):
   $$
   S_i = \sum_j w_j \cdot d_{SK}(\tilde{f}_j^*, \text{IFN}_{ij})
   $$
   $$
   R_i = \max_j \left(w_j \cdot d_{SK}(\tilde{f}_j^*, \text{IFN}_{ij})\right)
   $$

3. IFS-VIKOR Q index (same normalization as traditional VIKOR)

---

#### 2.2.3 IFS-PROMETHEE

**Extension:** Preference functions use IFS score differences.

**Preference Function:**
$$
d = S(\text{IFN}_a) - S(\text{IFN}_b)
$$

Then apply standard V-shape function on d.

**Net Flow:** Same as traditional PROMETHEE II.

---

#### 2.2.4 IFS-COPRAS

**Extension:** Operate on IFS scores.

Replace $x_{ij}$ with $S(\text{IFN}_{ij})$ in COPRAS formulas.

---

#### 2.2.5 IFS-EDAS

**Extension:** Average solution is IFS.

Average IFN per criterion:
$$
\overline{\text{IFN}}_j = \left(\frac{1}{n}\sum_i \mu_{ij}, \frac{1}{n}\sum_i \nu_{ij}, \ldots\right)
$$

Distance from average uses IFS score differences.

---

#### 2.2.6 IFS-SAW

**Extension:** Weighted sum of IFS score values.

The implementation uses the Chen–Tan score function for each cell,
then applies a simple weighted sum:

$$
\text{IFS-SAW}_i = \sum_{j=1}^p w_j \cdot S(\text{IFN}_{ij})
$$

where $S(\text{IFN}) = \mu - \nu$.

> **Theoretical alternative (not implemented):** The IFWA operator
> aggregates directly in IFS space:
>
> $\text{IFWA}(\text{IFN}_1, \ldots, \text{IFN}_p) = \left(1 - \prod_j (1-\mu_j)^{w_j},\; \prod_j \nu_j^{w_j}\right)$
>
> The score-based approach is adopted for computational simplicity
> and consistency with the other IFS-MCDM methods.

---

## Part III: Evidential Reasoning (ER)

### 3.1 Theoretical Foundation

**Reference:** Yang, J.B., & Xu, D.L. (2002). "On the evidential reasoning algorithm for multiple attribute decision analysis under uncertainty." *IEEE Transactions on Systems, Man, and Cybernetics—Part A*, 32(3), 289-304.

**Purpose:** Combine multiple uncertain assessments (from 12 MCDM methods) into a single belief distribution.

### 3.2 Belief Distribution Representation

Each MCDM method produces a score → convert to **belief distribution** over 5 evaluation grades:

$$
\text{Belief} = \{(\text{Excellent}, \beta_E), (\text{Good}, \beta_G), (\text{Fair}, \beta_F), (\text{Poor}, \beta_P), (\text{Bad}, \beta_B), (H, \beta_H)\}
$$

Where:
- $\beta_E, \beta_G, \beta_F, \beta_P, \beta_B \geq 0$ (belief masses assigned to grades)
- $H$ = unassigned belief (unknown/uncertain)
- $\sum \beta_i + \beta_H = 1$

**Grade Utility Values:**
- Excellent: 1.0
- Good: 0.75
- Fair: 0.5
- Poor: 0.25
- Bad: 0.0

### 3.3 Score-to-Belief Conversion

**Method:** Linear interpolation between adjacent grade utilities.

Grade utilities are evenly spaced from 1.0 (Excellent) to 0.0 (Bad):

| Grade     | Utility |
|-----------|---------|
| Excellent | 1.00    |
| Good      | 0.75    |
| Fair      | 0.50    |
| Poor      | 0.25    |
| Bad       | 0.00    |

Given normalized score $s \in [0, 1]$, find the two adjacent grades
$H_k, H_{k+1}$ such that $u_k \geq s \geq u_{k+1}$ and interpolate:

$$
\beta_k = \frac{s - u_{k+1}}{u_k - u_{k+1}}, \quad \beta_{k+1} = 1 - \beta_k
$$

**Example (s = 0.65):**

```python
# s=0.65 falls between Good (0.75) and Fair (0.50)
β_Good = (0.65 - 0.50) / (0.75 - 0.50) = 0.60
β_Fair = 1 - 0.60 = 0.40
# All other grades receive zero belief
```

**Rationale:** Uncertainty is captured by splitting belief between grades, not by H (we have 12 sources, consensus reduces H).

### 3.4 ER Analytical Algorithm

**Recursive Combination of N Sources:**

For each alternative, combine beliefs from 12 MCDM methods:

$$
\text{Combined} = \text{ER}(\text{ER}(\ldots\text{ER}(\text{Belief}_1, \text{Belief}_2), \ldots), \text{Belief}_{12})
$$

**Pairwise Combination Formula:**

Given two beliefs $B_1, B_2$ over grades $H_n$ (n = 1..5):

$$
\beta_n = K \left[\beta_{1,n}\beta_{2,n} + \beta_{1,n}\beta_{2,H} + \beta_{1,H}\beta_{2,n}\right]
$$

$$
\beta_H = K \left[\beta_{1,H}\beta_{2,H}\right]
$$

Where normalization constant:
$$
K = \left[1 - \sum_{i=1}^5 \sum_{j=1, j \neq i}^5 \beta_{1,i}\beta_{2,j}\right]^{-1}
$$

**Interpretation:**
- Conflicting evidence (e.g., one says "Excellent", another says "Bad") increases uncertainty
- Consensus evidence strengthens belief
- K handles normalization when sources conflict

### 3.5 Expected Utility

After ER combination, compute final score:

$$
\text{Score} = \sum_{n=1}^5 \beta_n \cdot u(H_n) + \beta_H \cdot \bar{u}
$$

Where:
- $u(H_n)$ = utility of grade n (1.0, 0.75, 0.5, 0.25, 0.0)
- $\bar{u}$ = average utility = 0.5 (neutral assumption for unassigned belief)

### 3.6 Two-Stage ER Architecture

#### Stage 1: Within Each Criterion

For criterion C (e.g., "Economic Development" with 4 subcriteria):

1. 12 MCDM methods → 12 scores per province
2. Convert each score to belief distribution
3. ER combination → 1 belief distribution per province for criterion C
4. Compute expected utility → criterion-level score

**Repeat for all 8 criteria.**

#### Stage 2: Global Combination

1. 8 criterion-level beliefs (from Stage 1)
2. Weight each belief by criterion importance $w_k$ (from GTWC)
3. Weighted ER combination:
   $$
   \beta_{k,n}^{weighted} = w_k \cdot \beta_{k,n}
   $$
4. Final ER aggregation → global belief distribution
5. Expected utility → final ranking score

---

## Part IV: Adaptive Data Handling

### 4.1 Zero-Handling Mechanism

**Problem:** Some provinces may have all-zero data for certain criteria groups (missing data).

**Solution:** Adaptive filtering + restoration.

**Algorithm:**

```python
# Step 1: Filter provinces with all-zero rows
row_sums = data.sum(axis=1)
valid_provinces = data[row_sums > 0]
excluded_provinces = [p for p in provinces if p not in valid_provinces]

# Step 2: Filter subcriteria with all-zero columns
col_sums = valid_provinces.sum(axis=0)
valid_subcriteria = valid_provinces[:, col_sums > 0]
excluded_subcriteria = [c for c in subcriteria if c not in valid_subcriteria]

# Step 3: Perform MCDM ranking on filtered data
rankings = run_12_methods(valid_provinces, valid_subcriteria)

# Step 4: Restore excluded provinces with median rank/score
median_score = rankings['scores'].median()
median_rank = int(rankings['ranks'].median())

for p in excluded_provinces:
    rankings['scores'][p] = median_score
    rankings['ranks'][p] = median_rank
```

**Rationale:**
- Zero data shouldn't distort min/max normalization
- Excluded provinces still appear in output (completeness)
- Median restoration is conservative (neutral assumption)

### 4.2 Normalization After Filtering

**Critical Order:**

```
Raw Data → Zero Filtering → Min-Max Normalization → IFS Construction → MCDM
```

**Why?**
- Zero rows/columns would artificially shift min/max
- Normalization on valid data only ensures accurate [0, 1] scaling
- Each criterion group normalizes independently

**Formula:**

For benefit criteria:
$$
x_{ij}^{norm} = \frac{x_{ij} - \min_i x_{ij}}{\max_i x_{ij} - \min_i x_{ij}}
$$

For cost criteria (inverted):
$$
x_{ij}^{norm} = \frac{\max_i x_{ij} - x_{ij}}{\max_i x_{ij} - \min_i x_{ij}}
$$

Constant columns (range = 0):
$$
x_{ij}^{norm} = 0.5
$$

---

## Part V: Implementation Details

### 5.1 Pipeline Workflow

**File:** `src/ranking/pipeline.py`  
**Class:** `HierarchicalRankingPipeline`

**Main Method:** `rank(panel_data, weights_dict, target_year)`

**Pseudocode:**

```python
def rank(panel_data, weights_dict, target_year):
    # Prepare data structures
    alternatives = panel_data.provinces
    hierarchy = panel_data.hierarchy  # 28 subcrit → 8 crit
    
    # Stage 1: For each of 8 criteria
    for criterion_id in criteria:
        subcriteria_group = hierarchy.get_subcriteria(criterion_id)
        data = extract_data(criterion_id, target_year)
        
        # Adaptive filtering
        data_filtered = zero_filter(data)
        
        # Normalization
        data_norm = minmax_normalize(data_filtered)
        
        # IFS construction
        ifs_matrix = IFSDecisionMatrix.from_temporal_variance(
            current_data=data_norm,
            historical_std=compute_std(panel_data),
            global_range=compute_range(panel_data),
            spread_factor=0.5
        )
        
        # Run 12 MCDM methods
        results = {}
        for method in [TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW,
                       IFS_TOPSIS, IFS_VIKOR, IFS_PROMETHEE, 
                       IFS_COPRAS, IFS_EDAS, IFS_SAW]:
            scores, ranks = method.calculate(ifs_matrix, weights)
            results[method.name] = normalize_scores(scores)
        
        # ER combination (12 methods → 1 belief per province)
        criterion_beliefs = {}
        for province in alternatives:
            beliefs = [score_to_belief(results[m][province]) for m in methods]
            combined = ER_combine(beliefs)
            criterion_beliefs[province] = combined
        
        save_criterion_result(criterion_id, criterion_beliefs)
    
    # Stage 2: Global ER with criterion weights
    final_beliefs = {}
    for province in alternatives:
        # Collect 8 criterion-level beliefs
        crit_beliefs = [
            get_criterion_belief(criterion_id, province) 
            for criterion_id in criteria
        ]
        
        # Weight by criterion importance
        weighted_beliefs = [
            weight_belief(b, weights_dict[crit]) 
            for b, crit in zip(crit_beliefs, criteria)
        ]
        
        # Final ER combination
        final_beliefs[province] = ER_combine(weighted_beliefs)
    
    # Compute final scores and rankings
    final_scores = {p: expected_utility(final_beliefs[p]) for p in alternatives}
    final_rankings = rank_by_scores(final_scores)
    
    return HierarchicalRankingResult(
        final_scores=final_scores,
        final_rankings=final_rankings,
        final_beliefs=final_beliefs,
        kendall_w=compute_concordance(final_rankings)
    )
```

### 5.2 Configuration

**File:** `src/config.py`

```python
@dataclass
class IFSConfig:
    """IFS construction parameters."""
    spread_factor: float = 0.5           # ν sensitivity
    min_membership: float = 0.0
    max_nonmembership: float = 1.0
    
@dataclass
class EvidentialReasoningConfig:
    """ER aggregation parameters."""
    grade_utilities: List[float] = field(
        default_factory=lambda: [1.0, 0.75, 0.5, 0.25, 0.0]
    )
    default_unassigned: float = 0.5      # Utility for H
    score_grade_boundaries: List[float] = field(
        default_factory=lambda: [0.8, 0.6, 0.4, 0.2]
    )
```

### 5.3 Key Data Structures

**IFN (Intuitionistic Fuzzy Number):**
```python
@dataclass
class IFN:
    mu: float      # Membership
    nu: float      # Non-membership
    pi: float      # Hesitancy
    
    def score(self) -> float:
        return (self.mu - self.nu + 1) / 2
```

**IFSDecisionMatrix:**
```python
class IFSDecisionMatrix:
    def __init__(self, matrix: Dict[str, Dict[str, IFN]], 
                 alternatives: List[str], 
                 criteria: List[str]):
        self.matrix = matrix  # {alternative: {criterion: IFN}}
        self.alternatives = alternatives
        self.criteria = criteria
```

**BeliefDistribution:**
```python
@dataclass
class BeliefDistribution:
    Excellent: float = 0.0
    Good: float = 0.0
    Fair: float = 0.0
    Poor: float = 0.0
    Bad: float = 0.0
    H: float = 1.0  # Unassigned (initially all uncertain)
    
    def expected_utility(self, utilities=[1.0, 0.75, 0.5, 0.25, 0.0]) -> float:
        return (self.Excellent * utilities[0] + 
                self.Good * utilities[1] + 
                self.Fair * utilities[2] + 
                self.Poor * utilities[3] + 
                self.Bad * utilities[4] + 
                self.H * 0.5)
```

---

## Part VI: Advantages & Limitations

### 6.1 Advantages

1. **Mathematical Rigor**
   - IFS provides formal uncertainty representation
   - ER has proven convergence properties
   - No arbitrary voting schemes

2. **Uncertainty Quantification**
   - μ, ν, π capture multiple dimensions of uncertainty
   - Temporal variance informs non-membership
   - Belief distributions preserve uncertainty through aggregation

3. **Robustness**
   - 12 methods provide diverse perspectives
   - ER handles conflicting evidence gracefully
   - Adaptive zero-handling prevents data artifacts

4. **Hierarchical Structure**
   - Two-stage design respects criterion hierarchy
   - Within-criterion ER captures subcriteria interactions
   - Global ER applies criterion weights at final stage

5. **Transparency**
   - Each MCDM method interpretable independently
   - Belief distributions show consensus level
   - Final scores traceable to source methods

### 6.2 Limitations

1. **Computational Complexity**
   - O(n × m × k) where n=provinces, m=methods, k=criteria
   - ER pairwise combination is O(m²) per criterion

2. **Hyperparameter Sensitivity**
   - `spread_factor` affects ν magnitude (requires tuning)
   - Grade boundary thresholds (0.8, 0.6, 0.4, 0.2) are domain-specific

3. **Temporal Variance Assumption**
   - Assumes historical std reflects current uncertainty
   - May not capture sudden regime changes

4. **Grade Discretization**
   - 5 grades may lose information
   - Linear interpolation is simplification

### 6.3 Future Enhancements

1. **Dynamic IFS Construction**
   - Adaptive spread_factor based on data characteristics
   - Non-linear membership functions

2. **Advanced ER**
   - Weighted ER with method reliability scores
   - Dempster-Shafer extensions for conflict handling

3. **Distributional ER**
   - Use full score distributions instead of point estimates
   - Bayesian inference for belief combination

---

## References

1. **Atanassov, K.T.** (1986). Intuitionistic fuzzy sets. *Fuzzy Sets and Systems*, 20(1), 87-96.

2. **Yang, J.B., & Xu, D.L.** (2002). On the evidential reasoning algorithm for multiple attribute decision analysis under uncertainty. *IEEE Transactions on Systems, Man, and Cybernetics—Part A*, 32(3), 289-304.

3. **Hwang, C.L., & Yoon, K.** (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer.

4. **Opricovic, S., & Tzeng, G.H.** (2004). Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. *European Journal of Operational Research*, 156(2), 445-455.

5. **Brans, J.P., & Vincke, P.** (1985). A preference ranking organisation method. *Management Science*, 31(6), 647-656.

6. **Zavadskas, E.K., Kaklauskas, A., & Sarka, V.** (1994). The new method of multicriteria complex proportional assessment of projects. *Technological and Economic Development of Economy*, 1(3), 131-139.

7. **Keshavarz Ghorabaee, M., Zavadskas, E.K., Olfat, L., & Turskis, Z.** (2015). Multi-criteria inventory classification using a new method of evaluation based on distance from average solution (EDAS). *Informatica*, 26(3), 435-451.

8. **Fishburn, P.C.** (1967). Additive utilities with incomplete product set: Applications to priorities and assignments. *Operations Research*, 15(3), 537-542.

---

**Document Version:** 1.0  
**Last Updated:** February 14, 2026  
**Status:** Production
