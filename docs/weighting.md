# Weight Calculation: Detailed Workflow

## Overview

The objective weight calculation system uses a **Game Theory Weight Combination (GTWC)** approach that combines four complementary weighting methods through intra-group hybridization and cooperative game optimization, validated via Bayesian Bootstrap and temporal stability verification.

**Key Principle:** Operate on the entire panel dataset (all entities × all time periods) simultaneously to preserve temporal trends and structural relationships.

---

## System Architecture

```
Input: Panel Data (N observations × p criteria)
  ↓
Step 1: Global Normalization (preserve temporal trends)
  ↓
Step 2: Calculate 4 Independent Base Weight Vectors
  ├── W_entropy (information-theoretic dispersion)
  ├── W_sd (variance-based)
  ├── W_critic (contrast + correlation)
  └── W_merec (impact-based via removal)
  ↓
Step 3: Game Theory Weight Combination (GTWC)
  │
  ├── Phase 2: Intra-Group Hybridization
  │   ├── Group A (Dispersion Camp): Geometric Mean(W_entropy, W_sd)
  │   └── Group B (Interaction Camp): Harmonic Mean(W_critic, W_merec)
  │
  ├── Phase 3: Cooperative Game Optimization
  │   ├── Construct L2-norm objective function
  │   ├── Solve via Lagrange multipliers for α coefficients
  │   └── Normalize: α₁ + α₂ = 1
  │
  └── Phase 4: Final Aggregation
      └── W* = α₁·W_GroupA + α₂·W_GroupB
  ↓
Step 4: Bayesian Bootstrap (uncertainty quantification)
  ├── Dirichlet resampling (999 iterations)
  ├── Re-compute Steps 2-3 on each sample
  └── Calculate posterior statistics
  ↓
Step 5: Temporal Stability Verification
  ├── Split-half analysis
  ├── Cosine similarity + Spearman correlation
  └── Validate structural stability
  ↓
Output: Final Weights with Uncertainty Bounds
```

---

## Detailed Step-by-Step Workflow

### **Step 1: Global Min-Max Normalization**

**Purpose:** Normalize all criteria to [0,1] scale while preserving temporal trends across the entire time series.

**Method:**
```python
x_normalized = (x - min_global) / (max_global - min_global) + ε
```

**Key Points:**
- **Global normalization** uses min/max across ALL time periods (not per-year)
- Preserves growth/decline patterns over the 14-year span
- Epsilon shift (ε = 1e-10) ensures strictly positive values for logarithmic operations
- Applied to entire N × p matrix: (896 observations × 29 criteria)

**Why Global?**
- Year-by-year normalization would erase temporal trends
- MCDM forecasting requires comparability across time
- Entropy calculation needs to see how criteria evolved

**Implementation:**
```python
def _global_min_max_normalize(self, X: np.ndarray) -> np.ndarray:
    """Normalize each column to [0,1] using global min/max."""
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    X_norm = (X - min_vals) / (max_vals - min_vals + self.epsilon)
    return X_norm + self.epsilon
```

---

### **Step 2: Calculate Four Independent Weight Vectors**

Each method captures different aspects of criterion importance.

#### **2.1 Entropy Weights**

**Reference:** Shannon (1948), Bell System Technical Journal

**Principle:** Criteria with more variation (higher information content) should receive higher weights.

**Formula:**
```
1. Convert to proportions:    p_ij = x_ij / Σ_i x_ij

2. Calculate entropy:          E_j = -k × Σ_i (p_ij × ln(p_ij))
                               where k = 1/ln(m), m = number of alternatives

3. Convert to weights:         D_j = 1 - E_j  (divergence coefficient)
                               w_j = D_j / Σ_k D_k
```

**Interpretation:**
- High entropy → uniform distribution → low importance
- Low entropy → concentrated distribution → high importance
- Measures information content from probability distribution

**Implementation:**
```python
class EntropyWeightCalculator:
    def calculate(self, data: pd.DataFrame) -> WeightResult:
        # Normalize columns to proportions
        proportions = data / data.sum(axis=0)
        
        # Calculate entropy
        k = 1.0 / np.log(len(data))
        entropy = -k * (proportions * np.log(proportions + self.epsilon)).sum(axis=0)
        
        # Convert to weights
        divergence = 1 - entropy
        weights = divergence / divergence.sum()
        
        return WeightResult(weights=weights, method='entropy', details={...})
```

---

#### **2.2 CRITIC Weights**

**Reference:** Diakoulaki et al. (1995), Computers & Operations Research

**Principle:** Important criteria should have high variation (contrast intensity) AND low correlation with other criteria (conflict).

**Formula:**
```
1. Calculate standard deviation:   σ_j = std(x_j)  (contrast intensity)

2. Calculate correlation matrix:   R = corr(X)  (Pearson correlation)

3. Calculate conflict measure:     f_j = Σ_k (1 - r_jk)

4. Information content:             C_j = σ_j × f_j

5. Normalize to weights:            w_j = C_j / Σ_k C_k
```

**Interpretation:**
- σ_j: How much the criterion varies (absolute dispersion)
- f_j: How independent the criterion is from others
- High weight → high variation AND low redundancy
- Handles multicollinearity by down-weighting correlated criteria

**Implementation:**
```python
class CRITICWeightCalculator:
    def calculate(self, data: pd.DataFrame) -> WeightResult:
        # Calculate standard deviation (contrast intensity)
        std = data.std(ddof=1)
        
        # Calculate correlation matrix
        corr = data.corr()
        
        # Calculate conflict (sum of 1 - correlation)
        conflict = (1 - corr).sum(axis=0)
        
        # Information content = contrast × conflict
        info_content = std * conflict
        
        # Normalize to weights
        weights = info_content / info_content.sum()
        
        return WeightResult(weights=weights, method='critic', details={...})
```

---

#### **2.3 MEREC Weights**

**Reference:** Keshavarz-Ghorabaee et al. (2021), Symmetry, 13(4), 525

**Principle:** Important criteria are those whose removal causes the largest change in overall performance rankings.

**Formula:**
```
1. Normalize to [0,1]:           x'_ij = (x_ij - min_j) / (max_j - min_j) + ε

2. Calculate overall performance: S_i = ln(1 + (1/n) Σ_j |ln(x'_ij)|)

3. For each criterion j:
   a. Set x'_ij = 1 (remove criterion effect)
   b. Recalculate S_i^j

4. Calculate removal effect:     E_j = Σ_i |ln(S_i^j) - ln(S_i)|

5. Normalize to weights:          w_j = E_j / Σ_k E_k
```

**Interpretation:**
- Direct measure of criterion impact on decision outcomes
- Removal effect quantifies importance via counterfactual analysis
- MCDM-native: evaluates actual decision consequences
- Higher removal effect → more influential criterion

**Why MEREC?**
- Addresses a key weakness of Entropy/CRITIC: they measure statistical properties but not decision impact
- Aligns with MCDM goal: identify criteria that actually affect rankings
- Non-compensatory: each criterion evaluated independently

**Implementation:**
```python
class MERECWeightCalculator:
    def calculate(self, data: pd.DataFrame) -> WeightResult:
        # Normalize data
        X_norm = self._normalize(data.values)
        
        # Calculate baseline performance
        S_baseline = self._calculate_performance(X_norm)
        
        # Calculate removal effect for each criterion
        removal_effects = []
        for j in range(X_norm.shape[1]):
            # Remove criterion j by setting to 1
            X_removed = X_norm.copy()
            X_removed[:, j] = 1.0
            
            # Recalculate performance
            S_removed = self._calculate_performance(X_removed)
            
            # Measure effect
            effect = np.abs(np.log(S_removed + self.epsilon) - 
                          np.log(S_baseline + self.epsilon)).sum()
            removal_effects.append(effect)
        
        # Normalize to weights
        removal_effects = np.array(removal_effects)
        weights = removal_effects / removal_effects.sum()
        
        return WeightResult(weights=weights, method='merec', details={...})
```

---

#### **2.4 Standard Deviation Weights**

**Reference:** Wang & Luo (2010), Mathematical and Computer Modelling

**Principle:** Criteria with higher variance contain more information and should receive higher weights.

**Formula:**
```
1. Calculate standard deviation:  σ_j = sqrt(Σ_i (x_ij - μ_j)² / (n-1))

2. Normalize to weights:           w_j = σ_j / Σ_k σ_k
```

**Interpretation:**
- Simplest variance-based method
- Direct measure: more variation = more discriminative power
- Fast computation: O(mn) time complexity
- Baseline for comparison with more complex methods

**Implementation:**
```python
class StandardDeviationWeightCalculator:
    def calculate(self, data: pd.DataFrame) -> WeightResult:
        # Calculate standard deviation
        std = data.std(ddof=1)
        
        # Normalize to weights
        weights = std / std.sum()
        
        # Additional statistics
        cv = std / data.mean()  # Coefficient of variation
        ranges = data.max() - data.min()
        
        return WeightResult(
            weights=weights,
            method='std_dev',
            details={
                'std': std,
                'cv': cv,
                'range': ranges
            }
        )
```

---

### **Step 3: Game Theory Weight Combination (GTWC)**

**References:** Nash (1950), Econometrica; Ding & Shi (2005), European Journal of OR; Shapley (1953), Contributions to the Theory of Games

**Purpose:** Combine the four weight vectors through principled intra-group hybridization and cooperative game optimization that prevents variance bias and achieves Nash equilibrium.

**Why GTWC is better than simple averaging or reliability-weighting:**
- **Prevents Variance Bias:** Groups redundant methods to avoid over-counting similar signals
- **Theoretically Optimal:** Game-theoretic equilibrium rather than heuristic scoring
- **Robust:** Intra-group operators (geometric/harmonic mean) prevent zero-dominance

#### **Phase 2: Intra-Group Hybridization**

The four methods are clustered into two thematic "super-weights" by measurement philosophy:

**Group A — Dispersion Camp (Entropy + Standard Deviation):**
- Both focus on *within-criterion* variance
- Aggregation: **Geometric Mean** — amplifies shared signals, prevents zero-dominance
```
W_GroupA = Normalize( √(W_entropy × W_sd) )
```

*Why geometric mean instead of raw product?*
- Raw product `W_e × W_sd` would destroy criterion importance if either method assigns near-zero
- `√(W_e × W_sd)` is more balanced while still amplifying agreement

**Group B — Interaction Camp (CRITIC + MEREC):**
- Both focus on *between-criterion* relationships and structural impact
- Aggregation: **Harmonic Mean** — conservative, requires agreement from both methods
```
W_GroupB = Normalize( 2 / (1/W_critic + 1/W_merec) )
```

*Why harmonic mean?*
- A criterion is only favored if BOTH methods agree on its importance
- A low weight from either method pulls the result down (conservative)

#### **Phase 3: Game Theory Optimization**

Treat W_GroupA and W_GroupB as players in a cooperative game. Find optimal coefficients (α₁, α₂) that minimize the combined L2-distance to both groups:

**Objective Function:**
```
Minimize: L = ‖α₁·W_A + α₂·W_B − W_A‖² + ‖α₁·W_A + α₂·W_B − W_B‖²
```

**Solution via Lagrange Multipliers:**
```
Solve the system: A · α = b

Where:
A = ⎡ W_A·W_A   W_A·W_B ⎤    b = ⎡ W_A·W_A ⎤
    ⎣ W_B·W_A   W_B·W_B ⎦        ⎣ W_B·W_B ⎦

(· denotes dot product, yielding scalar values)
```

**Normalize coefficients:**
```
α_final_i = max(0, α_i) / Σ max(0, α_k)
```

#### **Phase 4: Final Aggregation**

```
W* = α₁ · W_GroupA + α₂ · W_GroupB
```

**Implementation:**
```python
class GameTheoryWeightCombination:
    def combine(self, weight_vectors: Dict[str, np.ndarray]) -> tuple:
        W_entropy = weight_vectors['entropy']
        W_sd = weight_vectors['std_dev']
        W_critic = weight_vectors['critic']
        W_merec = weight_vectors['merec']
        
        # Phase 2: Intra-Group Hybridization
        W_GroupA = normalize(np.sqrt(W_entropy * W_sd))
        W_GroupB = normalize(2.0 / (1.0/W_critic + 1.0/W_merec))
        
        # Phase 3: Game Theory Optimization
        A = [[np.dot(W_A, W_A), np.dot(W_A, W_B)],
             [np.dot(W_B, W_A), np.dot(W_B, W_B)]]
        b = [np.dot(W_A, W_A), np.dot(W_B, W_B)]
        alpha = np.linalg.solve(A, b)
        alpha = np.clip(alpha, 0, None)
        alpha = alpha / alpha.sum()
        
        # Phase 4: Final Aggregation
        W_final = alpha[0] * W_GroupA + alpha[1] * W_GroupB
        return normalize(W_final), details
```

---

### **Step 4: Bayesian Bootstrap**

**Reference:** Rubin (1981), The Annals of Statistics

**Purpose:** Quantify uncertainty in weight estimates via non-parametric Bayesian resampling.

**Why Bayesian Bootstrap (not standard bootstrap)?**
- Standard bootstrap: discrete uniform resampling (weights observations equally)
- Bayesian bootstrap: continuous Dirichlet weights (more efficient, theoretically principled)
- Dirichlet(1,...,1) is the non-informative prior
- More efficient for smooth statistics like weighted means

#### **Algorithm**

```
For each iteration b = 1, ..., B (default B=999):

  1. Draw Dirichlet weights:
     g_i ~ Exponential(1) for i = 1, ..., N
     w_i = g_i / Σ_k g_k
  
  2. Create weighted sample:
     Resample N rows with probabilities p = w
  
  3. Compute weights on bootstrap sample:
     a. Normalize data
     b. Calculate Entropy, CRITIC, MEREC, SD weights
     c. Fuse via reliability-weighted fusion
     d. Store fused weight vector
  
  4. Accumulate results

After B iterations:
  - Mean weights: w̄_j = (1/B) Σ_b w_j^(b)
  - Standard deviation: σ_j = sqrt((1/(B-1)) Σ_b (w_j^(b) - w̄_j)²)
  - 95% CI: [Q_0.025(w_j), Q_0.975(w_j)]
```

**Why B=999?**
- Odd number avoids interpolation at 2.5th and 97.5th percentiles
- Standard practice for percentile-based credible intervals (Davison & Hinkley, 1997)
- Provides stable posterior statistics

**Implementation:**
```python
def _bayesian_bootstrap(
    self,
    X_norm: np.ndarray,
    criteria_cols: List[str]
) -> Dict:
    
    N, p = X_norm.shape
    B = self.bootstrap_iterations
    
    # Storage for bootstrap samples
    all_weights = np.zeros((B, p))
    
    rng = np.random.RandomState(self.seed)
    
    for b in range(B):
        # 1. Draw Dirichlet weights (via exponential trick)
        g = rng.exponential(1.0, size=N)
        dirichlet_weights = g / g.sum()
        
        # 2. Resample observations
        indices = rng.choice(N, size=N, replace=True, p=dirichlet_weights)
        X_boot = X_norm[indices, :]
        
        # 3. Compute weights on bootstrap sample
        X_df = pd.DataFrame(X_boot, columns=criteria_cols)
        
        W_e = np.array([self.entropy_calc.calculate(X_df).weights[c] 
                       for c in criteria_cols])
        W_c = np.array([self.critic_calc.calculate(X_df).weights[c] 
                       for c in criteria_cols])
        W_m = np.array([self.merec_calc.calculate(X_df).weights[c] 
                       for c in criteria_cols])
        W_s = np.array([self.sd_calc.calculate(X_df).weights[c] 
                       for c in criteria_cols])
        
        weight_vectors = {'entropy': W_e, 'critic': W_c, 
                         'merec': W_m, 'std_dev': W_s}
        
        W_fused, _ = self.gtwc.combine(weight_vectors)
        
        all_weights[b, :] = W_fused
    
    # Calculate posterior statistics
    mean_weights = all_weights.mean(axis=0)
    std_weights = all_weights.std(axis=0, ddof=1)
    ci_lower = np.percentile(all_weights, 2.5, axis=0)
    ci_upper = np.percentile(all_weights, 97.5, axis=0)
    
    return {
        'mean_weights': mean_weights,
        'std_weights': std_weights,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'all_weights': all_weights
    }
```

---

### **Step 5: Temporal Stability Verification**

**Purpose:** Verify that weights are structurally stable over time (not dependent on which years are included).

**Why this matters:**
- Weights will be used for future predictions
- Need confidence that relationships are structural, not ephemeral
- Validates that patterns aren't driven by outlier years

#### **Split-Half Analysis**

```
1. Split data at median time point:
   - First half: years 1 to T/2
   - Second half: years T/2+1 to T

2. Compute weights on each half:
   - Run full pipeline (Steps 2-3) on first half → W_first
   - Run full pipeline (Steps 2-3) on second half → W_second

3. Measure agreement:
   a. Cosine similarity:
      cos(W_first, W_second) = (W_first · W_second) / (||W_first|| × ||W_second||)
   
   b. Spearman rank correlation:
      ρ(rank(W_first), rank(W_second))

4. Stability criterion:
   is_stable = (cosine_similarity ≥ threshold)  # default threshold = 0.95
```

**Interpretation:**
- **Cosine ≥ 0.95:** Weight magnitudes are highly consistent
- **Spearman ≈ 1.0:** Rank ordering of criteria is preserved
- **High stability:** Weights capture structural relationships
- **Low stability:** May indicate temporal non-stationarity

**Implementation:**
```python
def _stability_verification(
    self,
    X_raw: np.ndarray,
    time_values: np.ndarray,
    criteria_cols: List[str]
) -> Dict:
    
    # Find median time point
    unique_times = np.sort(np.unique(time_values))
    split_point = len(unique_times) // 2
    split_time = unique_times[split_point]
    
    # Split data
    first_half = X_raw[time_values <= split_time]
    second_half = X_raw[time_values > split_time]
    
    # Compute weights on each half
    W_first = self._compute_fused_weights(
        self._global_min_max_normalize(first_half),
        criteria_cols
    )
    W_second = self._compute_fused_weights(
        self._global_min_max_normalize(second_half),
        criteria_cols
    )
    
    # Cosine similarity
    cos_sim = np.dot(W_first, W_second) / (
        np.linalg.norm(W_first) * np.linalg.norm(W_second)
    )
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(W_first, W_second)
    
    return {
        'cosine_similarity': float(cos_sim),
        'spearman_correlation': float(spearman_r),
        'spearman_pvalue': float(spearman_p),
        'is_stable': cos_sim >= self.stability_threshold,
        'split_point': int(split_point)
    }
```

---

## Output Structure

The `calculate()` method returns a `WeightResult` object with:

```python
@dataclass
class WeightResult:
    weights: Dict[str, float]  # Final weights (posterior mean from bootstrap)
    method: str                # "robust_global_hybrid"
    details: Dict              # Comprehensive statistics
```

### Details Dictionary Structure

```python
details = {
    # Individual method weights
    'individual_weights': {
        'entropy': {criterion: weight, ...},
        'critic': {criterion: weight, ...},
        'merec': {criterion: weight, ...},
        'std_dev': {criterion: weight, ...},
        'fused': {criterion: weight, ...}  # Pre-bootstrap fusion
    },
    
    # Fusion information
    'fusion': {
        'reliability_scores': {
            'entropy': float,
            'critic': float,
            'merec': float,
            'std_dev': float
        },
        'method_weights': {  # Alpha values
            'entropy': float,
            'critic': float,
            'merec': float,
            'std_dev': float
        },
        'fusion_method': 'reliability_weighted'
    },
    
    # Bootstrap statistics
    'bootstrap': {
        'iterations': 999,
        'mean_weights': {criterion: float, ...},  # Final output
        'std_weights': {criterion: float, ...},   # Uncertainty
        'ci_lower_2_5': {criterion: float, ...},  # 95% CI lower
        'ci_upper_97_5': {criterion: float, ...}, # 95% CI upper
    },
    
    # Stability verification
    'stability': {
        'cosine_similarity': float,      # Target ≥ 0.95
        'spearman_correlation': float,   # Rank consistency
        'spearman_pvalue': float,
        'is_stable': bool,
        'split_point': int
    },
    
    # Metadata
    'n_observations': int,
    'n_criteria': int
}
```

---

## Computational Complexity

**Single weight calculation (no bootstrap):**
- Normalization: O(N × p)
- Entropy: O(N × p)
- CRITIC: O(N × p + p²) - correlation matrix
- MEREC: O(N × p²) - removal effects
- Standard Deviation: O(N × p)
- Fusion: O(p × m) where m=4 methods
- **Total: O(N × p²)** dominated by MEREC and CRITIC

**With Bayesian Bootstrap (B iterations):**
- **Total: O(B × N × p²)**
- For B=999, N=896, p=29: ~75M operations
- **Runtime:** 2-3 minutes on modern CPU

**Stability verification:**
- Computes weights twice on half-sized data
- **Additional cost: ~2× single calculation**

---

## Configuration Parameters

```python
@dataclass
class WeightingConfig:
    # Bayesian Bootstrap
    bootstrap_iterations: int = 999
    # Odd number → no interpolation at percentiles
    # 999 is standard for 95% CIs
    
    # Stability verification
    stability_threshold: float = 0.95
    # Cosine similarity threshold for stability
    # 0.95 indicates high structural consistency
    
    # Numerical stability
    epsilon: float = 1e-10
    # Small constant for log(0) and division safety
    
    # Reproducibility
    seed: int = 42
    # Random seed for bootstrap sampling
```

---

## Interpreting Results

### **High Reliability Score → Trust this method**
- Reliability > 0.35: Method is performing well on this data
- Reliability < 0.25: Method may be unreliable or uninformative

### **Wide Confidence Intervals → High Uncertainty**
- CI width > 0.01: Substantial uncertainty in this criterion's weight
- Narrow CIs: High confidence in weight estimate

### **Cosine < 0.95 → Investigate Temporal Instability**
- May indicate structural changes over time
- Consider time-specific weights or sub-period analysis
- Check for outlier years or regime shifts

### **Spearman ≈ 1.0 → Rank-Stable**
- Criterion importance ordering is consistent
- Even if magnitudes vary, relative importance preserved

---

---

## Troubleshooting

### **High Bootstrap Variance**
**Symptoms:** Wide CIs, high std/mean ratios  
**Causes:**
- Small sample size (N < 30)
- High disagreement between methods
- Temporal instability

**Solutions:**
- Increase data if possible
- Investigate outliers
- Check temporal trends
- Consider reducing bootstrap iterations for faster testing

### **Low Stability Score**
**Symptoms:** Cosine < 0.95  
**Causes:**
- Structural changes over time
- Non-stationary criteria
- Different data characteristics in time periods

**Solutions:**
- Split analysis by time blocks
- Time-varying weights
- Investigate temporal dynamics
- Check for regime shifts

### **Method Disagreement**
**Symptoms:** Large differences in individual weights  
**Causes:**
- Methods capture different aspects (normal)
- Data characteristics favor specific methods
- One method inappropriate for data type

**Solutions:**
- Check reliability scores
- Understand why disagreement occurs
- May indicate multi-dimensional problem
- Consider method-specific analysis

### **All Methods Similar Reliability**
**Symptoms:** Reliability scores within 0.01  
**Interpretation:**
- All methods equally valid (good!)
- No clear "best" method for this data
- Fusion will approximate equal weighting

**Action:** No action needed - this is a valid outcome

---

## References

### Core Methods
1. Shannon, C.E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379-423.
2. Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining objective weights in multiple criteria problems: The CRITIC method. *Computers & Operations Research*, 22(7), 763-770.
3. Keshavarz-Ghorabaee, M., Amiri, M., Zavadskas, E.K., Turskis, Z., & Antucheviciene, J. (2021). Determination of Objective Weights Using a New Method Based on the Removal Effects of Criteria (MEREC). *Symmetry*, 13(4), 525.
4. Wang, Y.M., & Luo, Y. (2010). Integration of correlations with standard deviations for determining attribute weights in multiple attribute decision making. *Mathematical and Computer Modelling*, 51(1-2), 1-12.

### Fusion & Bootstrap
5. Yager, R.R. (1988). On ordered weighted averaging aggregation operators in multicriteria decision making. *IEEE Transactions on Systems, Man, and Cybernetics*, 18(1), 183-190.
6. Hoeting, J.A., Madigan, D., Raftery, A.E., & Volinsky, C.T. (1999). Bayesian model averaging: a tutorial. *Statistical Science*, 14(4), 382-401.
7. Rubin, D.B. (1981). The Bayesian Bootstrap. *The Annals of Statistics*, 9(1), 130-134.
8. Davison, A.C. & Hinkley, D.V. (1997). *Bootstrap Methods and Their Application*. Cambridge University Press.

---
