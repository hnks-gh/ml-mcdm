# ML-MCDM Methods Documentation

This document provides detailed descriptions of all methods implemented in the ML-MCDM framework, including mathematical formulations, parameters, and usage guidelines.

---

## Table of Contents

1. [Weight Calculation Methods](#1-weight-calculation-methods)
   - [Entropy Weights](#11-entropy-weights)
   - [CRITIC Weights](#12-critic-weights)
   - [Ensemble Weights](#13-ensemble-weights)
2. [MCDM Methods](#2-mcdm-methods)
   - [TOPSIS](#21-topsis)
   - [Dynamic TOPSIS](#22-dynamic-topsis)
   - [VIKOR](#23-vikor)
   - [Fuzzy TOPSIS](#24-fuzzy-topsis)
3. [Machine Learning Methods](#3-machine-learning-methods)
   - [Panel Regression](#31-panel-regression)
   - [Random Forest with Time-Series CV](#32-random-forest-with-time-series-cv)
   - [LSTM Forecasting](#33-lstm-forecasting)
   - [Rough Set Attribute Reduction](#34-rough-set-attribute-reduction)
4. [Ensemble Methods](#4-ensemble-methods)
   - [Stacking Meta-Learner](#41-stacking-meta-learner)
   - [Borda Count](#42-borda-count)
   - [Copeland Method](#43-copeland-method)
5. [Analysis Methods](#5-analysis-methods)
   - [Convergence Analysis](#51-convergence-analysis)
   - [Sensitivity Analysis](#52-sensitivity-analysis)

---

## 1. Weight Calculation Methods

Weight calculation methods determine the importance of each criterion in the decision-making process. This framework implements objective weighting methods that derive weights from the data itself.

### 1.1 Entropy Weights

**Source:** Shannon (1948), Information Theory

**Purpose:** Assign higher weights to criteria with more variation (information content).

**Mathematical Formulation:**

1. **Normalize the decision matrix** to create proportions:
   $$p_{ij} = \frac{x_{ij}}{\sum_{i=1}^{m} x_{ij}}$$

2. **Calculate entropy** for each criterion:
   $$E_j = -k \sum_{i=1}^{m} p_{ij} \ln(p_{ij})$$
   where $k = \frac{1}{\ln(m)}$ is a normalizing constant

3. **Calculate divergence** (information content):
   $$d_j = 1 - E_j$$

4. **Normalize to obtain weights**:
   $$w_j = \frac{d_j}{\sum_{j=1}^{n} d_j}$$

**Interpretation:**
- Higher entropy → Lower weight (criterion doesn't discriminate well)
- Lower entropy → Higher weight (criterion discriminates alternatives)

**Parameters:**
- `epsilon`: Small constant to prevent log(0), default = 1e-10

**Code Location:** `src/mcdm/weights.py` → `EntropyWeightCalculator`

---

### 1.2 CRITIC Weights

**Source:** Diakoulaki et al. (1995)

**Full Name:** Criteria Importance Through Inter-criteria Correlation

**Purpose:** Consider both the contrast intensity (standard deviation) and conflict (correlation) between criteria.

**Mathematical Formulation:**

1. **Calculate standard deviation** (contrast intensity):
   $$\sigma_j = \sqrt{\frac{1}{m-1} \sum_{i=1}^{m} (x_{ij} - \bar{x}_j)^2}$$

2. **Calculate correlation matrix** between criteria:
   $$r_{jk} = \text{corr}(X_j, X_k)$$

3. **Calculate conflict** for each criterion:
   $$\text{conflict}_j = \sum_{k=1}^{n} (1 - r_{jk})$$

4. **Calculate information content**:
   $$C_j = \sigma_j \times \text{conflict}_j$$

5. **Normalize to obtain weights**:
   $$w_j = \frac{C_j}{\sum_{j=1}^{n} C_j}$$

**Interpretation:**
- High standard deviation → High weight (good discrimination)
- Low correlation with others → High weight (provides unique information)

**Code Location:** `src/mcdm/weights.py` → `CRITICWeightCalculator`

---

### 1.3 Ensemble Weights

**Purpose:** Combine multiple weighting methods for more robust weight estimation.

**Aggregation Methods:**

1. **Arithmetic Mean** (weighted):
   $$w_j^{ensemble} = \sum_{m} \alpha_m \cdot w_j^{(m)}$$
   where $\alpha_m$ is the weight for method $m$

2. **Geometric Mean** (default):
   $$w_j^{ensemble} = \left( \prod_{m} w_j^{(m)} \right)^{1/M}$$

3. **Harmonic Mean**:
   $$w_j^{ensemble} = \frac{M}{\sum_{m} \frac{1}{w_j^{(m)}}}$$

**Default Configuration:** 50% Entropy + 50% CRITIC with geometric mean

**Code Location:** `src/mcdm/weights.py` → `EnsembleWeightCalculator`

---

## 2. MCDM Methods

### 2.1 TOPSIS

**Source:** Hwang & Yoon (1981)

**Full Name:** Technique for Order of Preference by Similarity to Ideal Solution

**Purpose:** Rank alternatives based on their distance to the ideal (best) and anti-ideal (worst) solutions.

**Algorithm Steps:**

1. **Construct the decision matrix** $X = [x_{ij}]_{m \times n}$

2. **Normalize the matrix** (vector normalization):
   $$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{m} x_{ij}^2}}$$

3. **Apply weights**:
   $$v_{ij} = w_j \cdot r_{ij}$$

4. **Determine ideal solutions**:
   - Ideal (best): $A^+ = \{v_1^+, v_2^+, ..., v_n^+\}$
     - For benefit criteria: $v_j^+ = \max_i(v_{ij})$
     - For cost criteria: $v_j^+ = \min_i(v_{ij})$
   - Anti-ideal (worst): $A^- = \{v_1^-, v_2^-, ..., v_n^-\}$
     - For benefit criteria: $v_j^- = \min_i(v_{ij})$
     - For cost criteria: $v_j^- = \max_i(v_{ij})$

5. **Calculate distances**:
   $$D_i^+ = \sqrt{\sum_{j=1}^{n} (v_{ij} - v_j^+)^2}$$
   $$D_i^- = \sqrt{\sum_{j=1}^{n} (v_{ij} - v_j^-)^2}$$

6. **Calculate closeness coefficient**:
   $$C_i = \frac{D_i^-}{D_i^+ + D_i^-}$$

7. **Rank alternatives** by $C_i$ (higher is better)

**Normalization Options:**
- `vector`: Default, preserves relative distances
- `minmax`: Scales to [0, 1] range
- `max`: Divides by maximum value

**Code Location:** `src/mcdm/topsis.py` → `TOPSISCalculator`

---

### 2.2 Dynamic TOPSIS

**Purpose:** Extend TOPSIS for panel data by incorporating temporal dynamics.

**Key Features:**

1. **Temporal Discounting**: Recent years receive higher weights
   $$\delta_t = \gamma^{T-t}$$
   where $\gamma$ is the discount factor (default 0.95)

2. **Trajectory Analysis**: Consider score improvement trends
   $$\text{trajectory}_i = \frac{1}{T-1} \sum_{t=2}^{T} (s_{it} - s_{i,t-1})$$

3. **Stability Score**: Penalize high volatility
   $$\text{stability}_i = 1 - \frac{\sigma_i}{\max(\sigma)}$$

4. **Combined Dynamic Score**:
   $$S_i^{dynamic} = \alpha \cdot \bar{s}_i + \beta \cdot \text{trajectory}_i + \gamma \cdot \text{stability}_i$$

**Parameters:**
- `temporal_discount`: Discount factor (default 0.95)
- `trajectory_weight`: Weight for trajectory score (default 0.3)
- `stability_weight`: Weight for stability score (default 0.2)

**Code Location:** `src/mcdm/topsis.py` → `DynamicTOPSIS`

---

### 2.3 VIKOR

**Source:** Opricovic & Tzeng (2004)

**Full Name:** VIseKriterijumska Optimizacija I Kompromisno Resenje (Serbian: Multicriteria Optimization and Compromise Solution)

**Purpose:** Find compromise solutions that are closest to the ideal, balancing group utility (majority rule) and individual regret (minimum maximum regret).

**Algorithm Steps:**

1. **Determine best ($f_j^*$) and worst ($f_j^-$) values**:
   - Benefit criteria: $f_j^* = \max_i(f_{ij})$, $f_j^- = \min_i(f_{ij})$
   - Cost criteria: $f_j^* = \min_i(f_{ij})$, $f_j^- = \max_i(f_{ij})$

2. **Calculate S (group utility)** - sum of normalized weighted distances:
   $$S_i = \sum_{j=1}^{n} w_j \cdot \frac{f_j^* - f_{ij}}{f_j^* - f_j^-}$$

3. **Calculate R (individual regret)** - maximum weighted distance:
   $$R_i = \max_j \left[ w_j \cdot \frac{f_j^* - f_{ij}}{f_j^* - f_j^-} \right]$$

4. **Calculate Q (compromise index)**:
   $$Q_i = v \cdot \frac{S_i - S^*}{S^- - S^*} + (1-v) \cdot \frac{R_i - R^*}{R^- - R^*}$$
   where $v$ is the weight of group utility strategy (default 0.5)

5. **Rank by Q** (lower is better)

6. **Check acceptance conditions**:
   - **C1 (Acceptable Advantage)**: $Q(A^{(2)}) - Q(A^{(1)}) \geq \frac{1}{m-1}$
   - **C2 (Acceptable Stability)**: $A^{(1)}$ is also best ranked by S or R

**Parameter v Interpretation:**
- $v = 0.5$: Consensus by majority
- $v > 0.5$: Emphasis on group utility
- $v < 0.5$: Emphasis on individual regret

**Outputs:**
- Three ranking lists: by Q (compromise), by S (utility), by R (regret)
- Compromise solution identification
- Acceptance condition verification

**Code Location:** `src/mcdm/vikor.py` → `VIKORCalculator`

---

### 2.4 Fuzzy TOPSIS

**Source:** Chen (2000), based on Zadeh (1965) Fuzzy Set Theory

**Purpose:** Handle uncertainty and imprecision in decision making using triangular fuzzy numbers.

**Triangular Fuzzy Number (TFN):**
A TFN $\tilde{A} = (l, m, u)$ represents:
- $l$: Lower bound (pessimistic)
- $m$: Modal value (most likely)
- $u$: Upper bound (optimistic)

**Fuzzy Number Generation from Panel Data:**
Using temporal variance to capture uncertainty:
$$\tilde{x}_{ij} = (\bar{x}_{ij} - k\sigma_{ij}, \bar{x}_{ij}, \bar{x}_{ij} + k\sigma_{ij})$$
where $k$ is the spread factor (default 1.0)

**Fuzzy Operations:**

1. **Addition**: $(l_1, m_1, u_1) + (l_2, m_2, u_2) = (l_1+l_2, m_1+m_2, u_1+u_2)$

2. **Scalar Multiplication**: $k \cdot (l, m, u) = (kl, km, ku)$ for $k \geq 0$

3. **Vertex Distance**:
   $$d(\tilde{A}, \tilde{B}) = \sqrt{\frac{(l_A-l_B)^2 + (m_A-m_B)^2 + (u_A-u_B)^2}{3}}$$

**Defuzzification Methods:**
- **Centroid** (default): $\frac{l + m + u}{3}$
- **Mean of Maximum**: $m$
- **Bisector**: $\frac{l + 2m + u}{4}$

**Algorithm:**
Similar to standard TOPSIS but using fuzzy arithmetic for all operations.

**Code Location:** `src/mcdm/fuzzy_topsis.py` → `FuzzyTOPSIS`

---

## 3. Machine Learning Methods

### 3.1 Panel Regression

**Purpose:** Analyze relationships between criteria and outcomes while controlling for entity-specific and time-specific effects.

**Model Types:**

#### Fixed Effects (FE)
Controls for time-invariant entity-specific characteristics.

**Model:**
$$y_{it} = \alpha_i + X_{it}\beta + \epsilon_{it}$$

**Within Transformation:**
$$y_{it} - \bar{y}_i = (X_{it} - \bar{X}_i)\beta + (\epsilon_{it} - \bar{\epsilon}_i)$$

**Advantages:**
- Controls for unobserved heterogeneity
- Consistent when entity effects are correlated with regressors

#### Random Effects (RE)
Assumes entity effects are uncorrelated with regressors.

**Model:**
$$y_{it} = \alpha + X_{it}\beta + u_i + \epsilon_{it}$$

**Estimation:** GLS (Generalized Least Squares)

#### Pooled OLS
Ignores panel structure, pools all observations.

$$y_{it} = \alpha + X_{it}\beta + \epsilon_{it}$$

**Parameters:**
- `model_type`: 'fe' (Fixed Effects), 're' (Random Effects), or 'pooled'
- `time_effects`: Include year fixed effects (default True)
- `robust_se`: Use cluster-robust standard errors (default True)

**Outputs:**
- Coefficients with standard errors, t-statistics, p-values
- R-squared (within for FE)
- F-statistic
- Entity fixed effects (for FE)

**Code Location:** `src/ml/panel_regression.py` → `PanelRegression`

---

### 3.2 Random Forest with Time-Series CV

**Purpose:** Feature importance analysis and prediction with proper temporal validation to avoid data leakage.

**Key Features:**

#### Time-Series Cross-Validation
Unlike standard k-fold CV, respects temporal ordering:
```
Fold 1: Train [2020, 2021] → Test [2022]
Fold 2: Train [2020, 2021, 2022] → Test [2023]
Fold 3: Train [2020, 2021, 2022, 2023] → Test [2024]
```

**Random Forest Parameters:**
- `n_estimators`: Number of trees (default 200)
- `max_depth`: Maximum tree depth (default 10)
- `min_samples_split`: Minimum samples for split (default 5)
- `min_samples_leaf`: Minimum samples in leaf (default 2)
- `max_features`: Features per split (default 'sqrt')

**Feature Importance:**
Based on Mean Decrease in Impurity (MDI):
$$\text{importance}_j = \frac{1}{T} \sum_{t=1}^{T} \sum_{\text{node } v \text{ splits on } j} p(v) \cdot \Delta\text{impurity}(v)$$

**Outputs:**
- Feature importance ranking
- CV scores (R², MSE, MAE)
- Rank correlation between actual and predicted rankings
- Test set predictions

**Code Location:** `src/ml/random_forest_ts.py` → `RandomForestTS`

---

### 3.3 LSTM Forecasting

**Purpose:** Predict future scores based on historical trajectories using Long Short-Term Memory neural networks.

**LSTM Architecture:**

LSTMs address the vanishing gradient problem in standard RNNs through gating mechanisms:

1. **Forget Gate**: Decides what information to discard
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2. **Input Gate**: Decides what new information to store
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

3. **Cell State Update**:
   $$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$$

4. **Output Gate**:
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \cdot \tanh(C_t)$$

**Implementation Details:**

The framework includes a simplified LSTM implementation using numpy for environments without PyTorch/TensorFlow:
- Uses temporal feature aggregation (mean, last, trend, std, max, min)
- Gradient descent training with L2 regularization
- Adaptive learning rate decay

**Parameters:**
- `sequence_length`: Input sequence length (default 3)
- `hidden_units`: LSTM hidden size (default 64)
- `epochs`: Training epochs (default 100)
- `learning_rate`: Initial learning rate (default 0.001)
- `patience`: Early stopping patience (default 15)

**Outputs:**
- Predictions for test year
- Training/validation loss history
- Test metrics (MSE, MAE, RMSE)
- Rank correlation

**Code Location:** `src/ml/lstm_forecast.py` → `LSTMForecaster`

---

### 3.4 Rough Set Attribute Reduction

**Source:** Pawlak (1982)

**Purpose:** Identify minimal subsets of attributes (reducts) that preserve the same classification ability as the full attribute set.

**Key Concepts:**

#### Indiscernibility Relation
Objects $x$ and $y$ are indiscernible by attribute set $A$ if they have identical values for all attributes in $A$.

#### Lower Approximation
Objects that can be certainly classified:
$$\underline{A}X = \{x : [x]_A \subseteq X\}$$

#### Upper Approximation
Objects that may possibly belong:
$$\overline{A}X = \{x : [x]_A \cap X \neq \emptyset\}$$

#### Boundary Region
Objects with uncertainty:
$$BN_A(X) = \overline{A}X - \underline{A}X$$

#### Core
Attributes that cannot be removed without changing the classification:
$$\text{CORE}(A) = \bigcap \text{RED}(A)$$

#### Reduct
Minimal attribute subset with the same classification power as full set.

**Quality of Approximation:**
$$\gamma_A(D) = \frac{|\bigcup_{X \in U/D} \underline{A}X|}{|U|}$$

**Algorithm:**

1. **Discretize** continuous attributes into bins
2. **Find Core** attributes (indispensable)
3. **Find Reducts** using heuristic search
4. **Select Best Reduct** meeting quality threshold
5. **Identify Boundary Objects** with classification uncertainty

**Parameters:**
- `quality_threshold`: Minimum acceptable quality (default 0.95)
- `n_bins`: Discretization bins (default 5)
- `max_reducts`: Maximum reducts to find (default 5)
- `method`: 'heuristic' (fast) or 'exhaustive' (complete)

**Outputs:**
- Core attributes (indispensable)
- All reducts found
- Best reduct selection
- Reduction rate
- Quality measures
- Boundary objects

**Code Location:** `src/ml/rough_sets.py` → `RoughSetReducer`

---

## 4. Ensemble Methods

### 4.1 Stacking Meta-Learner

**Purpose:** Combine predictions from multiple base models using a meta-learner for improved accuracy.

**Architecture:**

```
Level 0 (Base Models):
├── TOPSIS Scores
├── VIKOR Q Values
├── Dynamic TOPSIS Scores
├── Fuzzy TOPSIS Scores
└── RF Predictions

Level 1 (Meta-Learner):
└── Ridge Regression → Final Predictions
```

**Meta-Learner Options:**

1. **Ridge Regression** (default):
   $$\hat{\beta} = (X^TX + \alpha I)^{-1}X^Ty$$

2. **Bayesian Ridge**: Iterative hyperparameter optimization

3. **Elastic Net**: Combines L1 and L2 regularization

4. **Linear OLS**: No regularization

**Weight Constraints:**
- Weights constrained to be non-negative
- Normalized to sum to 1 for interpretability

**Parameters:**
- `meta_learner`: 'ridge', 'bayesian', 'elastic', or 'linear'
- `cv_folds`: Cross-validation folds (default 5)
- `alpha`: Regularization strength (default 1.0)
- `use_features`: Include original features (default True)

**Code Location:** `src/ensemble/stacking.py` → `StackingEnsemble`

---

### 4.2 Borda Count

**Purpose:** Aggregate multiple rankings using positional voting.

**Algorithm:**

1. For each alternative $i$ and ranking method $m$:
   $$\text{Borda}_i^m = n - \text{rank}_i^m$$
   where $n$ is the number of alternatives

2. Aggregate:
   $$\text{Borda}_i = \sum_m w_m \cdot \text{Borda}_i^m$$

3. Final ranking by Borda score (higher is better)

**Properties:**
- Simple and intuitive
- Satisfies monotonicity
- May not satisfy Condorcet criterion

**Code Location:** `src/ensemble/aggregation.py` → `BordaCount`

---

### 4.3 Copeland Method

**Purpose:** Aggregate rankings based on pairwise comparisons.

**Algorithm:**

1. For each pair of alternatives $(i, j)$:
   - Count methods where $i$ beats $j$: $W_{ij}$
   - Count methods where $j$ beats $i$: $L_{ij}$

2. Copeland score:
   $$C_i = \sum_{j \neq i} (\text{sign}(W_{ij} - L_{ij}))$$

3. Final ranking by Copeland score

**Properties:**
- Satisfies Condorcet criterion
- Based on majority rule
- More robust to extreme rankings

**Kendall's W (Agreement Coefficient):**
$$W = \frac{12S}{k^2(n^3-n)}$$
where $S$ = sum of squared deviations from mean rank sum

**Code Location:** `src/ensemble/aggregation.py` → `CopelandMethod`

---

## 5. Analysis Methods

### 5.1 Convergence Analysis

**Purpose:** Test whether entities are converging (becoming more similar) over time.

#### Beta (β) Convergence

Tests if entities with lower initial scores grow faster (catch-up effect).

**Regression:**
$$\frac{1}{T} \ln\left(\frac{y_{i,T}}{y_{i,0}}\right) = \alpha + \beta \ln(y_{i,0}) + \epsilon_i$$

**Interpretation:**
- $\beta < 0$ and significant → Convergence (poor performers catching up)
- $\beta \geq 0$ → No convergence or divergence

**Convergence Speed:**
$$\lambda = -\frac{\ln(1+\beta)}{T}$$

**Half-Life** (years to halve the gap):
$$\tau = \frac{\ln(2)}{\lambda}$$

#### Sigma (σ) Convergence

Tests if the dispersion of scores is decreasing over time.

**Measure:**
$$\sigma_t = \sqrt{\frac{1}{n-1} \sum_i (y_{it} - \bar{y}_t)^2}$$

**Test:** Linear trend of $\sigma_t$ over time
- Negative trend → Sigma convergence
- Positive trend → Sigma divergence

#### Club Convergence

Tests if entities converge to different steady states (multiple equilibria).

**Code Location:** `src/analysis/convergence.py` → `ConvergenceAnalysis`

---

### 5.2 Sensitivity Analysis

**Purpose:** Test the robustness of rankings to changes in weights and input uncertainty.

**Methods:**

#### Weight Sensitivity Index
Measures how sensitive rankings are to changes in each criterion's weight:

$$\text{Sensitivity}_j = \frac{1}{N} \sum_{i=1}^{N} |\text{rank}_i^{original} - \text{rank}_i^{w_j \pm \delta}|$$

#### Monte Carlo Perturbation
1. Generate $N$ perturbed weight vectors (default 1000)
2. For each perturbation, calculate new rankings
3. Analyze ranking distribution

$$w_j^{perturbed} = w_j \cdot (1 + \epsilon_j)$$
where $\epsilon_j \sim U(-\delta, \delta)$

#### Critical Weight Ranges
Find weight boundaries that maintain the same ranking:

$$[w_j^{low}, w_j^{high}]$$ such that $\text{rank}_1 = \text{rank}_1^{original}$

#### Top-N Stability
Percentage of simulations where top N alternatives remain unchanged.

**Parameters:**
- `n_simulations`: Monte Carlo iterations (default 1000)
- `perturbation_range`: Maximum perturbation (default 0.2 = ±20%)
- `seed`: Random seed for reproducibility

**Outputs:**
- Weight sensitivity indices per criterion
- Rank stability per alternative
- Critical weight ranges
- Overall robustness score (0-1)
- Top-N stability metrics

**Code Location:** `src/analysis/sensitivity.py` → `SensitivityAnalysis`

---

## References

1. Chen, C. T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy environment. *Fuzzy Sets and Systems*, 114(1), 1-9.

2. Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining objective weights in multiple criteria problems: The CRITIC method. *Computers & Operations Research*, 22(7), 763-770.

3. Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer-Verlag.

4. Opricovic, S., & Tzeng, G. H. (2004). Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. *European Journal of Operational Research*, 156(2), 445-455.

5. Pawlak, Z. (1982). Rough sets. *International Journal of Computer & Information Sciences*, 11(5), 341-356.

6. Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.

7. Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338-353.
