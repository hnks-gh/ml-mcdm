# Ranking Methodology: Hierarchical MCDM with Evidential Reasoning

## Overview

This framework implements a **two-stage hierarchical ranking system** that combines:

1. **Six Traditional MCDM Methods** - TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW - applied
   independently within each criterion group to generate per-method scores.
2. **Evidential Reasoning (ER)** - Yang & Xu (2002) analytical algorithm for rigorous
   belief-based aggregation of the six method scores into a single ranking per criterion,
   and then across all eight criteria into a final global ranking.

**Application:** Vietnam PAPI - 63 provinces, 8 criteria (C01-C08), 29 sub-criteria
(SC11-SC83), 14 years (2011-2024).

---

## Part I: MCDM Methods

All six methods operate on the min-max normalized sub-criteria matrix in [0,1].

### 1.1 TOPSIS (Hwang & Yoon, 1981)

1. Vector-normalize: r_ij = x_ij / sqrt(sum_i x_ij^2)
2. Weighted normalized matrix: v_ij = w_j * r_ij
3. Ideal solutions: A+_j = max_i v_ij, A-_j = min_i v_ij (benefit)
4. Closeness coefficient: C_i = d-_i / (d+_i + d-_i) -- higher is better

### 1.2 VIKOR (Opricovic & Tzeng, 2004)

S_i = sum_j w_j*(f*_j - x_ij)/(f*_j - f-_j), R_i = max_j [w_j*(f*_j-x_ij)/(f*_j-f-_j)]
Q_i = v*(S_i-S*)/(S--S*) + (1-v)*(R_i-R*)/(R--R*), v=0.5 -- lower is better (inverted before ER)

### 1.3 PROMETHEE II (Brans & Vincke, 1985)

V-shape preference function (p=0.3, q=0.1): P_j(a,b) = max(0, (d-q)/(p-q)), capped at 1
Net flow: Phi_net(a) = [sum_{b!=a} pi(a,b) - sum_{b!=a} pi(b,a)] / (m-1) -- higher is better

### 1.4 COPRAS (Zavadskas et al., 1994)

S+_i = sum benefit criteria (weighted normalized), S-_i = sum cost criteria
Q_i = S+_i + (sum_k S-_k) / (S-_i * sum_k 1/S-_k) -- utility degree U_i = Q_i/Q_max

### 1.5 EDAS (Keshavarz Ghorabaee et al., 2015)

AV_j = mean(x_ij); PDA_ij = max(0, x_ij-AV_j)/AV_j; NDA_ij = max(0, AV_j-x_ij)/AV_j
AS_i = 0.5*(SP_i/max(SP) + 1 - SN_i/max(SN)) -- higher is better

### 1.6 SAW — Simple Additive Weighting (Fishburn, 1967)

Score_i = sum_j w_j * r_ij -- weighted sum of min-max normalized values
Serves as a transparent linear baseline; fully compensatory.
Higher score is better.

---

## Part II: Method Weighting (Rank Agreement)

`HierarchicalEvidentialReasoning._compute_method_weights()` uses inverse-CV weighting:
- Compute rank vectors for all 6 methods
- w_m proportional to 1/(CV_m + epsilon), then normalize

Kendall's W = 12*sum_i(R_i-R_bar)^2 / [k^2*(m^3-m)], k=6, m=63 -- stored in kendall_w

---

## Part III: Evidential Reasoning (ER)

**Reference:** Yang & Xu (2002), IEEE Trans. SMC-A, 32(3), 289-304.

### Grade Utilities

| Grade     | Utility |
|-----------|---------|
| Excellent | 1.00    |
| Good      | 0.75    |
| Fair      | 0.50    |
| Poor      | 0.25    |
| Bad       | 0.00    |

### Score-to-Belief

score_to_belief(s): linear interpolation between the two adjacent grades bracketing s.
Example: s=0.65 -> beta_Good=0.60, beta_Fair=0.40

### ER Combination (Yang & Xu Eq. 9)

A_{n,i} = w_i*beta_{n,i} + 1 - w_i*sum_j beta_{j,i}
B_i = 1 - w_i*sum_j beta_{j,i}; C_i = 1 - w_i

K = 1 / [sum_n prod_i A_{n,i} - (N-1)*prod_i B_i]
m_hat(H_n) = K*[prod_i A_{n,i} - prod_i B_i]
m_tilde(H) = K*[prod_i B_i - prod_i C_i]
beta_n_final = m_hat(H_n) / (1 - m_tilde(H))

### Expected Utility

u_min = sum_n beta_n*u_n + beta_H*0   (unassigned mass -> worst)
u_max = sum_n beta_n*u_n + beta_H*1   (unassigned mass -> best)
Score = (u_min + u_max) / 2

---

## Part IV: Two-Stage Architecture

**Stage 1 (x8 criterion groups):**
1. Min-max normalize SC matrix; apply YearContext SC exclusions.
2. Run 6 MCDM methods with Level-1 SC weights from CRITICWeightCalculator.
3. Derive method weights via inverse-CV.
4. Convert each method score to belief; ER-combine -> one belief per (province, criterion).
5. Average utility -> criterion-level score in [0,1].

**Stage 2 (global):**
1. Criterion weights from CRITICWeightCalculator Level 2.
2. ER-combine 8 criterion beliefs -> global belief per province.
3. Average utility -> global score -> final ranking.

---

## Part V: Adaptive Data Handling

- Constant column (range < 1e-12): all values set to 0.5.
- Partial NaN cells: filled with 0.5 after normalization.
- Inactive sub-criteria: dropped per YearContext before normalization.

---

## Part VI: Key Files

| File | Role |
|---|---|
| `ranking/evidential_reasoning/base.py` | BeliefDistribution, EvidentialReasoningEngine, score_to_belief() |
| `ranking/evidential_reasoning/hierarchical_er.py` | HierarchicalEvidentialReasoning, Kendall W |
| `ranking/hierarchical_pipeline.py` | HierarchicalRankingPipeline |
| `ranking/topsis.py` | TOPSISCalculator |
| `ranking/vikor.py` | VIKORCalculator |
| `ranking/promethee.py` | PROMETHEECalculator |
| `ranking/copras.py` | COPRASCalculator |
| `ranking/edas.py` | EDASCalculator |
| `ranking/saw.py` | SAWCalculator |

---

## Part VII: Result Structure

| Field | Type | Description |
|---|---|---|
| `final_ranking` | pd.Series | Province -> rank (1 = best) |
| `final_scores` | pd.Series | Province -> average utility in [0,1] |
| `kendall_w` | float | Inter-method concordance in [0,1] |
| `uncertainty` | Dict[str, float] | Per-province belief entropy |
| `criterion_beliefs` | Dict[str, Dict] | Per-criterion belief distributions |
| `method_weights` | Dict[str, float] | Per-method trust weights |

---

## References

1. Yang, J.B., & Xu, D.L. (2002). IEEE Trans. SMC-A, 32(3), 289-304.
2. Hwang, C.L., & Yoon, K. (1981). Multiple Attribute Decision Making. Springer-Verlag.
3. Opricovic, S., & Tzeng, G.H. (2004). EJOR, 156(2), 445-455.
4. Brans, J.P., & Vincke, P. (1985). Management Science, 31(6), 647-656.
5. Zavadskas, E.K. et al. (1994). Tech. & Econ. Dev. of Economy, 1(3), 131-139.
6. Keshavarz Ghorabaee, M. et al. (2015). Informatica, 26(3), 435-451.
7. Kendall, M.G., & Babington Smith, B. (1939). Ann. Math. Stat., 10(3), 275-287.
8. Fishburn, P.C. (1967). Operations Research, 15(3), 537–542. (SAW)
9. MacCrimmon, K.R. (1968). RAND Corporation, RM-4823-ARPA. (SAW)

---

**Last Updated:** March 2026
**Status:** Production
