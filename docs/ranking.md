# Ranking Methodology: Hierarchical MCDM

## Overview

> **Note:** The ranking phase runs 5 Traditional MCDM methods (TOPSIS, VIKOR, PROMETHEE II, COPRAS, EDAS) plus a **Raw Sum Baseline** and reports their individual scores.

This framework implements a **two-stage hierarchical ranking system** that combines:

1. **Five Traditional MCDM Methods** - TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS - applied
   independently within each criterion group to generate per-method scores.
2. **Raw Sum Baseline** - A transparent additive baseline calculating the sum of raw sub-criteria
   values per criterion (Stage 1) and the sum of criterion scores (Stage 2).

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
Q_i = v*(S_i-S*)/(S--S*) + (1-v)*(R_i-R*)/(R--R*), v=0.5 -- lower is better

### 1.3 PROMETHEE II (Brans & Vincke, 1985)

V-shape preference function (p=0.3, q=0.1): P_j(a,b) = max(0, (d-q)/(p-q)), capped at 1
Net flow: Phi_net(a) = [sum_{b!=a} pi(a,b) - sum_{b!=a} pi(b,a)] / (m-1) -- higher is better

### 1.4 COPRAS (Zavadskas et al., 1994)

S+_i = sum benefit criteria (weighted normalized), S-_i = sum cost criteria
Q_i = S+_i + (sum_k S-_k) / (S-_i * sum_k 1/S-_k) -- utility degree U_i = Q_i/Q_max

### 1.5 EDAS (Keshavarz Ghorabaee et al., 2015)

AV_j = mean(x_ij); PDA_ij = max(0, x_ij-AV_j)/AV_j; NDA_ij = max(0, AV_j-x_ij)/AV_j
AS_i = 0.5*(SP_i/max(SP) + 1 - SN_i/max(SN)) -- higher is better

### 1.6 Raw Sum Baseline (Base)

Score_i = sum_j x_ij -- Simple sum of raw, un-normalized sub-criteria values.
This serves as the most transparent possible baseline, requiring zero methodological 
choices beyond the raw data. 
Higher score is better.

> **Note:** **SAW (Simple Additive Weighting)** is also implemented and available in `ranking/saw.py`, but it is not included in the standard hierarchical ranking pipeline track.

---

---

## Part III: Two-Stage Architecture

**Stage 1 (x8 criterion groups):**
1. Min-max normalize SC matrix; apply YearContext SC exclusions.
2. Run 5 MCDM methods with Level-1 SC weights from `CRITICWeightingCalculator`.
3. Run Raw Sum Baseline on raw values.
4. Harmonize scores to produce per-criterion performance metrics.

**Stage 2 (global):**
1. Criterion weights from CRITICWeightCalculator Level 2.
2. Aggregate 8 criterion scores using weighted summation.
3. Produce global score -> final ranking.

---

## Part V: Adaptive Data Handling

- Constant column (range < 1e-12): all values set to 0.5.
- Partial NaN cells: filled with 0.5 after normalization.
- Inactive sub-criteria: dropped per YearContext before normalization.

---

## Part VI: Key Files

| File | Role |
|---|---|
| `ranking/hierarchical_pipeline.py` | HierarchicalRankingPipeline |
| `ranking/topsis.py` | TOPSISCalculator |
| `ranking/vikor.py` | VIKORCalculator |
| `ranking/promethee.py` | PROMETHEECalculator |
| `ranking/copras.py` | COPRASCalculator |
| `ranking/edas.py` | EDASCalculator |
| `ranking/saw.py` | SAWCalculator (Available) |

---

## Part VII: Result Structure

| Field | Type | Description |
|---|---|---|
| `final_ranking` | pd.Series | Province -> rank (1 = best) |
| `final_scores` | pd.Series | Province -> aggregate score in [0,1] |
| `kendall_w` | float | Inter-method concordance in [0,1] |
| `method_weights` | Dict[str, float] | Per-method trust weights |

---

## References

1. Hwang, C.L., & Yoon, K. (1981). Multiple Attribute Decision Making. Springer-Verlag.
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
