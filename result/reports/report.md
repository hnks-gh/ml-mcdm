---
title: "Multi-Criteria Decision Analysis of Vietnamese Provincial Competitiveness"
subtitle: "A Traditional MCDM + Evidential Reasoning Approach with Machine-Learning Forecasting"
date: "2026-03-05"
---

## Table of Contents

- [1. Executive Summary](#executive-summary)
- [2. Data Description and Descriptive Statistics](#data-description-and-descriptive-statistics)
- [3. Objective Weight Derivation](#objective-weight-derivation)
- [4. Hierarchical Evidential Reasoning Ranking](#hierarchical-evidential-reasoning-ranking)
- [5. Criterion-Level MCDM Evaluation](#criterion-level-mcdm-evaluation)
- [6. Inter-Method Agreement and Concordance Analysis](#inter-method-agreement-and-concordance-analysis)
- [7. Sensitivity and Robustness Analysis](#sensitivity-and-robustness-analysis)
- [8. Machine-Learning Forecasting](#machine-learning-forecasting)
- [9. Validity Assessment](#validity-assessment)
- [10. Methodological Notes and References](#methodological-notes-and-references)
- [A. Output File Inventory](#output-file-inventory)

> **Generated:** 2026-03-05 14:16:10  
> **Runtime:** 1011.17 s  
> **Framework:** ML-MCDM v4.0

# 1. Executive Summary

This report documents a comprehensive multi-criteria decision-making (MCDM) evaluation of **61** Vietnamese provinces over the period **2011–2024** (14 years). The analytical framework integrates 29 subcriteria organised into 8 criteria groups, evaluated through 5 classical MCDM methods.

Final provincial rankings are obtained via a two-stage Evidential Reasoning (ER) aggregation procedure that combines belief structures from all constituent methods while explicitly quantifying residual uncertainty.

> **Key Finding:** Top-ranked and bottom-ranked provinces:

**Table 1(a). Highest-ranked provinces.**

| Rank | Province | ER Score |
| ---: | :--- | ---: |
| 1 | P39 | 0.6310 |
| 2 | P12 | 0.5956 |
| 3 | P46 | 0.5894 |
| 4 | P30 | 0.5892 |
| 5 | P38 | 0.5819 |

**Table 1(b). Lowest-ranked provinces.**

| Rank | Province | ER Score |
| ---: | :--- | ---: |
| 61 | P54 | 0.3282 |
| 60 | P40 | 0.3497 |
| 59 | P58 | 0.3603 |
| 58 | P59 | 0.3614 |
| 57 | P43 | 0.3633 |

- **Kendall's $W$ (concordance):** 0.3818
- **Overall Robustness Index:** 0.8584
- **Confidence Level:** 95%

# 2. Data Description and Descriptive Statistics

The dataset comprises a balanced panel of 61 provinces observed annually from 2011 to 2024, yielding 854 province-year observations.

| Parameter | Value |
| :--- | ---: |
| Provinces ($N$) | 61 |
| Temporal span | 2011–2024 |
| Annual periods ($T$) | 14 |
| Criteria | 8 |
| Subcriteria | 29 |
| Total observations ($N \times T$) | 854 |

**Table 2. Descriptive statistics (2024 cross-section).**

| Subcriteria | Mean | Std Dev | Min | Max | CV |
| :--- | ---: | ---: | ---: | ---: | ---: |
| SC11 | 1.0857 | 0.1511 | 0.7542 | 1.4257 | 0.1392 |
| SC12 | 1.3922 | 0.2299 | 0.8518 | 1.9302 | 0.1651 |
| SC13 | 1.4175 | 0.1420 | 1.0710 | 1.6903 | 0.1002 |
| SC14 | 1.0837 | 0.2071 | 0.5564 | 1.5229 | 0.1911 |
| SC21 | 0.8503 | 0.0736 | 0.7104 | 1.0528 | 0.0866 |
| SC22 | 1.7211 | 0.1786 | 1.2144 | 2.0356 | 0.1038 |
| SC23 | 1.3809 | 0.1117 | 1.1195 | 1.7367 | 0.0809 |
| SC24 | 1.3524 | 0.1078 | 1.0517 | 1.5916 | 0.0797 |
| SC31 | 1.9785 | 0.1046 | 1.7150 | 2.2201 | 0.0529 |
| SC32 | 0.4737 | 0.0438 | 0.3884 | 0.6106 | 0.0925 |
| SC33 | 1.8382 | 0.1094 | 1.5066 | 2.0811 | 0.0595 |
| SC41 | 1.7313 | 0.1567 | 1.4095 | 2.1385 | 0.0905 |
| SC42 | 2.0602 | 0.0973 | 1.8528 | 2.2852 | 0.0472 |
| SC43 | 1.2558 | 0.1741 | 0.9574 | 1.7515 | 0.1386 |
| SC44 | 1.9711 | 0.0874 | 1.6047 | 2.1694 | 0.0443 |
| SC51 | 2.4106 | 0.0898 | 2.1439 | 2.6273 | 0.0372 |
| SC52 | nan | nan | nan | nan | nan |
| SC53 | 2.3347 | 0.1418 | 1.9168 | 2.5996 | 0.0608 |
| SC54 | 2.4614 | 0.0632 | 2.2860 | 2.5832 | 0.0257 |
| SC61 | 1.9452 | 0.0988 | 1.6863 | 2.1914 | 0.0508 |
| SC62 | 1.6983 | 0.2884 | 0.9969 | 2.1140 | 0.1698 |
| SC63 | 2.0036 | 0.1657 | 1.7171 | 2.3475 | 0.0827 |
| SC64 | 1.9534 | 0.0455 | 1.8536 | 2.0722 | 0.0233 |
| SC71 | 1.0619 | 0.1146 | 0.8168 | 1.3301 | 0.1080 |
| SC72 | 1.9897 | 0.1419 | 1.5691 | 2.3521 | 0.0713 |
| SC73 | 0.5885 | 0.2121 | 0.3416 | 1.3578 | 0.3604 |
| SC81 | 0.4982 | 0.0706 | 0.3836 | 0.6943 | 0.1417 |
| SC82 | 2.4639 | 0.2529 | 2.0006 | 2.9605 | 0.1026 |
| SC83 | 0.4316 | 0.0483 | 0.3491 | 0.6151 | 0.1120 |

# 3. Objective Weight Derivation

Subcriteria weights are derived through a two-level hierarchical Monte Carlo ensemble that blends Shannon Entropy and CRITIC via a Beta-distributed mixing coefficient.  Level 1 produces local SC weights within each criterion group; Level 2 determines criterion-level weights from a composite matrix.  Global SC weights are the product of local and criterion weights, re-normalised to the simplex.

The global weight of subcriteria $j$ in criterion group $C_k$ is:

$$w_j = \frac{u_j^{(k)} \cdot v_k}{\sum_{k^{\prime}} \sum_{j^{\prime} \in C_{k^{\prime}}} u_{j^{\prime}}^{(k^{\prime})} v_{k^{\prime}}}$$

where $u_j^{(k)}$ is the Level-1 local SC weight and $v_k$ is the Level-2 criterion weight.

**Table 3. Subcriteria global weights (Hybrid MC Ensemble).**

| Subcriteria | Criterion | Criterion Weight | Local Weight | Global Weight | MC Std | MC CV |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| SC11 | C01 | 0.1108 | 0.2652 | 0.0294 | 0.0283 | 0.1068 |
| SC12 | C01 | 0.1108 | 0.2105 | 0.0233 | 0.0177 | 0.0842 |
| SC13 | C01 | 0.1108 | 0.2124 | 0.0235 | 0.0158 | 0.0744 |
| SC14 | C01 | 0.1108 | 0.3119 | 0.0346 | 0.0217 | 0.0697 |
| SC21 | C02 | 0.0792 | 0.5342 | 0.0423 | 0.0488 | 0.0913 |
| SC22 | C02 | 0.0792 | 0.1805 | 0.0143 | 0.0165 | 0.0915 |
| SC23 | C02 | 0.0792 | 0.1855 | 0.0147 | 0.0173 | 0.0935 |
| SC24 | C02 | 0.0792 | 0.0998 | 0.0079 | 0.0274 | 0.2740 |
| SC31 | C03 | 0.2592 | 0.1660 | 0.0430 | 0.0319 | 0.1922 |
| SC32 | C03 | 0.2592 | 0.5944 | 0.1541 | 0.0531 | 0.0894 |
| SC33 | C03 | 0.2592 | 0.2395 | 0.0621 | 0.0246 | 0.1029 |
| SC41 | C04 | 0.0968 | 0.2219 | 0.0215 | 0.0246 | 0.1108 |
| SC42 | C04 | 0.0968 | 0.2004 | 0.0194 | 0.0222 | 0.1108 |
| SC43 | C04 | 0.0968 | 0.3349 | 0.0324 | 0.0370 | 0.1105 |
| SC44 | C04 | 0.0968 | 0.2428 | 0.0235 | 0.0424 | 0.1747 |
| SC51 | C05 | 0.0819 | 0.2107 | 0.0173 | 0.0204 | 0.0969 |
| SC52 | C05 | 0.0819 | 0.1962 | 0.0161 | 0.0463 | 0.2360 |
| SC53 | C05 | 0.0819 | 0.2242 | 0.0184 | 0.0204 | 0.0909 |
| SC54 | C05 | 0.0819 | 0.3690 | 0.0302 | 0.0402 | 0.1090 |
| SC61 | C06 | 0.1227 | 0.2136 | 0.0262 | 0.0184 | 0.0862 |
| SC62 | C06 | 0.1227 | 0.1969 | 0.0242 | 0.0144 | 0.0730 |
| SC63 | C06 | 0.1227 | 0.2852 | 0.0350 | 0.0202 | 0.0707 |
| SC64 | C06 | 0.1227 | 0.3043 | 0.0373 | 0.0219 | 0.0720 |
| SC71 | C07 | 0.1184 | 0.3813 | 0.0451 | 0.0241 | 0.0631 |
| SC72 | C07 | 0.1184 | 0.1740 | 0.0206 | 0.0384 | 0.2210 |
| SC73 | C07 | 0.1184 | 0.4448 | 0.0526 | 0.0402 | 0.0905 |
| SC81 | C08 | 0.1309 | 0.2434 | 0.0319 | 0.0179 | 0.0734 |
| SC82 | C08 | 0.1309 | 0.1735 | 0.0227 | 0.0330 | 0.1900 |
| SC83 | C08 | 0.1309 | 0.5830 | 0.0763 | 0.0488 | 0.0836 |

- **Sum of global weights:** 1.000000
- **Max weight:** 0.154093 (SC32)
- **Min weight:** 0.007912 (SC24)
- **Shannon entropy $H(\mathbf{{w}})$:** 3.1533

**Table 4. Level-2 criterion weights (MC diagnostics).**

| Criterion | Weight | MC Mean | MC Std | 95% CI Lower | 95% CI Upper |
| :--- | ---: | ---: | ---: | ---: | ---: |
| C01 | 0.1108 | 0.1108 | 0.0063 | 0.0978 | 0.1230 |
| C02 | 0.0792 | 0.0792 | 0.0067 | 0.0657 | 0.0915 |
| C03 | 0.2592 | 0.2592 | 0.0198 | 0.2209 | 0.2963 |
| C04 | 0.0968 | 0.0968 | 0.0111 | 0.0758 | 0.1180 |
| C05 | 0.0819 | 0.0819 | 0.0140 | 0.0572 | 0.1084 |
| C06 | 0.1227 | 0.1227 | 0.0108 | 0.1007 | 0.1429 |
| C07 | 0.1184 | 0.1184 | 0.0091 | 0.1022 | 0.1379 |
| C08 | 0.1309 | 0.1309 | 0.0109 | 0.1105 | 0.1534 |

- **Level-2 Avg Kendall τ:** 0.9695
- **Level-2 Kendall's W:** 0.9959

# 4. Hierarchical Evidential Reasoning Ranking

The ER approach (Yang & Xu, 2002) aggregates MCDM scores into belief structures.  The recursive ER algorithm for combining two evidence bodies is:

$$m_{1 \oplus 2}(H_n) = \frac{m_1(H_n) m_2(\Theta) + m_2(H_n) m_1(\Theta) + m_1(H_n) m_2(H_n)}{1 - K}$$

where $K = \sum_{H_i \cap H_j = \varnothing} m_1(H_i) m_2(H_j)$ is the conflict factor.

- **Aggregation:** Evidential Reasoning (Yang & Xu, 2002)
- **MCDM Methods:** 5
- **Kendall's $W$:** 0.3818
- **Target Year:** 2024

**Table 5. Complete provincial ranking by ER composite score.**

| Rank | Province | ER Score | $z$-Score | Quartile |
| ---: | :--- | ---: | ---: | :---: |
| 1 | P39 | 0.6310 | +2.200 | Q1 |
| 2 | P12 | 0.5956 | +1.707 | Q1 |
| 3 | P46 | 0.5894 | +1.619 | Q1 |
| 4 | P30 | 0.5892 | +1.617 | Q1 |
| 5 | P38 | 0.5819 | +1.515 | Q1 |
| 6 | P49 | 0.5720 | +1.378 | Q1 |
| 7 | P14 | 0.5567 | +1.164 | Q1 |
| 8 | P28 | 0.5559 | +1.153 | Q1 |
| 9 | P22 | 0.5533 | +1.116 | Q1 |
| 10 | P63 | 0.5522 | +1.101 | Q1 |
| 11 | P29 | 0.5447 | +0.996 | Q1 |
| 12 | P26 | 0.5426 | +0.968 | Q1 |
| 13 | P18 | 0.5384 | +0.909 | Q1 |
| 14 | P60 | 0.5361 | +0.877 | Q1 |
| 15 | P31 | 0.5337 | +0.843 | Q1 |
| 16 | P62 | 0.5283 | +0.767 | Q1 |
| 17 | P04 | 0.5261 | +0.738 | Q2 |
| 18 | P47 | 0.5163 | +0.600 | Q2 |
| 19 | P21 | 0.5135 | +0.561 | Q2 |
| 20 | P24 | 0.5127 | +0.551 | Q2 |
| 21 | P56 | 0.5123 | +0.545 | Q2 |
| 22 | P27 | 0.5095 | +0.505 | Q2 |
| 23 | P13 | 0.5017 | +0.397 | Q2 |
| 24 | P15 | 0.4996 | +0.367 | Q2 |
| 25 | P34 | 0.4935 | +0.282 | Q2 |
| 26 | P11 | 0.4919 | +0.260 | Q2 |
| 27 | P09 | 0.4907 | +0.243 | Q2 |
| 28 | P32 | 0.4897 | +0.229 | Q2 |
| 29 | P02 | 0.4865 | +0.184 | Q2 |
| 30 | P16 | 0.4843 | +0.154 | Q2 |
| 31 | P61 | 0.4834 | +0.141 | Q2 |
| 32 | P35 | 0.4786 | +0.074 | Q3 |
| 33 | P25 | 0.4744 | +0.016 | Q3 |
| 34 | P05 | 0.4706 | -0.037 | Q3 |
| 35 | P42 | 0.4616 | -0.162 | Q3 |
| 36 | P01 | 0.4586 | -0.205 | Q3 |
| 37 | P23 | 0.4545 | -0.261 | Q3 |
| 38 | P07 | 0.4509 | -0.312 | Q3 |
| 39 | P37 | 0.4470 | -0.366 | Q3 |
| 40 | P08 | 0.4305 | -0.596 | Q3 |
| 41 | P33 | 0.4291 | -0.615 | Q3 |
| 42 | P55 | 0.4279 | -0.632 | Q3 |
| 43 | P19 | 0.4223 | -0.711 | Q3 |
| 44 | P06 | 0.4219 | -0.716 | Q3 |
| 45 | P10 | 0.4188 | -0.759 | Q3 |
| 46 | P57 | 0.4188 | -0.759 | Q3 |
| 47 | P20 | 0.4153 | -0.808 | Q4 |
| 48 | P50 | 0.4105 | -0.875 | Q4 |
| 49 | P51 | 0.4074 | -0.918 | Q4 |
| 50 | P36 | 0.4062 | -0.935 | Q4 |
| 51 | P45 | 0.3922 | -1.130 | Q4 |
| 52 | P53 | 0.3891 | -1.174 | Q4 |
| 53 | P41 | 0.3890 | -1.175 | Q4 |
| 54 | P03 | 0.3824 | -1.268 | Q4 |
| 55 | P48 | 0.3706 | -1.431 | Q4 |
| 56 | P44 | 0.3641 | -1.522 | Q4 |
| 57 | P43 | 0.3633 | -1.533 | Q4 |
| 58 | P59 | 0.3614 | -1.560 | Q4 |
| 59 | P58 | 0.3603 | -1.575 | Q4 |
| 60 | P40 | 0.3497 | -1.723 | Q4 |
| 61 | P54 | 0.3282 | -2.022 | Q4 |

### Distributional Properties

| Statistic | Value |
| :--- | ---: |
| Mean | 0.4732 |
| Median | 0.4834 |
| Std Dev | 0.0717 |
| Skewness | -0.0204 |
| Excess Kurtosis | -0.8249 |
| IQR | 0.1095 |

### Evidential Reasoning Uncertainty

- **Mean Belief Entropy:** 1.3552 (SD = 0.1423)
- **Mean Utility Interval Width:** 0.4623 (SD = 0.0063)

# 5. Criterion-Level MCDM Evaluation

Each of the 8 criteria groups is independently evaluated by 5 MCDM methods.

**Table 6. Criterion weights (Stage 2 ER).**

| Criterion | Weight |
| :--- | ---: |
| C01 | 0.110850 |
| C02 | 0.079242 |
| C03 | 0.259231 |
| C04 | 0.096792 |
| C05 | 0.081916 |
| C06 | 0.122719 |
| C07 | 0.118361 |
| C08 | 0.130889 |

**C01** — top 3: P28 (1.0000), P12 (0.9728), P15 (0.9549)
**C02** — top 3: P39 (1.0000), P12 (0.9650), P01 (0.8483)
**C03** — top 3: P29 (0.9948), P12 (0.8609), P30 (0.8467)
**C04** — top 3: P46 (1.0000), P14 (0.8944), P39 (0.7988)
**C05** — top 3: P49 (1.0000), P18 (0.9851), P56 (0.9581)
**C06** — top 3: P14 (0.9844), P49 (0.9599), P18 (0.9074)
**C07** — top 3: P56 (0.9604), P60 (0.9489), P57 (0.9086)
**C08** — top 3: P47 (0.9911), P49 (0.9828), P01 (0.7554)

# 6. Inter-Method Agreement and Concordance Analysis

Kendall's coefficient of concordance $W = 0.3818$ indicates **fair** agreement among the 5 methods.

**Table 7. Provinces most frequently ranked in the top 5.**

| Province | Count | Frequency |
| :--- | ---: | ---: |
| P46 | 18 | 45.0% |
| P49 | 16 | 40.0% |
| P12 | 15 | 37.5% |
| P14 | 15 | 37.5% |
| P39 | 15 | 37.5% |
| P01 | 10 | 25.0% |
| P18 | 10 | 25.0% |
| P56 | 10 | 25.0% |
| P47 | 9 | 22.5% |
| P61 | 8 | 20.0% |

# 7. Sensitivity and Robustness Analysis

- **Overall Robustness Index:** 0.8584
- **Confidence Level:** 95%

## 7.1 Criteria Weight Sensitivity

**Table 8. Criteria weight sensitivity indices.**

| Criterion | Sensitivity | Classification |
| :--- | ---: | :---: |
| C03 | 1.0000 | High |
| C07 | 0.9225 | High |
| C06 | 0.7819 | High |
| C08 | 0.7239 | High |
| C01 | 0.6723 | High |
| C04 | 0.5554 | High |
| C02 | 0.4448 | High |
| C05 | 0.3972 | High |

## 7.2 Subcriteria Weight Sensitivity (Top 15)

**Table 9. Most influential subcriteria.**

| Subcriteria | Sensitivity |
| :--- | ---: |
| SC32 | 1.0000 |
| SC73 | 0.6742 |
| SC83 | 0.6067 |
| SC24 | 0.5974 |
| SC22 | 0.5829 |
| SC71 | 0.5807 |
| SC42 | 0.5780 |
| SC11 | 0.5768 |
| SC61 | 0.5768 |
| SC13 | 0.5723 |
| SC72 | 0.5672 |
| SC12 | 0.5625 |
| SC81 | 0.5625 |
| SC43 | 0.5618 |
| SC82 | 0.5618 |

## 7.3 Top-N Ranking Stability

**Table 10. Top-N set stability under weight perturbation.**

| Tier | Index | Percentage |
| :--- | ---: | ---: |
| Top-3 | 0.4710 | 47.1% |
| Top-5 | 1.0000 | 100.0% |
| Top-10 | 0.9190 | 91.9% |

## 7.4 Temporal Rank Stability

**Table 11. Year-to-year rank correlation.**

| Year Pair | Spearman $\rho$ | Strength |
| :--- | ---: | :---: |
| 2020-2021 | 0.3146 | Weak |
| 2021-2022 | 0.7582 | Moderate |
| 2022-2023 | 0.6504 | Moderate |
| 2023-2024 | 0.6595 | Moderate |

## 7.5 Provincial Rank Stability

**Table 12(a). Ten most volatile provinces.**

| Province | Stability |
| :--- | ---: |
| P14 | 0.9549 |
| P56 | 0.9561 |
| P57 | 0.9561 |
| P47 | 0.9577 |
| P59 | 0.9614 |
| P32 | 0.9616 |
| P27 | 0.9634 |
| P06 | 0.9646 |
| P61 | 0.9658 |
| P29 | 0.9676 |

**Table 12(b). Ten most stable provinces.**

| Province | Stability |
| :--- | ---: |
| P62 | 0.9882 |
| P35 | 0.9912 |
| P12 | 0.9914 |
| P03 | 0.9982 |
| P38 | 0.9990 |
| P49 | 0.9990 |
| P39 | 1.0000 |
| P40 | 1.0000 |
| P48 | 1.0000 |
| P54 | 1.0000 |

# 8. Machine-Learning Forecasting

A Super Learner meta-ensemble (van der Laan et al., 2007) forecasts provincial scores one period ahead.  Prediction intervals are obtained through Conformal Prediction (Vovk et al., 2005).

## 8.1 Super Learner Model Contributions

**Table 13. Base-learner weights.**

| Model | Weight | Contribution |
| :--- | ---: | ---: |
| BayesianRidge | 0.4100 | 41.0% |
| QuantileRF | 0.2096 | 21.0% |
| NAM | 0.1412 | 14.1% |
| GradientBoosting | 0.1354 | 13.5% |
| PanelVAR | 0.1038 | 10.4% |

## 8.2 Individual Model Performance

**Table 14. Out-of-sample performance metrics.**

| Model | MEAN_R2 | STD_R2 |
| :--- | ---: | ---: |
| BayesianRidge | -9.9617 | 0.0000 |
| GradientBoosting | -17.3678 | 0.0000 |
| NAM | -10.0646 | 0.0000 |
| PanelVAR | -9.2047 | 0.0000 |
| QuantileRF | -10.6493 | 0.0000 |

## 8.3 Cross-Validation Results

**Table 15. K-fold CV summary ($R^2$).**

| Model | Mean | Std Dev | Min | Max |
| :--- | ---: | ---: | ---: | ---: |
| BayesianRidge | -9.9617 | 0.0000 | -9.9617 | -9.9617 |
| GradientBoosting | -17.3678 | 0.0000 | -17.3678 | -17.3678 |
| NAM | -10.0646 | 0.0000 | -10.0646 | -10.0646 |
| PanelVAR | -9.2047 | 0.0000 | -9.2047 | -9.2047 |
| QuantileRF | -10.6493 | 0.0000 | -10.6493 | -10.6493 |

## 8.4 Holdout Validation

- **r2:** 0.9740
- **rmse:** 0.0820
- **mae:** 0.0514

## 8.6 Conformal Prediction Interval Diagnostics

- **Nominal Coverage:** 95%
- **Mean Width:** 0.8435
- **Median Width:** 0.6770
- **Range:** [0.2412, 2.4786]

# 9. Validity Assessment

**Table 17. Internal validity diagnostics.**

| Criterion | Result |
| :--- | :---: |
| Weight normalisation ($\sum w = 1$) | PASS |
| Rank completeness (all provinces ranked) | PASS |
| Score range ($0 \le s \le 1$) | PASS |
| Kendall's $W \ge 0.5$ | FAIL |
| Robustness $\ge 0.7$ | PASS |

# 10. Methodological Notes and References

## 10.1 Objective Weighting Methods

- **Entropy** — Shannon (1948). Information-theoretic derivation exploiting probability-distribution diversity across alternatives.
- **CRITIC** — Diakoulaki, Mavrotas & Papayannakis (1995). Weights based on contrast intensity and conflicting inter-criteria correlation.
- **Hybrid MC Ensemble** — Reliability-weighted Bayesian bootstrap combination accounting for inter-method agreement at the subcriteria level.

## 10.2 MCDM Methods

Six classical methods: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW. All share the same normalized decision matrix and weight vector.

## 10.3 Evidential Reasoning

The ER framework (Yang & Xu, 2002) transforms method outputs into basic probability assignments over an evaluation grade set. Stage 1 aggregates methods within each criterion; Stage 2 aggregates criteria into the final score.

## 10.4 Machine-Learning Forecasting

Super Learner (van der Laan, Polley & Hubbard, 2007) constructs an optimal convex combination of heterogeneous base learners. Conformal Prediction (Vovk, Gammerman & Shafer, 2005) provides distribution-free intervals.

# Appendix A — Output File Inventory

### CSV Result Files

- `cross_validation_scores.csv`
- `feature_importance.csv`
- `forecast_predictions.csv`
- `holdout_performance.csv`
- `individual_model_predictions.csv`
- `model_contributions.csv`
- `model_performance.csv`
- `prediction_intervals.csv`
- `mcdm_composite_scores.csv`
- `mcdm_rank_comparison.csv`
- `mcdm_scores_C01.csv`
- `mcdm_scores_C02.csv`
- `mcdm_scores_C03.csv`
- `mcdm_scores_C04.csv`
- `mcdm_scores_C05.csv`
- `mcdm_scores_C06.csv`
- `mcdm_scores_C07.csv`
- `mcdm_scores_C08.csv`
- `criterion_er_scores_all_years.csv`
- `final_rankings.csv`
- `prediction_uncertainty_er.csv`
- `rankings_all_years.csv`
- `ranks_all_years.csv`
- `perturbation_detail.csv`
- `sensitivity_criteria.csv`
- `sensitivity_rank_stability.csv`
- `sensitivity_subcriteria.csv`
- `sensitivity_temporal.csv`
- `sensitivity_top_n_stability.csv`
- `data_summary_statistics.csv`
- `criterion_weights.csv`
- `critic_weights.csv`
- `entropy_weights.csv`
- `mc_province_rankings.csv`
- `method_weights_comparison.csv`
- `weights_analysis.csv`

### JSON Metadata Files

- `forecast_summary.json`
- `sensitivity_summary.json`
- `config_snapshot.json`
- `execution_summary.json`

### Figures (47 files)

- `fig06_method_agreement.png`
- `fig06b_agreement_per_criterion.png`
- `fig07_criterion_parallel_grid.png`
- `fig08_C01_scores.png`
- `fig08_C02_scores.png`
- `fig08_C03_scores.png`
- `fig08_C04_scores.png`
- `fig08_C05_scores.png`
- `fig08_C06_scores.png`
- `fig08_C07_scores.png`
- `fig08_C08_scores.png`
- `fig08b_mcdm_composite_scatter.png`
- `fig08c_criterion_er_utility.png`
- `fig01_final_er_ranking.png`
- `fig01b_tier_ranking.png`
- `fig01d_belief_heatmap.png`
- `fig01e_rank_uncertainty_scatter.png`
- `fig02_score_distribution.png`
- `fig02b_mc_rank_uncertainty.png`
- `fig09_criteria_sensitivity.png`
- `fig09b_tornado_butterfly.png`
- `fig09c_subcriteria_dotstrip.png`
- `fig10_subcriteria_sensitivity.png`
- `fig11_top_n_stability.png`
- `fig12_temporal_stability.png`
- `fig13_rank_volatility.png`
- `fig13b_stability_line_ci.png`
- `fig14b_rank_change_violin.png`
- `fig15_er_uncertainty.png`
- `fig25_robustness_summary.png`
- `fig03_weights_comparison.png`
- `fig03c_criterion_weights.png`
- `fig03d_weight_deviation.png`
- `fig04_weight_radar.png`
- `fig04a_weight_radar_criteria.png`
- `fig04a_weight_radar_criteria_critic.png`
- `fig04a_weight_radar_criteria_entropy.png`
- `fig04b_C01_radar.png`
- `fig04b_C02_radar.png`
- `fig04b_C03_radar.png`
- `fig04b_C04_radar.png`
- `fig04b_C05_radar.png`
- `fig04b_C06_radar.png`
- `fig04b_C07_radar.png`
- `fig04b_C08_radar.png`
- `fig04c_weight_hierarchical_rose.png`
- `fig05_weight_heatmap.png`

---
*End of report.*