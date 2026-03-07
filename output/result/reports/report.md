---
title: "Multi-Criteria Decision Analysis of Vietnamese Provincial Competitiveness"
subtitle: "A Traditional MCDM + Evidential Reasoning Approach with Machine-Learning Forecasting"
date: "2026-03-06"
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

> **Generated:** 2026-03-06 17:59:57  
> **Runtime:** 589.79 s  
> **Framework:** ML-MCDM v4.0

# 1. Executive Summary

This report documents a comprehensive multi-criteria decision-making (MCDM) evaluation of **61** Vietnamese provinces over the period **2011–2024** (14 years). The analytical framework integrates 29 subcriteria organised into 8 criteria groups, evaluated through 6 classical MCDM methods.

Final provincial rankings are obtained via a two-stage Evidential Reasoning (ER) aggregation procedure that combines belief structures from all constituent methods while explicitly quantifying residual uncertainty.

> **Key Finding:** Top-ranked and bottom-ranked provinces:

**Table 1(a). Highest-ranked provinces.**

| Rank | Province | ER Score |
| ---: | :--- | ---: |
| 1 | P39 | 0.6584 |
| 2 | P46 | 0.6358 |
| 3 | P12 | 0.6242 |
| 4 | P14 | 0.6205 |
| 5 | P38 | 0.6129 |

**Table 1(b). Lowest-ranked provinces.**

| Rank | Province | ER Score |
| ---: | :--- | ---: |
| 61 | P58 | 0.3455 |
| 60 | P40 | 0.3530 |
| 59 | P54 | 0.3551 |
| 58 | P59 | 0.3670 |
| 57 | P43 | 0.3761 |

- **Kendall's $W$ (concordance):** 0.4321
- **Overall Robustness Index:** 0.8609
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

Subcriteria weights are derived through a two-level deterministic CRITIC pipeline.  Level 1 runs CRITIC on each criterion group independently, producing local SC weights that sum to 1 within each group.  Level 2 runs CRITIC on a criterion composite matrix, producing criterion-level weights that sum to 1 globally.  Global SC weights are the product of local and criterion weights, re-normalised to the simplex.

The global weight of subcriteria $j$ in criterion group $C_k$ is:

$$w_j = \frac{u_j^{(k)} \cdot v_k}{\sum_{k^{\prime}} \sum_{j^{\prime} \in C_{k^{\prime}}} u_{j^{\prime}}^{(k^{\prime})} v_{k^{\prime}}}$$

where $u_j^{(k)}$ is the Level-1 local SC weight and $v_k$ is the Level-2 criterion weight.

**Table 3. Subcriteria global weights (CRITIC two-level).**

| Subcriteria | Criterion | Criterion Weight | Local Weight | Global Weight |
| :--- | :--- | ---: | ---: | ---: |
| SC11 | C01 | 0.1225 | 0.2311 | 0.0283 |
| SC12 | C01 | 0.1225 | 0.2303 | 0.0282 |
| SC13 | C01 | 0.1225 | 0.2290 | 0.0281 |
| SC14 | C01 | 0.1225 | 0.3095 | 0.0379 |
| SC21 | C02 | 0.0968 | 0.3974 | 0.0385 |
| SC22 | C02 | 0.0968 | 0.2248 | 0.0218 |
| SC23 | C02 | 0.0968 | 0.2182 | 0.0211 |
| SC24 | C02 | 0.0968 | 0.1596 | 0.0155 |
| SC31 | C03 | 0.1936 | 0.2592 | 0.0502 |
| SC32 | C03 | 0.1936 | 0.4513 | 0.0874 |
| SC33 | C03 | 0.1936 | 0.2895 | 0.0561 |
| SC41 | C04 | 0.1264 | 0.2226 | 0.0281 |
| SC42 | C04 | 0.1264 | 0.1897 | 0.0240 |
| SC43 | C04 | 0.1264 | 0.2547 | 0.0322 |
| SC44 | C04 | 0.1264 | 0.3330 | 0.0421 |
| SC51 | C05 | 0.1087 | 0.2023 | 0.0220 |
| SC52 | C05 | 0.1087 | 0.3218 | 0.0350 |
| SC53 | C05 | 0.1087 | 0.2103 | 0.0229 |
| SC54 | C05 | 0.1087 | 0.2656 | 0.0289 |
| SC61 | C06 | 0.1491 | 0.2470 | 0.0368 |
| SC62 | C06 | 0.1491 | 0.2140 | 0.0319 |
| SC63 | C06 | 0.1491 | 0.2685 | 0.0400 |
| SC64 | C06 | 0.1491 | 0.2705 | 0.0403 |
| SC71 | C07 | 0.1083 | 0.3786 | 0.0410 |
| SC72 | C07 | 0.1083 | 0.2678 | 0.0290 |
| SC73 | C07 | 0.1083 | 0.3535 | 0.0383 |
| SC81 | C08 | 0.0945 | 0.2867 | 0.0271 |
| SC82 | C08 | 0.0945 | 0.2694 | 0.0255 |
| SC83 | C08 | 0.0945 | 0.4439 | 0.0419 |

- **Sum of global weights:** 1.000000
- **Max weight:** 0.087395 (SC32)
- **Min weight:** 0.015454 (SC24)
- **Weight entropy $H(\mathbf{{w}})$:** 3.3018

**Table 4. Level-2 criterion weights.**

| Criterion | Weight |
| :--- | ---: |
| C01 | 0.122495 |
| C02 | 0.096823 |
| C03 | 0.193632 |
| C04 | 0.126407 |
| C05 | 0.108750 |
| C06 | 0.149087 |
| C07 | 0.108331 |
| C08 | 0.094474 |

# 4. Hierarchical Evidential Reasoning Ranking

The ER approach (Yang & Xu, 2002) aggregates MCDM scores into belief structures.  The recursive ER algorithm for combining two evidence bodies is:

$$m_{1 \oplus 2}(H_n) = \frac{m_1(H_n) m_2(\Theta) + m_2(H_n) m_1(\Theta) + m_1(H_n) m_2(H_n)}{1 - K}$$

where $K = \sum_{H_i \cap H_j = \varnothing} m_1(H_i) m_2(H_j)$ is the conflict factor.

- **Aggregation:** Evidential Reasoning (Yang & Xu, 2002)
- **MCDM Methods:** 6
- **Kendall's $W$:** 0.4321
- **Target Year:** 2024

**Table 5. Complete provincial ranking by ER composite score.**

| Rank | Province | ER Score | $z$-Score | Quartile |
| ---: | :--- | ---: | ---: | :---: |
| 1 | P39 | 0.6584 | +2.155 | Q1 |
| 2 | P46 | 0.6358 | +1.864 | Q1 |
| 3 | P12 | 0.6242 | +1.713 | Q1 |
| 4 | P14 | 0.6205 | +1.665 | Q1 |
| 5 | P38 | 0.6129 | +1.567 | Q1 |
| 6 | P49 | 0.6045 | +1.459 | Q1 |
| 7 | P28 | 0.5825 | +1.175 | Q1 |
| 8 | P30 | 0.5818 | +1.166 | Q1 |
| 9 | P18 | 0.5751 | +1.080 | Q1 |
| 10 | P22 | 0.5708 | +1.024 | Q1 |
| 11 | P26 | 0.5660 | +0.962 | Q1 |
| 12 | P63 | 0.5655 | +0.956 | Q1 |
| 13 | P31 | 0.5600 | +0.884 | Q1 |
| 14 | P62 | 0.5505 | +0.763 | Q1 |
| 15 | P47 | 0.5425 | +0.659 | Q1 |
| 16 | P21 | 0.5424 | +0.657 | Q1 |
| 17 | P60 | 0.5399 | +0.626 | Q2 |
| 18 | P24 | 0.5355 | +0.569 | Q2 |
| 19 | P56 | 0.5336 | +0.543 | Q2 |
| 20 | P13 | 0.5328 | +0.533 | Q2 |
| 21 | P61 | 0.5285 | +0.477 | Q2 |
| 22 | P04 | 0.5281 | +0.472 | Q2 |
| 23 | P29 | 0.5264 | +0.452 | Q2 |
| 24 | P15 | 0.5239 | +0.419 | Q2 |
| 25 | P09 | 0.5181 | +0.344 | Q2 |
| 26 | P16 | 0.5165 | +0.323 | Q2 |
| 27 | P25 | 0.5123 | +0.269 | Q2 |
| 28 | P27 | 0.5074 | +0.206 | Q2 |
| 29 | P32 | 0.5069 | +0.200 | Q2 |
| 30 | P02 | 0.5045 | +0.169 | Q2 |
| 31 | P11 | 0.5015 | +0.130 | Q2 |
| 32 | P34 | 0.4985 | +0.091 | Q3 |
| 33 | P37 | 0.4867 | -0.061 | Q3 |
| 34 | P42 | 0.4845 | -0.090 | Q3 |
| 35 | P35 | 0.4839 | -0.097 | Q3 |
| 36 | P01 | 0.4812 | -0.132 | Q3 |
| 37 | P23 | 0.4782 | -0.172 | Q3 |
| 38 | P05 | 0.4758 | -0.202 | Q3 |
| 39 | P08 | 0.4584 | -0.427 | Q3 |
| 40 | P19 | 0.4513 | -0.519 | Q3 |
| 41 | P57 | 0.4485 | -0.555 | Q3 |
| 42 | P55 | 0.4449 | -0.601 | Q3 |
| 43 | P20 | 0.4408 | -0.654 | Q3 |
| 44 | P07 | 0.4365 | -0.710 | Q3 |
| 45 | P50 | 0.4342 | -0.739 | Q3 |
| 46 | P10 | 0.4288 | -0.809 | Q3 |
| 47 | P51 | 0.4273 | -0.828 | Q4 |
| 48 | P06 | 0.4178 | -0.950 | Q4 |
| 49 | P53 | 0.4118 | -1.028 | Q4 |
| 50 | P45 | 0.4082 | -1.074 | Q4 |
| 51 | P33 | 0.4078 | -1.080 | Q4 |
| 52 | P36 | 0.4075 | -1.084 | Q4 |
| 53 | P48 | 0.3961 | -1.231 | Q4 |
| 54 | P41 | 0.3929 | -1.272 | Q4 |
| 55 | P03 | 0.3923 | -1.279 | Q4 |
| 56 | P44 | 0.3791 | -1.451 | Q4 |
| 57 | P43 | 0.3761 | -1.489 | Q4 |
| 58 | P59 | 0.3670 | -1.607 | Q4 |
| 59 | P54 | 0.3551 | -1.760 | Q4 |
| 60 | P40 | 0.3530 | -1.788 | Q4 |
| 61 | P58 | 0.3455 | -1.884 | Q4 |

### Distributional Properties

| Statistic | Value |
| :--- | ---: |
| Mean | 0.4915 |
| Median | 0.5015 |
| Std Dev | 0.0775 |
| Skewness | 0.0082 |
| Excess Kurtosis | -0.7624 |
| IQR | 0.1136 |

### Evidential Reasoning Uncertainty

- **Mean Belief Entropy:** 1.3323 (SD = 0.1348)
- **Mean Utility Interval Width:** 0.4675 (SD = 0.0061)

# 5. Criterion-Level MCDM Evaluation

Each of the 8 criteria groups is independently evaluated by 6 MCDM methods.

**Table 6. Criterion weights (Stage 2 ER).**

| Criterion | Weight |
| :--- | ---: |
| C01 | 0.122495 |
| C02 | 0.096823 |
| C03 | 0.193632 |
| C04 | 0.126407 |
| C05 | 0.108750 |
| C06 | 0.149087 |
| C07 | 0.108331 |
| C08 | 0.094474 |

**C01** — top 3: P28 (0.9961), P15 (0.9824), P12 (0.9782)
**C02** — top 3: P39 (1.0000), P12 (0.9281), P46 (0.8665)
**C03** — top 3: P29 (0.9746), P12 (0.9484), P63 (0.8724)
**C04** — top 3: P46 (1.0000), P14 (0.9204), P39 (0.7765)
**C05** — top 3: P49 (0.9976), P18 (0.9765), P46 (0.9714)
**C06** — top 3: P14 (0.9668), P49 (0.9619), P18 (0.9363)
**C07** — top 3: P57 (0.9522), P56 (0.9401), P60 (0.9178)
**C08** — top 3: P47 (1.0000), P49 (0.9115), P01 (0.7973)

# 6. Inter-Method Agreement and Concordance Analysis

Kendall's coefficient of concordance $W = 0.4321$ indicates **fair** agreement among the 6 methods.

**Table 7. Provinces most frequently ranked in the top 5.**

| Province | Count | Frequency |
| :--- | ---: | ---: |
| P46 | 24 | 50.0% |
| P49 | 22 | 45.8% |
| P14 | 19 | 39.6% |
| P39 | 19 | 39.6% |
| P12 | 18 | 37.5% |
| P18 | 12 | 25.0% |
| P01 | 11 | 22.9% |
| P47 | 11 | 22.9% |
| P28 | 10 | 20.8% |
| P56 | 10 | 20.8% |

# 7. Sensitivity and Robustness Analysis

- **Overall Robustness Index:** 0.8609
- **Confidence Level:** 95%

## 7.1 Criteria Weight Sensitivity

**Table 8. Criteria weight sensitivity indices.**

| Criterion | Sensitivity | Classification |
| :--- | ---: | :---: |
| C03 | 1.0000 | High |
| C08 | 0.9905 | High |
| C07 | 0.9844 | High |
| C02 | 0.9794 | High |
| C01 | 0.9778 | High |
| C04 | 0.9719 | High |
| C06 | 0.9303 | High |
| C05 | 0.9000 | High |

## 7.2 Subcriteria Weight Sensitivity (Top 15)

**Table 9. Most influential subcriteria.**

| Subcriteria | Sensitivity |
| :--- | ---: |
| SC11 | 1.0000 |
| SC63 | 0.9969 |
| SC51 | 0.9953 |
| SC12 | 0.9953 |
| SC83 | 0.9938 |
| SC42 | 0.9937 |
| SC61 | 0.9922 |
| SC41 | 0.9901 |
| SC43 | 0.9891 |
| SC21 | 0.9875 |
| SC44 | 0.9875 |
| SC82 | 0.9875 |
| SC62 | 0.9869 |
| SC14 | 0.9856 |
| SC13 | 0.9849 |

## 7.3 Top-N Ranking Stability

**Table 10. Top-N set stability under weight perturbation.**

| Tier | Index | Percentage |
| :--- | ---: | ---: |
| Top-3 | 0.7790 | 77.9% |
| Top-5 | 1.0000 | 100.0% |
| Top-10 | 0.9910 | 99.1% |

## 7.4 Temporal Rank Stability

**Table 11. Year-to-year rank correlation.**

| Year Pair | Spearman $\rho$ | Strength |
| :--- | ---: | :---: |
| 2020-2021 | 0.3048 | Weak |
| 2021-2022 | 0.7756 | Moderate |
| 2022-2023 | 0.6417 | Moderate |
| 2023-2024 | 0.6635 | Moderate |

## 7.5 Provincial Rank Stability

**Table 12(a). Ten most volatile provinces.**

| Province | Stability |
| :--- | ---: |
| P29 | 0.9570 |
| P61 | 0.9632 |
| P56 | 0.9713 |
| P45 | 0.9715 |
| P01 | 0.9723 |
| P33 | 0.9732 |
| P32 | 0.9734 |
| P47 | 0.9753 |
| P42 | 0.9754 |
| P35 | 0.9754 |

**Table 12(b). Ten most stable provinces.**

| Province | Stability |
| :--- | ---: |
| P08 | 0.9990 |
| P38 | 0.9990 |
| P44 | 0.9990 |
| P59 | 0.9990 |
| P62 | 0.9990 |
| P06 | 1.0000 |
| P39 | 1.0000 |
| P46 | 1.0000 |
| P49 | 1.0000 |
| P58 | 1.0000 |

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

- **CRITIC** — Diakoulaki, Mavrotas & Papayannakis (1995). Weights based on contrast intensity and conflicting inter-criteria correlation.
- **Two-Level Deterministic Pipeline** — Level 1: CRITIC per criterion group → local SC weights. Level 2: CRITIC on criterion composite matrix → criterion weights. Fully deterministic — no Monte Carlo, no Beta blending.

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
- `criterion_er_scores_all_years.csv`
- `final_rankings.csv`
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
- `prediction_uncertainty_er.csv`
- `rankings_all_years.csv`
- `ranks_all_years.csv`
- `mcdm_scores_2018.csv`
- `mcdm_scores_2019.csv`
- `mcdm_scores_2020.csv`
- `mcdm_scores_2021.csv`
- `mcdm_scores_2022.csv`
- `mcdm_scores_2023.csv`
- `mcdm_scores_2024.csv`
- `perturbation_detail.csv`
- `sensitivity_criteria.csv`
- `sensitivity_rank_stability.csv`
- `sensitivity_subcriteria.csv`
- `sensitivity_temporal.csv`
- `sensitivity_top_n_stability.csv`
- `data_summary_statistics.csv`
- `criterion_weights.csv`
- `critic_criterion_weights_all_years.csv`
- `critic_sc_weights_all_years.csv`
- `critic_weights.csv`
- `critic_weights_2011.csv`
- `critic_weights_2012.csv`
- `critic_weights_2013.csv`
- `critic_weights_2014.csv`
- `critic_weights_2015.csv`
- `critic_weights_2016.csv`
- `critic_weights_2017.csv`
- `critic_weights_2018.csv`
- `critic_weights_2019.csv`
- `critic_weights_2020.csv`
- `critic_weights_2021.csv`
- `critic_weights_2022.csv`
- `critic_weights_2023.csv`
- `critic_weights_2024.csv`
- `weights_analysis.csv`

### JSON Metadata Files

- `forecast_summary.json`
- `sensitivity_summary.json`
- `config_snapshot.json`
- `execution_summary.json`

### Figures (43 files)

- `fig01_final_er_ranking.png`
- `fig01b_tier_ranking.png`
- `fig01d_belief_heatmap.png`
- `fig01e_rank_uncertainty_scatter.png`
- `fig02_score_distribution.png`
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
- `fig04_weight_radar.png`
- `fig04a_weight_radar_criteria.png`
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