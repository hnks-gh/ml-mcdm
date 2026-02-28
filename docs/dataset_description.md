# Dataset Description

## Overview
This document provides a comprehensive description of the annual CSV data files, which contain panel data of 29 sub-criteria scores for the governance index of 63 provinces in Vietnam from 2011-2024 (14 years). Each year's data is stored in a separate CSV file.

---

## File Information

- **File Path**: `data/YYYY.csv` (where YYYY = 2011 to 2024)
- **Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8
- **Files**: 14 separate files (2011.csv, 2012.csv, ..., 2024.csv)
- **Rows per File**: 64 (including header)
- **Data Rows per File**: 63
- **Columns**: 30 (Province and 29 sub-criteria)

---

## Dataset Structure

### Temporal Coverage
- **Start Year**: 2011
- **End Year**: 2024
- **Duration**: 14 years
- **Frequency**: Annual observations

### Spatial Coverage
- **Geographic Region**: Vietnam
- **Administrative Units**: 63 provinces
- **Province Identifiers**: P01 through P63

### Total Observations
- **Formula**: 63 provinces × 14 years = 882 observations
- **Distribution**: 63 provinces per year file × 14 year files
- **Data Structure**: Separate CSV file for each year

---

## Column Specifications

### 1. Spatial Identifier
- **Column Name**: `Province`
- **Data Type**: String (Categorical)
- **Format**: P## (where ## is a two-digit number)
- **Range**: P01 to P63
- **Purpose**: Unique identifier for each province

### 2. Sub-Criteria (29 columns)
- **Column Names**: SC11, SC12, SC13, ..., SC83
- **Data Type**: Float (Decimal numbers)
- **Value Range**: 0-3.33
- **Total Sub-Criteria**: 29 governance indicators
- **Purpose**: Provincial governance and public administration index sub-criteria

#### Hierarchical Structure
The 29 sub-criteria are organized under 8 main criteria (not present in dataset):

| Main Criteria | Sub-Criteria Range | Count | Description |
|---------------|-------------------|-------|-------------|
| Criteria 1: Participation | SC11 - SC14 | 4 | Civic engagement and electoral participation |
| Criteria 2: Transparency of Local Decision-making | SC21 - SC24 | 4 | Information access and budget transparency |
| Criteria 3: Vertical Accountability | SC31 - SC33 | 3 | Government responsiveness and justice access |
| Criteria 4: Control of Corruption | SC41 - SC44 | 4 | Anti-corruption measures and equity |
| Criteria 5: Public Administrative Procedures | SC51 - SC54 | 4 | Administrative efficiency and procedures |
| Criteria 6: Public Service Delivery | SC61 - SC64 | 4 | Healthcare, education, and infrastructure |
| Criteria 7: Environmental Governance | SC71 - SC73 | 3 | Environmental protection and quality |
| Criteria 8: E-Governance | SC81 - SC83 | 3 | Digital government and internet access |

**Note**: Each sub-criterion (SC11-SC83) represents a specific aspect of provincial governance. The values are not normalized

---

## File List

The dataset consists of 14 separate CSV files located in the `data/` directory:

- `2011.csv` - Data for year 2011
- `2012.csv` - Data for year 2012
- `2013.csv` - Data for year 2013
- `2014.csv` - Data for year 2014
- `2015.csv` - Data for year 2015
- `2016.csv` - Data for year 2016
- `2017.csv` - Data for year 2017
- `2018.csv` - Data for year 2018
- `2019.csv` - Data for year 2019
- `2020.csv` - Data for year 2020
- `2021.csv` - Data for year 2021
- `2022.csv` - Data for year 2022
- `2023.csv` - Data for year 2023
- `2024.csv` - Data for year 2024

Each file contains 63 rows (one per province) plus a header row.

---

## Additional Resources

For detailed variable descriptions, provincial names, and sub-criteria definitions, please refer to the **codebook files** located in the `data/codebook` directory:

- **`codebook_provinces.csv`** - Province codes (P01-P63) and their full names
- **`codebook_criteria.csv`** - Main criteria codes (C01-C08) and descriptions
- **`codebook_subcriteria.csv`** - Sub-criteria codes (SC11-SC83), names, and their parent criteria mapping

---

## Missing Data Report

### Overview

Analysis of all 14 annual CSV files (2011–2024) reveals three categories of missing data, totalling approximately **3,424 missing cells** out of 25,578 possible data points (63 provinces × 29 sub-criteria × 14 years) — a **13.4% overall missingness rate**.

| Category | Description |
|---|---|
| **Type 1 — Entire column missing** | A sub-criterion column is absent or all-null for an entire year |
| **Type 2 — Entire province missing** | A province row exists but all 29 sub-criteria cells are blank |
| **Type 3 — Partial province missing** | A province is present but specific sub-criteria cells are blank |

---

### Type 1: Entire Sub-Criteria Columns Missing by Year

These represent **structural gaps** caused by the governance index expanding over time (SC71–SC83 added in 2018) or a sub-criterion being discontinued (SC52 from 2021 onward).

| Year | Entirely Missing Sub-Criteria Columns | Count |
|------|---------------------------------------|-------|
| 2011 | SC24, SC71, SC72, SC73, SC81, SC82, SC83 | 7 |
| 2012 | SC24, SC71, SC72, SC73, SC81, SC82, SC83 | 7 |
| 2013 | SC24, SC71, SC72, SC73, SC81, SC82, SC83 | 7 |
| 2014 | SC24, SC71, SC72, SC73, SC81, SC82, SC83 | 7 |
| 2015 | SC24, SC71, SC72, SC73, SC81, SC82, SC83 | 7 |
| 2016 | SC24, SC71, SC72, SC73, SC81, SC82, SC83 | 7 |
| 2017 | SC24, SC71, SC72, SC73, SC81, SC82, SC83 | 7 |
| 2018 | SC83 | 1 |
| 2019 | *(none)* | 0 |
| 2020 | *(none)* | 0 |
| 2021 | SC52 | 1 |
| 2022 | SC52 | 1 |
| 2023 | SC52 | 1 |
| 2024 | SC52 | 1 |

**Key observations:**
- **SC24** (Transparency sub-criterion): absent for the entire 2011–2017 period; consistently present from 2018 onward, except for partial gaps on P14 and P56 in 2018.
- **SC71–SC73** (Environmental Governance) and **SC81–SC83** (E-Governance): introduced in 2018; SC83 first fully available in 2019.
- **SC52** (Public Administrative Procedures sub-criterion): discontinued entirely starting in 2021.

---

### Type 2: Entire Province Rows Missing

Nine province-year combinations have all 29 sub-criteria cells blank, representing completely missing annual observations.

| Year | Province(s) Missing | Count |
|------|---------------------|-------|
| 2014 | P15, P56 | 2 |
| 2021 | P14, P15, P18 | 3 |
| 2023 | P14, P47 | 2 |
| 2024 | P17, P52 | 2 |

**Total**: 9 province-year observations entirely absent = **261 missing cells** from this category alone.

---

### Type 3: Partial Province Missing Data

Four province-year cases have some sub-criteria present but specific cells blank.

#### Year 2018

| Province | Missing Sub-Criteria | Missing Count |
|----------|----------------------|---------------|
| P14 | SC21, SC22, SC23, SC24, SC41, SC42, SC43, SC44 | 8 |
| P56 | SC21, SC22, SC23, SC24, SC41, SC42, SC43, SC44 | 8 |

Both provinces are missing the Transparency (SC21–SC24) and Control of Corruption (SC41–SC44) sub-criteria.

#### Year 2022

| Province | Missing Sub-Criteria | Missing Count |
|----------|----------------------|---------------|
| P15 | SC11, SC12, SC13, SC14, SC21, SC22, SC23, SC24, SC41, SC42, SC43, SC44, SC51, SC52, SC53, SC54 | 16 |
| P18 | SC21, SC22, SC23, SC24, SC31, SC32, SC33, SC41, SC42, SC43, SC44, SC52, SC81, SC82, SC83 | 15 |

P15 is missing most of Criteria 1 (Participation), 2 (Transparency), 4 (Corruption), and 5 (Administrative Procedures). P18 is missing Criteria 2, 3 (Vertical Accountability), 4, part of 5, and Criteria 8 (E-Governance).

---

### Per-Year Missing Cell Counts

| Year | Type 1 (column-wide) | Type 2 (blank provinces) | Type 3 (partial cells) | Year Total |
|------|----------------------|--------------------------|------------------------|------------|
| 2011 | 441 | 0 | 0 | **441** |
| 2012 | 441 | 0 | 0 | **441** |
| 2013 | 441 | 0 | 0 | **441** |
| 2014 | 427 | 58 | 0 | **485** |
| 2015 | 441 | 0 | 0 | **441** |
| 2016 | 441 | 0 | 0 | **441** |
| 2017 | 441 | 0 | 0 | **441** |
| 2018 | 63 | 0 | 16 | **79** |
| 2019 | 0 | 0 | 0 | **0** |
| 2020 | 0 | 0 | 0 | **0** |
| 2021 | 60 | 87 | 0 | **147** |
| 2022 | 61 | 0 | 31 | **92** |
| 2023 | 61 | 58 | 0 | **119** |
| 2024 | 61 | 58 | 0 | **119** |
| **Total** | **3,116** | **261** | **47** | **3,424** |

> 2019 and 2020 are the only complete years with no missing data.

---

### Affected Provinces Summary

| Province | Entirely Blank In | Partially Missing In | Total Cells Missing (beyond Type 1) |
|----------|-------------------|----------------------|--------------------------------------|
| P14 | 2021, 2023 | 2018 (8 cells) | **66** |
| P15 | 2014, 2021 | 2022 (16 cells) | **74** |
| P18 | 2021 | 2022 (15 cells) | **44** |
| P47 | 2023 | — | **29** |
| P56 | 2014 | 2018 (8 cells) | **37** |
| P17 | 2024 | — | **29** |
| P52 | 2024 | — | **29** |

All other provinces (P01–P63 not listed above) have no province-specific missingness beyond the structural column gaps in Type 1.

---

### Sub-Criterion Missingness Summary

| Sub-Criterion | Years Entirely Absent | Years with Partial Absence |
|---------------|-----------------------|---------------------------|
| SC24 | 2011–2017 | 2018 (P14, P56) |
| SC52 | 2021–2024 | 2022 (P15, P18) |
| SC71 | 2011–2017 | — |
| SC72 | 2011–2017 | — |
| SC73 | 2011–2017 | — |
| SC81 | 2011–2017 | 2022 (P18) |
| SC82 | 2011–2017 | 2022 (P18) |
| SC83 | 2011–2018 | — |
| SC21, SC22, SC23 | — | 2018 (P14, P56); 2022 (P15, P18) |
| SC41, SC42, SC43, SC44 | — | 2018 (P14, P56); 2022 (P15, P18) |
| SC11, SC12, SC13, SC14 | — | 2022 (P15) |
| SC31, SC32, SC33 | — | 2022 (P18) |
| SC51, SC53, SC54 | — | 2022 (P15) |