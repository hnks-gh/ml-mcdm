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