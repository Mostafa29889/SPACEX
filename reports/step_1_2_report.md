# SpaceX Falcon 9 Landing Prediction - Step 1 & 2 Analysis Report

## 1. Dataset Understanding
- **Dataset Name:** `spacex_web_scraped.csv`
- **Shape:** 557 rows, 11 columns
- **Target Variable:** derived from `Booster landing` (Landing Outcome)
  - **Class 1 (Success):** Booster landing contains the string `Success`
  - **Class 0 (Failure):** All other landing outcomes (including no attempts, failures, crashes)

### Feature Classification & Descriptions:
| Feature Name | Data Type | Description |
| --- | --- | --- |
| `Flight No.` | Numerical (Discrete) | The sequential number of the Falcon 9 launch. |
| `Launch site` | Categorical (Nominal) | The spaceport launch pad (e.g. Cape Canaveral, VAFB, KSC). |
| `Payload` | Categorical (Nominal) | Name of the satellite or spacecraft being launched. |
| `Payload mass` | Numerical (Continuous) | Mass of the payload in kilograms. |
| `Orbit` | Categorical (Nominal) | The target destination orbit (e.g. LEO, GTO, ISS). |
| `Customer` | Categorical (Nominal) | The entity paying for the launch service. |
| `Launch outcome` | Categorical (Binary) | Success or failure of the ascent phase. |
| `Date` | Date/Time | The calendar date of the launch. |
| `Time` | Date/Time | The launch time (UTC). |
| `Version Booster` | Categorical (Nominal) | Specific rocket model/variant version. |
| `Booster landing` | Categorical (Nominal) | Landing pad method and outcome details. |

## 2. Exploratory Data Analysis (EDA) Summary

### Data Quality Metrics:
- **Duplicates:** 0 duplicate records found.
- **Missing Values (or Unknowns) per Column:**
  - `Flight No.`: 0 missing/unknown entries
  - `Launch site`: 0 missing/unknown entries
  - `Payload`: 3 missing/unknown entries
  - `Payload mass`: 33 missing/unknown entries
  - `Orbit`: 2 missing/unknown entries
  - `Customer`: 15 missing/unknown entries
  - `Launch outcome`: 0 missing/unknown entries
  - `Date`: 0 missing/unknown entries
  - `Time`: 0 missing/unknown entries
  - `Version Booster`: 0 missing/unknown entries
  - `Booster landing`: 0 missing/unknown entries

### Target Class Distribution:
- **Successful Landings (Class 1):** 508 (91.20%)
- **Unsuccessful Landings (Class 0):** 49 (8.80%)

### Summary Statistics for Payload Mass (kg):
- **Mean Mass:** 12107.70 kg
- **Median Mass:** 15525.00 kg
- **Min Mass:** 325.00 kg
- **Max Mass:** 17500.00 kg

### Landing Success Rate by Launch Site:
| Launch Site | Total Launches | Landing Successes | Landing Success Rate |
| --- | --- | --- | --- |
| Cape Canaveral | 275 | 239 | 86.91% |
| Kennedy | 106 | 99 | 93.40% |
| Vandenberg | 176 | 170 | 96.59% |

### Landing Success Rate by Orbit Type:
| Orbit Type | Total Launches | Landing Successes | Landing Success Rate |
| --- | --- | --- | --- |
|  | 2 | 1 | 50.00% |
| BLT | 1 | 1 | 100.00% |
| Ballistic lunar transfer (BLT) | 1 | 1 | 100.00% |
| GTO | 59 | 40 | 67.80% |
| HEO | 1 | 1 | 100.00% |
| Heliocentric | 2 | 1 | 50.00% |
| LEO | 382 | 362 | 94.76% |
| MEO | 13 | 11 | 84.62% |
| Molniya | 1 | 1 | 100.00% |
| Polar | 12 | 9 | 75.00% |
| Polar orbit | 1 | 0 | 0.00% |
| Retrograde | 1 | 1 | 100.00% |
| SSO | 77 | 76 | 98.70% |
| Sub-orbital | 1 | 0 | 0.00% |
| TLI | 3 | 3 | 100.00% |

