# Branch-wise Analysis and Prediction of Student Placement Using Academic and Skill Factors

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Project Objectives](#project-objectives)
3. [Dataset Description](#dataset-description)
4. [Project Structure](#project-structure)
5. [Data Cleaning Pipeline](#data-cleaning-pipeline)
6. [Feature Engineering](#feature-engineering)
7. [Imputation Strategy](#imputation-strategy)
8. [Encoding and Preprocessing](#encoding-and-preprocessing)
9. [Final Dataset Summary](#final-dataset-summary)
10. [Tech Stack](#tech-stack)
11. [How to Run](#how-to-run)
12. [Results and Output](#results-and-output)
13. [Key Decisions and Rationale](#key-decisions-and-rationale)
14. [Limitations](#limitations)
15. [Authors](#authors)

---

## Problem Statement

The objective of this study is to predict whether a college student will be placed or not, while simultaneously analyzing placement patterns across different academic branches. By examining factors such as CGPA, communication skills, projects completed, internship experience, and academic branch, this project develops a predictive model to estimate placement probability and identifies branches with higher placement success rates.

---

## Project Objectives

- Predict whether a student will receive a placement offer based on academic and skill-related features.
- Analyze placement patterns across different engineering and technology branches.
- Handle severely corrupted, noisy, and inconsistently formatted real-world data through a robust cleaning and imputation pipeline.
- Produce a fully numeric, model-ready dataset for downstream classification tasks.

---

## Dataset Description

| Property | Details |
|---|---|
| Source File | Raw_College_student_placement_dataset.csv |
| Raw Records | 43,500 rows |
| Raw Columns | 21 columns |
| Final Records | 40,307 rows |
| Final Columns | 10 columns |
| Target Variable | Placement (1 = Placed, 0 = Not Placed) |

### Raw Columns (Original)

| Column | Description |
|---|---|
| College_ID | Unique college identifier |
| IQ | Student IQ score |
| Prev_Sem_Result | Previous semester result |
| CGPA | Cumulative Grade Point Average |
| Academic_Performance | Academic performance score (0–10) |
| Internship_Experience | Whether the student had internship experience (Yes/No) |
| Extra_Curricular_Score | Score for extracurricular activities (0–10) |
| Communication_Skills | Communication skills rating (0–10) |
| Projects_Completed | Number of projects completed |
| Placement | Target — whether student was placed |
| college_name | Name of the college |
| sem_1_result to sem_8_result | Semester-wise results |
| branch | Academic branch of the student |
| KT | Number of backlogs (KT = Keep Term) |

### Final Columns (After Processing)

| Column | Type | Description |
|---|---|---|
| IQ | Float | Student IQ score, cleaned and clipped to [50, 200] |
| Academic_Performance | Float | Academic performance score (0–10) |
| Internship_Experience | Integer | Binary — 1 = Yes, 0 = No |
| Extra_Curricular_Score | Float | Extracurricular score (0–10) |
| Communication_Skills | Float | Communication rating (0–10) |
| Projects_Completed | Integer | Count of projects completed |
| branch | String / Encoded | Standardized branch name or encoded integer |
| KT | Integer | Number of backlogs |
| CGPA_Final | Float | Reconstructed or original CGPA (0–10) |
| Placement | Integer | Target variable — 1 = Placed, 0 = Not Placed |

---

## Project Structure

```
Placement-Prediction/
│
├── Raw_College_student_placement_dataset.csv   # Original raw dataset
├── cleaned_placement_data.csv                  # Cleaned and imputed dataset
├── scaled_placement_data.csv                   # Feature-scaled dataset for modelling
├── data_modified.ipynb                         # Data cleaning and preprocessing pipeline
├── eda.ipynb                                   # Exploratory Data Analysis
├── placement_model.ipynb                       # Model training and evaluation
└── README.md                                   # Project documentation
```

---

## Data Cleaning Pipeline

The raw dataset contained severely corrupted values far beyond standard missing data. Each column required a tailored cleaning strategy.

### Issues Identified Per Column

| Column | Issues Found |
|---|---|
| IQ | Scientific notation (1e309), text values ("one hundred"), salary-like values ("5k"), extreme outliers (999999999, -1e308) |
| CGPA | Text junk ("zero.zero", "ten_thousand"), separator characters ("-"), values exceeding valid range |
| sem_1 to sem_8 | Text entries ("absent", "fail/pass", "zero.zero"), values outside [0, 10] |
| Internship_Experience | Mixed case and spacing ("Yes", "y", "YES", "Ye s"), HTML-injected strings |
| Placement | Inconsistent labels ("placed", "Placde", "NO", "N", "YES", empty string) |
| branch | Abbreviations ("C.S.E", "cse"), typos ("Compter Scince"), pipe-delimited junk |

### Cleaning Functions Applied

**clean_cgpa(value)**

- Converts value to lowercase string.
- Returns NaN for known junk tokens: "zero.zero", "ten_thousand", "?", "nan", "none", "".
- Returns NaN for values ending with "k" (salary-format entries).
- Returns NaN for values outside the range [0, 10] or negative values.

Applied to: CGPA, Extra_Curricular_Score, Communication_Skills, Academic_Performance, Projects_Completed, KT.

**clean_cgpa_value(val)**

- Extended version with explicit text mapping (e.g., "zero.zero" → 0.0).
- Clips result to [0, 10] range.
- Returns NaN for non-parseable strings.

Applied to: sem_1_result through sem_8_result.

**clean_iq(value)**

- Converts to lowercase string.
- Returns NaN for text tokens and "k"-suffixed values.
- Converts to float, then clips to valid IQ range [50, 200].
- Rejects infinity-like scientific notation values.

Applied to: IQ.

**clean_binary(value)**

- Standardizes binary Yes/No columns.
- Maps any string starting with "y" or containing "yes" → 1.
- Maps any string starting with "n", containing "no", or equal to "0" → 0.
- Maps strings starting with "p" (e.g., "placed", "Placde") → 1.
- Returns NaN for all unrecognized values (HTML junk, random strings).

Applied to: Internship_Experience, Placement.

**clean_branch(value)**

- Keyword-based categorization using lowercase string matching.
- Maps to five standardized branch names.
- Returns NaN for unrecognized or junk values.

| Keywords Detected | Standardized Label |
|---|---|
| "cs", "computer" | Computer Science |
| "civil" | Civil Engineering |
| "mech" | Mechanical Engineering |
| "elect", "ece", "eee" | Electronics Engineering |
| "it" (without "cs") | Information Technology |

### Target Column Handling

Rows where `Placement` was NaN after the initial read were dropped before any further processing, as the target variable is mandatory for supervised learning.

---

## Feature Engineering

### CGPA Reconstruction

A significant portion of the CGPA column was missing (13.26%). However, many of these students had partial or complete semester result data available. A row-wise mean was computed across all eight semester columns to derive a proxy CGPA.

```
CGPA_Final = CGPA (if present) else mean(sem_1 to sem_8)
```

| Metric | Before | After |
|---|---|---|
| CGPA missing rate | 13.26% | 0.23% |
| Coverage | 86.74% | 99.77% |

This domain-aware reconstruction significantly reduced missingness in the most important continuous predictor without introducing external assumptions.

---

## Imputation Strategy

After cleaning, the remaining missing rates were as follows:

| Column | Missing Rate |
|---|---|
| IQ | 62.41% |
| Academic_Performance | 55.71% |
| Extra_Curricular_Score | 54.54% |
| Communication_Skills | 54.99% |
| Projects_Completed | 56.98% |
| KT | 43.03% |
| branch | 8.93% |
| Placement | 9.75% |
| CGPA_Final | 0.23% |
| Internship_Experience | 8.91% |

### Method: MICE (Multiple Imputation by Chained Equations)

MICE was selected over simpler alternatives (mean, median, mode) for the following reasons:

- Mean/median imputation on columns with 50–60% missingness would collapse the variance of those features, introducing severe bias.
- MICE treats each column with missing values as a dependent variable and regresses it against all other columns iteratively, producing statistically consistent estimates.
- The iterative process (max_iter=10) ensures convergence across all columns simultaneously.

**Estimator Used:** BayesianRidge (scikit-learn default for IterativeImputer)

BayesianRidge was preferred over RandomForest for the following reasons:

- Features in this dataset (scores, IQ, CGPA) are approximately linearly correlated.
- RandomForest with n_estimators=100 would provide only marginal accuracy gains while increasing computation time by a factor of 10 to 30 on a 40,000-row dataset.
- BayesianRidge is robust, fast, and well-suited for continuous features with linear relationships.

**Configuration:**

```python
IterativeImputer(
    max_iter     = 10,
    random_state = 42,
    min_value    = 0
)
```

### Post-Imputation Corrections

| Column | Correction Applied |
|---|---|
| Internship_Experience | Rounded to nearest integer, cast to 0 or 1 |
| Placement | Rounded to nearest integer, cast to 0 or 1 |
| KT | Rounded to nearest integer |
| Projects_Completed | Rounded to nearest integer |
| IQ | Clipped to [50, 200], rounded |
| Academic_Performance | Clipped to [0, 10] |
| Extra_Curricular_Score | Clipped to [0, 10] |
| Communication_Skills | Clipped to [0, 10] |
| CGPA_Final | Clipped to [0, 10] |
| branch | Rounded encoded float to nearest class index, decoded back to string |

---

## Encoding and Preprocessing

### Branch Encoding

The branch column was encoded using LabelEncoder after standardization. The final mapping is:

| Branch Label | Encoded Value |
|---|---|
| Civil Engineering | 0 |
| Computer Science | 1 |
| Electronics Engineering | 2 |
| Information Technology | 3 |
| Mechanical Engineering | 4 |

Encoding choice for downstream modelling should follow these guidelines:

| Model Type | Recommended Encoding |
|---|---|
| Random Forest, XGBoost, Decision Tree | Label Encoding |
| Logistic Regression, SVM, Neural Networks | One-Hot Encoding |

### Binary Columns

Both `Internship_Experience` and `Placement` were standardized to integer binary values: 1 (Yes / Placed) and 0 (No / Not Placed).

---

## Final Dataset Summary

| Property | Value |
|---|---|
| Total rows | 40,307 |
| Total columns | 10 |
| Null values remaining | 0 |
| Target variable | Placement |
| Output file | cleaned_placement_data.csv |

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.13 | Core language |
| pandas | Latest | Data loading, manipulation, and export |
| numpy | Latest | Numerical operations and NaN handling |
| scikit-learn | Latest | IterativeImputer (MICE), LabelEncoder, model training |
| matplotlib | Latest | Distribution visualization |
| seaborn | Latest | Statistical plotting |

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Steps

1. Clone the repository and navigate to the project directory.
2. Ensure `Raw_College_student_placement_dataset.csv` is present in the root directory.
3. Run the notebooks in the following order:
   - `data_modified.ipynb` — Data cleaning and preprocessing
   - `eda.ipynb` — Exploratory Data Analysis
   - `placement_model.ipynb` — Model training and evaluation
4. The cleaned output `cleaned_placement_data.csv` and scaled output `scaled_placement_data.csv` will be generated in the working directory.

### Verify Output

```python
import pandas as pd

df = pd.read_csv('cleaned_placement_data.csv')
print(df.shape)
print(df.isnull().sum())
print(df.head())
```

---

## Results and Output

Upon successful execution of the full pipeline, the following is produced:

- A fully cleaned and imputed DataFrame with zero null values.
- All features in numeric format, ready for direct use in classification models.
- Branch labels standardized to five canonical engineering disciplines.
- Binary columns (`Placement`, `Internship_Experience`) encoded as integer 0/1.
- A feature-scaled dataset (`scaled_placement_data.csv`) for use with distance-sensitive models.
- Exported CSV: `cleaned_placement_data.csv`

### Recommended Next Steps

- Analyze placement rates by branch using groupby aggregations.
- Train and compare classification models: Logistic Regression, Random Forest, XGBoost.
- Evaluate models using: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Conduct feature importance analysis to identify the top predictors of placement.

---

## Key Decisions and Rationale

| Decision | Rationale |
|---|---|
| Dropped rows where Placement is NaN | Target variable is mandatory for supervised learning |
| Used CGPA row-mean from semesters | Domain knowledge — CGPA is the average of semester grades |
| Chose BayesianRidge over RandomForest for MICE | Linear features, large dataset, negligible accuracy difference |
| Clipped IQ to [50, 200] | Standard psychometric IQ range; values outside are data errors |
| Clipped scores to [0, 10] | All academic scores in this dataset are on a 10-point scale |
| Keyword-based branch cleaning | Handles abbreviations, typos, and junk uniformly without manual enumeration |
| random_state=42 | Ensures reproducibility across all runs |

---

## Limitations

- Columns with 50–60% missing data (IQ, Academic_Performance, etc.) are largely imputed. While MICE is statistically sound, imputed values at this scale should be interpreted with caution.
- The branch cleaning function uses keyword matching. Rare or highly corrupted branch names that do not contain recognizable keywords will be assigned NaN and subsequently imputed.
- The binary encoding of `Internship_Experience` and `Placement` using string-start matching may misclassify edge cases where junk strings coincidentally begin with "y" or "n".
- This pipeline assumes the raw data schema remains consistent. A different version of the source CSV may require adjustments to the cleaning functions.

---

## Authors

Prepared as part of an academic data science project on student placement prediction.

- Dataset: Raw College Student Placement Dataset
- Notebooks: `data_modified.ipynb`, `eda.ipynb`, `placement_model.ipynb`
