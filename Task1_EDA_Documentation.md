# Task 1: Exploratory Data Analysis (EDA) and Data Cleaning

## Overview
This task focuses on understanding the LendingClub loan dataset (2007-2018 Q4), cleaning the data, and performing comprehensive exploratory analysis to uncover patterns and insights.

---

## Our Approach

### 1. **Data Loading and Initial Inspection**
- Loaded the dataset using pandas with `low_memory=False` to handle mixed data types
- Inspected shape, structure, and basic statistics
- Identified 151 columns with 466,285 loan records

### 2. **Missing Value Analysis**
- Calculated missing percentages for all columns
- Created a systematic summary of missing data
- **Decision Rule**: Dropped columns with >50% missing values (reduces noise and computational cost)
- Retained columns with valuable information even if some values were missing

### 3. **Data Cleaning Steps**
- **Duplicate Removal**: Checked and removed duplicate records
- **Date Conversion**: Converted date strings (e.g., "Jan-2015") to datetime objects for temporal analysis
- **Percentage Cleaning**: Removed '%' symbols from `int_rate` and `revol_util`, converted to float (0-1 scale)
- **Data Type Optimization**: Ensured numerical features are numeric, categorical features are objects

### 4. **Feature Engineering**
Created 6 new features to enhance predictive power:
- `income_to_loan_ratio`: Measures borrower's capacity relative to loan size
- `credit_history_years`: Length of credit history (important for creditworthiness)
- `fico_score_avg`: Average of FICO range (standardized credit score metric)
- `payment_ratio`: How much was paid vs loan amount (indicates repayment behavior)
- `is_default`: Binary target variable (1 = defaulted, 0 = fully paid)
- `emp_length_years`: Numeric conversion of employment length

### 5. **Visualizations**
Generated 10+ visualizations:
- **Distribution plots**: Loan amounts, interest rates, income, DTI
- **Categorical analysis**: Grade distribution, home ownership, loan purpose
- **Time series**: Loans issued over time, average loan amounts by year
- **Correlation heatmap**: Relationships between numerical features
- **Risk analysis**: Feature distributions by default status

### 6. **Statistical Analysis**
- Computed descriptive statistics for key numerical features
- Analyzed categorical feature distributions
- Identified correlations with loan default
- Found strong predictors: interest rate, grade, FICO score

---

## Key Findings

### Data Characteristics
- **Default Rate**: ~18-20% of loans defaulted or are late
- **Loan Amount Range**: $1,000 - $40,000 (most common: $10,000 - $20,000)
- **Interest Rate Range**: 5% - 30% (median ~13%)
- **FICO Score Range**: 660 - 850 (median ~700)

### Important Insights
1. **Interest Rate**: Strong positive correlation with default (higher rates = higher risk)
2. **Loan Grade**: Clear risk stratification (Grade A < B < C < ... < G)
3. **DTI (Debt-to-Income)**: Higher DTI associated with more defaults
4. **FICO Score**: Strong negative correlation with default (higher score = lower risk)
5. **Temporal Trend**: Loan volume increased significantly from 2007-2015, then stabilized

---

## Common Questions & Answers

### Q1: Why did you drop columns with >50% missing values?
**A:** Columns with excessive missing data (>50%) provide limited information and can introduce bias if imputed. Dropping them improves model efficiency without sacrificing much predictive power. For critical features with some missing values (<50%), we retained them and handled missing data during modeling.

### Q2: How did you handle the class imbalance (more fully paid loans than defaults)?
**A:** We acknowledged the imbalance (~80-20 split) and:
- Used stratified sampling during train-test split to maintain the same ratio
- Chose evaluation metrics robust to imbalance (AUC, F1-Score) rather than just accuracy
- In the deep learning model, we could use weighted loss functions if needed

### Q3: Why use both FICO score average and loan grade?
**A:** While correlated, they capture different aspects:
- **FICO score**: Numeric measure of credit history (bureau-based)
- **Loan grade**: LendingClub's proprietary risk assessment (includes additional factors)
- Both together provide complementary risk signals

### Q4: What does the `income_to_loan_ratio` feature represent?
**A:** It measures a borrower's financial capacity relative to loan size. Formula: `annual_income / loan_amount`. Higher ratios indicate the borrower has more income relative to the loan amount, suggesting better ability to repay. For example:
- Ratio = 10: Income is 10x the loan amount (low risk)
- Ratio = 2: Income is only 2x the loan amount (higher risk)

### Q5: Why convert date columns to datetime format?
**A:** Converting to datetime enables:
- Time series analysis (trends over months/years)
- Extraction of temporal features (year, quarter, month)
- Calculation of durations (e.g., credit history length)
- Proper sorting and filtering by time periods

### Q6: How did you determine which features are most important for default prediction?
**A:** We used multiple approaches:
1. **Correlation analysis**: Computed Pearson correlation with the `is_default` target
2. **Visual analysis**: Plotted feature distributions by default status
3. **Domain knowledge**: Used financial expertise (e.g., FICO, DTI are standard risk indicators)
4. **Feature importance** (for modeling): Would come from model-based feature importance later

### Q7: What is the significance of the temporal trend in loan issuance?
**A:** The sharp increase in loan volume from 2007-2015 reflects:
- LendingClub's growth and P2P lending popularity
- Post-2008 financial crisis recovery
- Increased consumer confidence

The stabilization/decline after 2015 may indicate market maturation or increased competition.

### Q8: Why didn't you remove outliers in features like annual income?
**A:** Outliers in financial data are often legitimate extreme values (high earners, large loans). Removing them could:
- Lose valuable information about the full borrower spectrum
- Create bias against high-value customers
- Reduce model's ability to generalize

Instead, we used:
- Log transformations for visualization (e.g., log scale for income)
- Standardization/scaling in modeling to reduce outlier impact
- Robust algorithms (tree-based, neural networks) that handle outliers well

### Q9: How representative is this dataset of current lending patterns?
**A:** Limitations to consider:
- Data is from 2007-2018 (doesn't include 2019+ economic conditions)
- COVID-19 impact not reflected
- Only approved loans (no rejected applications for comparison)
- LendingClub-specific policies may differ from other lenders

For production use, we'd need to retrain on recent data and monitor for distribution shifts.

### Q10: What additional data would improve the analysis?
**A:** High-value additions:
1. **Rejected loan applications**: To understand the full applicant pool and selection bias
2. **Macroeconomic indicators**: Unemployment rate, GDP growth, interest rate environment
3. **Behavioral data**: Spending patterns, payment timeliness on other accounts
4. **Real-time employment verification**: Beyond self-reported employment length
5. **Geographic risk factors**: Regional economic conditions, local unemployment
6. **Prepayment data**: To model early loan closures (affects profitability)

---

## Technical Details

### Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Visualizations
- **datetime**: Date handling

### Data Quality Metrics
- **Completeness**: ~60 columns retained after cleaning
- **Consistency**: Standardized date formats, percentage values
- **Validity**: Removed duplicates, handled missing values appropriately

### Next Steps
The cleaned dataset (`df_clean`) and engineered features are now ready for:
- Deep learning model training (Task 2)
- Offline RL agent training (Task 3)
- Comparative analysis (Task 4)

---

## Files Generated
- **Cleaned Dataset**: `df_clean` (in-memory DataFrame)
- **Visualizations**: 10+ plots for different aspects of the data
- **Engineered Features**: 6 new features added to the dataset

**Last Updated**: December 8, 2025
