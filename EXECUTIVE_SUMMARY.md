# Census Income Analysis - Executive Summary

## Project Overview

**Objective**: Identify characteristics associated with individuals earning more or less than $50,000 per year using US Census data.

**Dataset**: ~300,000 anonymized records from the US Census Bureau with 42 features including demographics, employment status, education, and financial attributes.

**Target Variable**: Binary classification - Income less than $50K vs. $50K or more

---

## Key Findings

### 1. Target Variable Distribution

- **Class Imbalance**: 93.8% earn <$50K, only 6.2% earn ≥$50K
- This significant imbalance required careful model evaluation using appropriate metrics (F1-score, ROC-AUC) rather than just accuracy

### 2. Most Important Predictive Features

Based on the best performing model (XGBoost), the top characteristics associated with income level are:

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | Education Level | 20.6% | Highest predictor - advanced degrees strongly correlate with higher income |
| 2 | Weeks Worked per Year | 18.7% | Work consistency is critical - full-year employment indicates higher earnings |
| 3 | Sex | 12.2% | Gender disparity exists in income distribution |
| 4 | Total Capital | 9.3% | Capital gains, losses, and dividends significantly impact income prediction |
| 5 | Occupation | 5.1% | Type of occupation matters substantially |

### 3. Model Performance Comparison

| Model | Test Accuracy | Test F1-Score | Test ROC-AUC |
|-------|---------------|---------------|--------------|
| **XGBoost** | **95.68%** | **0.568** | **0.954** |
| Gradient Boosting | 95.74% | 0.582 | 0.953 |
| Random Forest | 95.53% | 0.519 | 0.951 |
| Logistic Regression | 95.06% | 0.469 | 0.938 |
| Decision Tree | 95.20% | 0.528 | 0.921 |

**Best Model: XGBoost** (selected based on Test ROC-AUC score)

### 4. Model Performance Details

For the XGBoost model:
- **Overall Accuracy**: 95.68%
- **Precision for ≥$50K**: 75% (when model predicts high income, it's correct 75% of the time)
- **Recall for ≥$50K**: 46% (model identifies 46% of high earners)
- **ROC-AUC**: 0.954 (excellent discrimination ability)

---

## Demographic Insights

### Education
- Clear correlation between education level and income
- Individuals with Bachelor's, Master's, or Professional degrees have substantially higher probability of earning ≥$50K
- High school education or less strongly associated with <$50K income

### Work Patterns
- Working 52 weeks per year (full-time, year-round employment) is strongly associated with higher income
- Part-time or intermittent work correlates with lower income

### Age
- Middle-aged individuals (35-55) have higher representation in the ≥$50K category
- Very young (<25) and older (>65) age groups predominantly earn <$50K

### Gender
- Significant gender disparity in income distribution
- Males have disproportionately higher representation in the ≥$50K category

### Capital & Investments
- Presence of capital gains, dividends from stocks, and net capital position are strong indicators of higher income
- Most individuals earning <$50K have zero or minimal capital gains/dividends

---

## Technical Approach

### 1. Exploratory Data Analysis
- Analyzed distribution of 42 features across ~300K records
- Identified class imbalance (93.8% vs 6.2%)
- Examined correlations between continuous features
- Visualized key relationships between features and target variable

### 2. Data Preprocessing
- **Verified train/test representativeness**: Confirmed similar distributions across both sets
- Removed instance weight (as per metadata guidelines)
- Handled missing values in hispanic_origin column
- Cleaned whitespace from categorical variables
- Applied label encoding to 29 categorical features
- **Standardized numerical features** using StandardScaler (chosen over MinMaxScaler for robustness to outliers)
- **No data leakage**: Scaler fitted only on training data, then transformed on test data

### 3. Feature Engineering
Created 8 new features with clear rationale:
- **Age groups** (categorical binning) - Based on standard demographic lifecycle stages
- **Binary indicators** for capital gains/losses/dividends - Presence often more predictive than amount
- **Total capital** (combined financial metrics) - Net capital position for comprehensive view
- **Work intensity** (weeks worked / 52) - Normalized for easy interpretation
- **Education level** (ordinal encoding) - Preserves natural hierarchy of education
- **Native born status** - US citizenship indicator

### 4. Model Development
Trained and evaluated 5 different algorithms with justified hyperparameters:
- **Logistic Regression** (baseline linear model)
- **Decision Tree** (interpretable non-linear, max_depth=10)
- **Random Forest** (ensemble method, max_depth=15 - deeper for bagging)
- **Gradient Boosting** (sequential ensemble, max_depth=5 - shallow for boosting)
- **XGBoost** (optimized gradient boosting, max_depth=5 - weak learners)

**Note**: Different max_depth values are intentional - bagging methods benefit from deeper trees while boosting uses shallow weak learners.

### 5. Evaluation Strategy
- **Verified train/test representativeness** before modeling
- Used stratified train/test split provided in dataset
- Evaluated using multiple metrics: Accuracy, F1-Score, ROC-AUC
- Focused on ROC-AUC as primary metric due to class imbalance
- **Class Imbalance Approach**: No resampling (SMOTE/undersampling) used
  - Rationale: Large dataset provides sufficient minority samples (~12K)
  - Real-world distribution preserved for better generalization
  - Robust metrics (F1, ROC-AUC) used instead
- Generated confusion matrices and ROC curves
- Analyzed feature importance for interpretability

---

## Recommendations

### For Policy Makers
1. **Education Investment**: Strong correlation between education and income suggests continued investment in accessible higher education programs
2. **Employment Stability**: Programs supporting full-time, year-round employment could help increase income levels
3. **Gender Equity**: Address gender-based income disparities through policy interventions

### For Further Analysis
1. **Deep Dive on Gender Gap**: Analyze interaction effects between gender, occupation, and industry
2. **Regional Analysis**: Examine geographic patterns (state/region) and their impact on income
3. **Temporal Trends**: Analyze differences between years 1994 and 1995 (year feature was excluded from current analysis)
4. **Cost-Benefit Analysis**: Evaluate ROI of different education levels accounting for opportunity costs

### For Model Improvement
1. **Hyperparameter Tuning**: Optimize XGBoost parameters using grid search or Bayesian optimization
2. **Address Class Imbalance**: Implement SMOTE, class weighting, or threshold tuning to improve recall for high earners
3. **Feature Interactions**: Create interaction terms between key features (e.g., education × occupation)
4. **Ensemble Stacking**: Combine predictions from multiple models for potentially better performance
5. **Interpretability**: Apply SHAP values for more detailed feature contribution analysis

---

## Limitations

1. **Class Imbalance**: Severe imbalance (93.8% vs 6.2%) makes predicting high earners challenging
2. **Temporal Scope**: Data from 1994-1995 may not reflect current economic conditions
3. **Threshold Selection**: Current 50/50 probability threshold may not be optimal for business objectives
4. **Missing Data**: Hispanic origin had 874 missing values (handled via label encoding)
5. **Causality**: Analysis shows correlations, not causal relationships

---

## Deliverables

1. **Code**: Production-ready Python script with comprehensive documentation
2. **Visualizations**: 8 charts covering EDA, model comparison, and feature importance
3. **Results**: Detailed model performance metrics and comparison tables
4. **Documentation**: This executive summary and presentation materials

---

## Conclusion

The analysis successfully identified key characteristics associated with income levels using US Census data. **Education level** and **weeks worked per year** emerged as the strongest predictors, followed by gender and capital/investment status. The **XGBoost model** achieved excellent performance (95.4% ROC-AUC) in distinguishing between income classes.

The findings provide actionable insights for policy decisions around education, employment, and economic equity. The production-ready code pipeline enables easy replication and extension of this analysis.

---

**Date**: October 19, 2025
**Analysis Tool**: Python (scikit-learn, XGBoost, pandas, matplotlib, seaborn)
**Dataset**: US Census Income (199,523 training records, 99,762 test records)
