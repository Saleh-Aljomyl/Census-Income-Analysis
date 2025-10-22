# Census Income Prediction
## Identifying Characteristics of High Earners

**US Census Bureau Income Analysis**

Date: October 19, 2025

---

# Agenda

1. Project Background & Objectives
2. Dataset Overview
3. Exploratory Data Analysis
4. Data Preprocessing & Feature Engineering
5. Model Development & Comparison
6. Results & Key Findings
7. Feature Importance Analysis
8. Business Recommendations
9. Limitations & Future Work
10. Q&A

---

# 1. Project Background

## The Challenge
- US Census Bureau collects demographic and economic data to inform strategic initiatives
- Need to understand factors that influence income levels
- Binary classification: Predict whether individuals earn <$50K or ≥$50K annually

## Objectives
- Identify key characteristics associated with income levels
- Build predictive models to classify income brackets
- Provide actionable insights for policy decisions
- Deliver production-ready, replicable solution

---

# 2. Dataset Overview

## Data Characteristics
- **Source**: US Census Bureau (1994-1995)
- **Training Set**: 199,523 records
- **Test Set**: 99,762 records
- **Features**: 42 variables
  - 7 continuous (age, capital gains, weeks worked, etc.)
  - 33 nominal (education, occupation, marital status, etc.)
  - 2 derived (target income, year)

## Target Variable Distribution
- **Less than $50K**: 93.8% (187,141 records)
- **$50K or more**: 6.2% (12,382 records)
- **Challenge**: Severe class imbalance

---

# 3. Exploratory Data Analysis

## Key Observations

### Data Quality
- Minimal missing data (only 874 missing values in hispanic_origin)
- No duplicate handling required
- Clean categorical encoding needed

### Continuous Features
- Age: Range 0-90, mean 34.5 years
- Weeks worked: Highly bimodal (0 or 52 weeks)
- Capital gains/losses: Extremely right-skewed (most zeros)
- Dividends: Similar pattern to capital gains

---

# 3. EDA - Visualizations

## Income Distribution
![Target Distribution](01_target_distribution.png)

**Key Insight**: Significant class imbalance (94% vs 6%)

---

# 3. EDA - Continuous Features

![Continuous Features](02_continuous_features_distribution.png)

**Key Insights**:
- Age shows normal-like distribution
- Financial features (capital, dividends) heavily right-skewed
- Weeks worked is bimodal (unemployed vs full-time)

---

# 3. EDA - Feature vs Target

![Feature vs Target](03_feature_vs_target.png)

**Key Insights**:
- Higher earners concentrated in ages 35-55
- Education level strongly correlates with income
- Gender gap visible in income distribution
- Full-year workers (52 weeks) more likely to earn ≥$50K

---

# 3. EDA - Correlations

![Correlation Matrix](04_correlation_matrix.png)

**Key Insights**:
- Strong correlation between weeks_worked and num_persons_worked_for_employer (0.75)
- Age moderately correlates with weeks worked (0.21)
- Financial variables relatively independent

---

# 4. Data Preprocessing

## Data Cleaning
- Removed instance_weight variable (per metadata guidelines)
- Stripped whitespace from all categorical variables
- Handled missing values in hispanic_origin column
- Standardized feature names for consistency

## Encoding & Scaling
- **Label Encoding**: Applied to 29 categorical variables
- **Standardization**: Applied StandardScaler to 14 numerical features
- Ensured consistent encoding across train and test sets

---

# 5. Feature Engineering

## Created 8 New Features

1. **age_group**: Categorical bins (18-25, 26-35, 36-45, 46-55, 56-65, 65+)
2. **has_capital_gains**: Binary indicator (0/1)
3. **has_capital_losses**: Binary indicator (0/1)
4. **has_dividends**: Binary indicator (0/1)
5. **total_capital**: Combined financial metric (gains - losses + dividends)
6. **work_intensity**: Proportion of year worked (weeks/52)
7. **education_level**: Ordinal encoding (0-15 scale)
8. **is_native_born**: Binary indicator for US-born

**Result**: 47 total features for modeling

---

# 6. Model Development

## Models Tested

1. **Logistic Regression** - Linear baseline
2. **Decision Tree** - Interpretable non-linear
3. **Random Forest** - Ensemble of trees
4. **Gradient Boosting** - Sequential ensemble
5. **XGBoost** - Optimized gradient boosting

## Evaluation Strategy
- Used provided train/test split
- Multiple metrics: Accuracy, F1-Score, ROC-AUC
- **Primary Metric**: ROC-AUC (handles class imbalance)
- Cross-validation on training set

---

# 7. Model Comparison Results

![Model Comparison](05_model_comparison.png)

| Model | Test Accuracy | Test F1 | Test AUC |
|-------|--------------|---------|----------|
| Logistic Regression | 95.06% | 0.469 | **0.938** |
| Decision Tree | 95.20% | 0.528 | 0.921 |
| Random Forest | 95.53% | 0.519 | 0.951 |
| Gradient Boosting | 95.74% | 0.582 | **0.953** |
| **XGBoost** | **95.68%** | **0.568** | **0.954** |

**Winner: XGBoost** (Best ROC-AUC: 0.954)

---

# 7. ROC Curves

![ROC Curves](06_roc_curves.png)

**Key Insight**: All models significantly outperform random classifier
- XGBoost and Gradient Boosting show nearly identical ROC curves
- Excellent discrimination ability (AUC > 0.95)

---

# 7. Confusion Matrix - XGBoost

![Confusion Matrix](07_confusion_matrix.png)

## Performance Breakdown
- **True Negatives**: 92,380 (correctly identified <$50K)
- **True Positives**: 2,839 (correctly identified ≥$50K)
- **False Positives**: 1,196 (predicted high, actually low)
- **False Negatives**: 3,347 (predicted low, actually high)

---

# 7. Classification Metrics

## XGBoost Performance Details

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **<$50K** | 0.97 | 0.99 | 0.98 | 93,576 |
| **≥$50K** | 0.75 | 0.46 | 0.57 | 6,186 |

### Interpretation
- **Precision (≥$50K)**: 75% - When model predicts high income, correct 3/4 times
- **Recall (≥$50K)**: 46% - Model identifies less than half of actual high earners
- **Trade-off**: Conservative predictions due to class imbalance

---

# 8. Feature Importance

![Feature Importance](08_feature_importance.png)

**Top 5 Predictors**:
1. **Education Level** (20.6%) - Dominant factor
2. **Weeks Worked per Year** (18.7%) - Work consistency
3. **Sex** (12.2%) - Gender disparity
4. **Total Capital** (9.3%) - Financial assets
5. **Occupation** (5.1%) - Job type

---

# 8. Key Findings - Education

## Education is the #1 Predictor (20.6% importance)

### Income by Education Level
- **Advanced Degrees** (PhD, Professional, Masters): Highest probability of ≥$50K
- **Bachelor's Degree**: Strong positive correlation
- **High School or Less**: Strong negative correlation

### Actionable Insight
- Education policy investments have measurable economic impact
- Support for higher education access could increase income levels
- Focus on STEM and professional degree programs

---

# 8. Key Findings - Employment

## Weeks Worked is #2 Predictor (18.7% importance)

### Work Patterns
- **52 weeks (full-time)**: Strongly associated with ≥$50K
- **Part-time/Seasonal**: Associated with <$50K
- **Unemployed (0 weeks)**: Almost always <$50K

### Actionable Insight
- Employment stability programs crucial
- Focus on reducing unemployment gaps
- Support for year-round employment initiatives

---

# 8. Key Findings - Demographics

## Gender Disparity (#3, 12.2% importance)

### Gender Gap
- Males disproportionately represented in ≥$50K category
- Significant income inequality by gender

### Age Patterns
- Peak earning ages: 35-55 years
- Young (<25) and older (>65) earn less

### Actionable Insight
- Gender equity policies needed
- Support for equal pay initiatives
- Focus on mid-career advancement programs

---

# 8. Key Findings - Financial Assets

## Capital & Investments (#4, 9.3% importance)

### Patterns
- Most <$50K earners have zero capital gains/dividends
- Presence of investments strongly predicts ≥$50K
- Total capital (gains + dividends - losses) is powerful predictor

### Interpretation
- Wealth accumulation correlates with income
- Financial literacy and investment access important
- Compound effect: higher income enables investment, investments indicate higher income

---

# 9. Business Recommendations

## For Policy Makers

### Education Investment
- Expand access to higher education programs
- Support for vocational and professional training
- Student loan assistance and scholarship programs

### Employment Stability
- Job creation focusing on full-time positions
- Unemployment assistance and job training
- Support for career advancement

### Equity Initiatives
- Address gender-based income disparities
- Equal pay enforcement
- Minority community support programs

---

# 9. Technical Recommendations

## Model Improvements

1. **Address Class Imbalance**
   - Implement SMOTE (Synthetic Minority Over-sampling)
   - Adjust classification threshold from 0.5
   - Use class weights in model training

2. **Hyperparameter Optimization**
   - Grid search or Bayesian optimization for XGBoost
   - Optimize for recall on high-income class

3. **Feature Engineering**
   - Create interaction terms (education × occupation)
   - Polynomial features for age
   - Geographic clustering

---

# 9. Further Analysis

## Deep Dive Opportunities

1. **Regional Analysis**
   - State and region impact on income
   - Cost of living adjustments
   - Urban vs rural differences

2. **Temporal Trends**
   - Year-over-year changes (1994 vs 1995)
   - Longitudinal analysis if more data available

3. **Interaction Effects**
   - Education × Gender × Occupation
   - Age × Industry interactions

4. **Interpretability**
   - SHAP values for individual predictions
   - Partial dependence plots

---

# 10. Limitations

## Current Analysis Constraints

1. **Class Imbalance** (93.8% vs 6.2%)
   - Limits recall on minority class
   - May not generalize well to balanced populations

2. **Temporal Scope** (1994-1995 data)
   - Economic conditions have changed significantly
   - May not reflect current labor market

3. **Feature Limitations**
   - No geographic granularity below state level
   - Missing cost-of-living adjustments
   - No household composition details

4. **Correlation vs Causation**
   - Findings show associations, not causal relationships
   - Cannot infer policy interventions will have predicted effects

---

# 10. Future Work

## Next Steps

1. **Model Deployment**
   - Create REST API for real-time predictions
   - Build monitoring dashboard
   - Implement model retraining pipeline

2. **Enhanced Features**
   - Integrate cost-of-living data
   - Add industry growth rates
   - Include economic indicators (GDP, unemployment rate)

3. **Advanced Techniques**
   - Neural networks for complex patterns
   - Ensemble stacking
   - Automated feature selection

4. **Business Integration**
   - Build decision support tool
   - Create scenario analysis capability
   - Develop ROI calculator for policy interventions

---

# Summary

## What We Learned

**Top 3 Income Predictors**:
1. Education Level (20.6%)
2. Weeks Worked per Year (18.7%)
3. Gender (12.2%)

**Best Model**: XGBoost
- 95.68% Accuracy
- 0.954 ROC-AUC
- 75% Precision on high earners

**Key Insight**: Education and employment stability are the strongest levers for income improvement

---

# Deliverables

## Project Outputs

1. **Production-Ready Code**
   - Modular, well-documented Python pipeline
   - Replicable analysis workflow
   - ~700 lines of clean, commented code

2. **Comprehensive Visualizations**
   - 8 publication-quality charts
   - Clear, interpretable graphics
   - Ready for stakeholder presentation

3. **Documentation**
   - Executive summary
   - Technical methodology
   - Results interpretation

4. **Insights**
   - Actionable recommendations
   - Policy implications
   - Future research directions

---

# Questions & Discussion

## Contact Information

- **Code Repository**: Available for review
- **Data**: US Census Bureau public dataset
- **Reproducibility**: All results fully replicable

## Discussion Topics

- Model selection rationale
- Feature engineering choices
- Handling class imbalance
- Real-world deployment considerations
- Policy implications
- Technical deep dives

---

# Thank You

## Key Takeaways

1. **Education matters most** - Invest in accessible higher education
2. **Employment stability critical** - Support full-time, year-round work
3. **Gender equity needed** - Address systematic income disparities
4. **Predictive models work** - 95%+ accuracy achievable
5. **Data-driven policy** - Census data enables evidence-based decisions

**Questions?**
