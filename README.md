# Census Income Analysis - Dataiku Technical Assessment

## Project Overview

This project analyzes US Census data to identify characteristics associated with individuals earning more or less than $50,000 per year. The analysis uses machine learning techniques to build predictive models and extract insights about income determinants.

**Objective**: Build a production-ready data science pipeline to predict income levels and identify key socioeconomic factors.

## Repository Structure

```
.
├── census_income_analysis.py          # Main analysis pipeline (production-ready)
├── census_income_learn.csv            # Training dataset (199,523 records)
├── census_income_test.csv             # Test dataset (99,762 records)
├── census_income_metadata.txt         # Dataset documentation
├── ME__Dataiku_Data_Scientist_Technical_Assessment_and_Presentation.pdf
├── README.md                          # This file
├── EXECUTIVE_SUMMARY.md               # Executive summary of findings
├── PRESENTATION_SLIDES.md             # Presentation deck
│
├── Visualizations (Generated):
│   ├── 01_target_distribution.png
│   ├── 02_continuous_features_distribution.png
│   ├── 03_feature_vs_target.png
│   ├── 04_correlation_matrix.png
│   ├── 05_model_comparison.png
│   ├── 06_roc_curves.png
│   ├── 07_confusion_matrix.png
│   └── 08_feature_importance.png
```

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Required Packages
- `pandas >= 1.3.0` - Data manipulation
- `numpy >= 1.21.0` - Numerical operations
- `matplotlib >= 3.4.0` - Visualization
- `seaborn >= 0.11.0` - Statistical visualization
- `scikit-learn >= 1.0.0` - Machine learning models
- `xgboost >= 1.5.0` - Gradient boosting

## Quick Start

### 1. Run Complete Analysis

Simply execute the main script:

```bash
python census_income_analysis.py
```

This will:
- Load and explore the data
- Generate all visualizations (8 PNG files)
- Train and evaluate 5 different models
- Output performance metrics
- Create feature importance analysis

**Expected Runtime**: 5-10 minutes (depending on hardware)

### 2. View Results

After execution, check:
- **Console Output**: Detailed metrics and progress
- **PNG Files**: 8 visualizations in current directory
- **EXECUTIVE_SUMMARY.md**: High-level findings
- **PRESENTATION_SLIDES.md**: Full presentation deck

## Pipeline Components

### 1. Data Loading

```python
train_df, test_df = load_data()
```

Loads both training and test datasets with proper column names.

### 2. Exploratory Data Analysis

```python
explorer = CensusDataExplorer(train_df)
explorer.basic_info() \
       .target_distribution() \
       .continuous_features_analysis() \
       .feature_vs_target() \
       .correlation_analysis()
```

Performs comprehensive EDA with visualizations.

### 3. Data Preprocessing

```python
preprocessor = CensusDataPreprocessor(train_df, test_df)
preprocessor.clean_data() \
            .engineer_features()
X_train, X_test, y_train, y_test = preprocessor.encode_and_scale()
```

Cleans data, engineers features, and prepares for modeling.

### 4. Model Training

```python
trainer = CensusModelTrainer(X_train, X_test, y_train, y_test)
trainer.initialize_models() \
       .train_and_evaluate()
```

Trains 5 models and evaluates performance.

### 5. Model Comparison

```python
_, best_model_name = trainer.compare_models()
trainer.plot_roc_curves() \
       .confusion_matrix_best_model(best_model_name) \
       .feature_importance_analysis(best_model_name, X_train.columns)
```

Compares models and analyzes best performer.

## Key Features

### Data Preprocessing
- ✅ **Verifies train/test representativeness** - Ensures similar distributions
- ✅ Handles missing values
- ✅ Removes instance weight (per metadata)
- ✅ Label encoding for categorical variables
- ✅ **StandardScaler for numerical features** (chosen over MinMaxScaler for outlier robustness)
- ✅ **No data leakage** - Scaler fitted on train only, transformed on test
- ✅ Train/test consistency ensured

### Feature Engineering
Creates 8 new features with clear justification:
1. `age_group` - Categorical age bins (based on standard lifecycle stages)
2. `has_capital_gains` - Binary indicator (presence vs amount)
3. `has_capital_losses` - Binary indicator
4. `has_dividends` - Binary indicator
5. `total_capital` - Combined financial metric (net capital position)
6. `work_intensity` - Proportion of year worked (normalized 0-1)
7. `education_level` - Ordinal encoding (0-15, preserves hierarchy)
8. `is_native_born` - US-born indicator

### Models Implemented
1. **Logistic Regression** - Linear baseline model
2. **Decision Tree** - Interpretable non-linear model (max_depth=10)
3. **Random Forest** - Bagging ensemble (max_depth=15 for deeper trees)
4. **Gradient Boosting** - Sequential boosting (max_depth=5 for weak learners)
5. **XGBoost** - Optimized gradient boosting (max_depth=5) ⭐ Best performer

**Note**: Different max_depth values intentional - bagging benefits from deeper trees, boosting uses shallow weak learners.

### Evaluation Metrics
- Accuracy
- F1-Score (robust to class imbalance)
- ROC-AUC (primary metric - robust to class imbalance)
- Precision/Recall
- Confusion Matrix
- ROC Curves

### Class Imbalance Strategy
- ✅ **No resampling applied** (SMOTE/undersampling not used)
- ✅ **Rationale**: Large dataset (~12K minority samples) + robust metrics + real-world distribution preservation
- ✅ Alternative: Could add `class_weight='balanced'` in future iterations

## Results Summary

### Best Model: XGBoost

| Metric | Value |
|--------|-------|
| Test Accuracy | 95.68% |
| Test F1-Score | 0.568 |
| Test ROC-AUC | 0.954 |
| Precision (≥$50K) | 75% |
| Recall (≥$50K) | 46% |

### Top 5 Predictive Features

1. **Education Level** (20.6% importance)
2. **Weeks Worked per Year** (18.7%)
3. **Sex** (12.2%)
4. **Total Capital** (9.3%)
5. **Detailed Occupation** (5.1%)

## Key Findings

1. **Education is the strongest predictor** - Advanced degrees dramatically increase probability of high income
2. **Employment stability matters** - Full-year employment (52 weeks) strongly correlates with higher income
3. **Gender disparity exists** - Significant income gap between males and females
4. **Capital/investments important** - Presence of capital gains and dividends indicates higher income
5. **Class imbalance challenge** - Only 6.2% earn ≥$50K, affecting model recall

## Code Quality Features

### Production-Ready
- ✅ Modular, object-oriented design
- ✅ Comprehensive docstrings
- ✅ Type hints where applicable
- ✅ Error handling
- ✅ Consistent naming conventions
- ✅ Logging and progress output

### Best Practices
- ✅ Separation of concerns (EDA, preprocessing, modeling)
- ✅ Method chaining for clean workflow
- ✅ Reusable classes and functions
- ✅ Reproducible (random_state set)
- ✅ Efficient memory usage
- ✅ Scalable architecture

### Documentation
- Inline comments for complex logic
- Detailed docstrings for all functions
- Clear variable names
- Section headers for organization

## Customization

### Modify Models

Edit the `initialize_models()` method in `CensusModelTrainer`:

```python
self.models = {
    'Your Model': YourClassifier(param1=value1, param2=value2),
    # Add more models...
}
```

### Adjust Feature Engineering

Edit the `engineer_features()` method in `CensusDataPreprocessor`:

```python
# Add your custom features
df['your_feature'] = your_transformation(df['existing_feature'])
```

### Change Evaluation Metrics

Edit the `train_and_evaluate()` method to add custom metrics:

```python
from sklearn.metrics import your_metric
custom_score = your_metric(y_test, y_pred)
```

## Troubleshooting

### Memory Issues

If you encounter memory errors:

1. Reduce number of parallel jobs:
   ```python
   RandomForestClassifier(n_jobs=1)  # Instead of n_jobs=-1
   ```

2. Reduce model complexity:
   ```python
   RandomForestClassifier(n_estimators=50, max_depth=10)
   ```

### Missing Dependencies

```bash
pip install --upgrade pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Data File Not Found

Ensure data files are in the same directory as the script:
- `census_income_learn.csv`
- `census_income_test.csv`
- `census_income_metadata.txt`

## Performance Optimization

### For Faster Execution
1. Reduce number of models (edit `initialize_models()`)
2. Reduce n_estimators for ensemble models
3. Use fewer cross-validation folds
4. Skip some visualizations

### For Better Accuracy
1. Increase n_estimators (Random Forest, XGBoost)
2. Perform hyperparameter tuning (GridSearchCV)
3. Create more engineered features
4. Try ensemble stacking

## Advanced Usage

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

### SMOTE for Class Imbalance

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

## Testing

To verify the pipeline works correctly:

```python
# Run main pipeline
python census_income_analysis.py

# Check outputs
assert os.path.exists('01_target_distribution.png')
assert os.path.exists('08_feature_importance.png')
```

## Contributing

This is a technical assessment project. For improvements:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows PEP 8 style guide
5. Update documentation
6. Submit pull request

## License

This project is created for educational and assessment purposes using public US Census data.

## References

- **Data Source**: US Census Bureau - https://www.census.gov/data.html
- **Dataset Details**: See `census_income_metadata.txt`
- **Original Donors**: Terran Lane and Ronny Kohavi, Silicon Graphics

## Contact

For questions about this analysis:
- Review the `EXECUTIVE_SUMMARY.md` for high-level findings
- Check `PRESENTATION_SLIDES.md` for detailed presentation
- Examine code comments for technical implementation details

## Acknowledgments

- US Census Bureau for providing the dataset
- Dataiku for the technical assessment opportunity
- scikit-learn and XGBoost teams for excellent ML libraries

---

## Quick Commands

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# Run full analysis
python census_income_analysis.py

# View results
ls *.png  # List all generated visualizations
cat EXECUTIVE_SUMMARY.md  # Read findings

# Clean up outputs (optional)
rm *.png  # Remove visualizations to re-generate
```

---

**Last Updated**: October 19, 2025
**Version**: 1.0
**Status**: Production-Ready
