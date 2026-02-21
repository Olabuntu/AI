# Crop Yield Prediction Using Machine Learning

## Project Overview

This project develops and compares multiple machine learning models to predict crop yields (in tons per hectare) based on country, crop type, and year. Using historical agricultural data from the Food and Agriculture Organization (FAO) spanning 1961-2016, I trained and evaluated four different regression algorithms to identify the most accurate approach for yield prediction.

**The Problem**: Accurate crop yield prediction is essential for food security planning, agricultural resource management, and economic forecasting. Traditional methods often rely on expert knowledge and historical averages, which may not capture complex patterns in the data.

**Why It Matters**: With global population growth and climate change affecting agricultural productivity, data-driven yield predictions can help governments, organizations, and farmers make informed decisions about crop selection, resource allocation, and food distribution strategies.

## Data Description

### Data Source
- **Source**: Food and Agriculture Organization (FAO) of the United Nations
- **Dataset**: `yield.csv`
- **Time Period**: 1961-2016 (56 years of historical data)
- **Format**: CSV file with country-level crop yield records

### Dataset Structure
The dataset contains the following key variables:
- **Area**: Country name (212 unique countries)
- **Item**: Crop type (10 different crops)
- **Year**: Temporal feature (1961-2016)
- **Value**: Yield in hectograms per hectare (hg/ha)
- **Unit**: Unit of measurement (hg/ha)

### Dataset Statistics
- **Total Records**: 56,708 (after data cleaning)
- **Countries**: 212 countries worldwide
- **Crop Types**: 10 different crops
  - Maize
  - Potatoes
  - Rice, paddy
  - Wheat
  - Sorghum
  - Soybeans
  - Cassava
  - Yams
  - Sweet potatoes
  - Plantains and others

### Data Preprocessing
- **Unit Conversion**: Converted from hg/ha to tons/hectare for easier interpretation
- **Data Cleaning**: Removed invalid yield values (≤0 or >100 tons/ha)
- **Data Quality**: No missing values in key columns
- **Yield Range**: 0.005 to 57.9 tons/hectare (realistic agricultural range)

## Biological / Genomics Relevance

While this project focuses on agricultural yield prediction rather than genomics, it demonstrates important principles relevant to biological data analysis:

1. **Pattern Recognition in Biological Systems**: Just as genomic data contains complex patterns that require sophisticated algorithms to decode, agricultural yield data reflects the interaction of multiple biological and environmental factors (genetics, climate, soil, management practices).

2. **Temporal Analysis**: The 56-year dataset allows us to observe long-term trends, similar to how longitudinal genomic studies track changes over time. The increasing yields over time reflect improvements in agricultural technology, crop genetics, and farming practices.

3. **Categorical Feature Handling**: The one-hot encoding of countries and crops mirrors how genomic data often requires encoding of categorical biological features (e.g., gene variants, tissue types, disease states) for machine learning applications.

4. **Non-linear Relationships**: The superior performance of neural networks suggests that crop yields depend on complex, non-linear interactions between factors—a common characteristic of biological systems where simple linear models often fail.

5. **Predictive Modeling for Biological Outcomes**: This project demonstrates how machine learning can predict biological outcomes (yield) from multiple input features, similar to predicting disease risk from genomic variants or drug response from molecular profiles.

## Methods and Pipeline

### 1. Data Preprocessing

**Data Cleaning**:
- Selected relevant columns (Area, Item, Year, Value)
- Renamed columns for clarity (Area → Country, Item → Crop, Value → Yield_hg_per_ha)
- Converted yield units from hg/ha to tons/hectare (1 hg/ha = 0.0001 tons/ha)
- Filtered out invalid data points (yield ≤ 0 or > 100 tons/ha)

**Rationale**: Unit conversion improves interpretability, and filtering ensures we're working with realistic agricultural yields.

### 2. Feature Engineering

**One-Hot Encoding**:
- Encoded 212 countries into binary features (Country_CountryName: 0 or 1)
- Encoded 10 crop types into binary features (Crop_CropName: 0 or 1)
- Converted all one-hot columns to integers (0/1) for ML compatibility

**Temporal Features**:
- Kept Year as a numerical feature (1961-2016)
- Year captures technological improvements and agricultural practices over time

**Final Feature Set**:
- 223 total features: 212 country features + 10 crop features + 1 year feature
- Target variable: Yield in tons per hectare

**Rationale**: One-hot encoding allows models to learn country-specific and crop-specific patterns. The large number of features (223) reflects the diversity of agricultural systems across 212 countries.

### 3. Data Splitting and Scaling

**Train/Test Split**:
- Training set: 80% (45,366 samples)
- Test set: 20% (11,342 samples)
- Random state: 42 (for reproducibility)
- **Critical**: Split performed BEFORE scaling to prevent data leakage

**Feature Scaling**:
- Method: StandardScaler (Z-score normalization)
- Fitted on: Training data only
- Applied to: Both training and test sets using the same scaler

**Rationale**: Splitting before scaling ensures the test set doesn't influence preprocessing. Scaling is essential because Year (values ~1961-2016) has a much larger scale than binary one-hot features (0/1), which helps algorithms like neural networks converge better.

## Models and Algorithms

I evaluated four different machine learning algorithms to find the best approach:

### 1. Linear Regression (Baseline)
- **Type**: Linear model
- **Purpose**: Baseline for comparison
- **Performance**: 
  - RMSE: 3.91 tons/hectare
  - MAE: 2.66 tons/hectare
  - R²: 0.66

**Why**: Simple linear models provide a baseline to measure improvement from more complex algorithms.

### 2. Random Forest Regressor
- **Type**: Ensemble tree-based method
- **Parameters**:
  - n_estimators: 100
  - max_depth: 15
  - min_samples_split: 5
  - min_samples_leaf: 2
- **Performance**:
  - RMSE: 3.45 tons/hectare
  - MAE: 2.30 tons/hectare
  - R²: 0.74
- **Feature Importance**: Provides insights into which features (crops, countries, year) most affect yield

**Why**: Random forests can capture non-linear relationships and feature interactions while providing interpretable feature importance.

### 3. XGBoost Regressor
- **Type**: Gradient boosting ensemble
- **Parameters**:
  - n_estimators: 200
  - max_depth: 8
  - learning_rate: 0.05
  - subsample: 0.8
  - colsample_bytree: 0.8
- **Performance**:
  - RMSE: 2.97 tons/hectare
  - MAE: 2.00 tons/hectare
  - R²: 0.81

**Why**: XGBoost is a powerful gradient boosting algorithm that often performs well on structured data with many features.

### 4. Neural Network (MLPRegressor)
- **Type**: Multi-layer perceptron (deep learning)
- **Architecture**: 
  - Input layer: 223 features
  - Hidden layers: 128 → 64 → 32 neurons
  - Output layer: 1 (yield prediction)
- **Parameters**:
  - Activation: ReLU
  - Solver: Adam optimizer
  - Regularization: L2 (alpha=0.001)
  - Early stopping: Enabled (prevents overfitting)
- **Performance**:
  - RMSE: 1.30 tons/hectare
  - MAE: 0.67 tons/hectare
  - R²: 0.96

**Why**: Neural networks can capture complex non-linear relationships and interactions between features that simpler models might miss. The decreasing neuron architecture (128→64→32) helps learn hierarchical patterns.

## Results and Findings

### Model Comparison

| Model | RMSE (tons/ha) | MAE (tons/ha) | R² Score |
|-------|----------------|---------------|----------|
| **Neural Network** | **1.30** | **0.67** | **0.96** |
| XGBoost | 2.97 | 2.00 | 0.81 |
| Random Forest | 3.45 | 2.30 | 0.74 |
| Linear Regression | 3.91 | 2.66 | 0.66 |

### Best Model: Neural Network
- **R² Score**: 0.9629 (96.29% variance explained)
- **RMSE**: 1.2977 tons/hectare
- **MAE**: 0.6709 tons/hectare

### Key Insights

1. **Neural Network Superiority**: The neural network significantly outperformed all other models, achieving an R² of 0.96. This suggests that crop yields depend on complex, non-linear interactions between country, crop type, and temporal factors.

2. **Non-linear Relationships**: The poor performance of linear regression (R² = 0.66) compared to tree-based and neural network models indicates that simple linear relationships cannot adequately capture yield patterns.

3. **Temporal Trends**: The dataset shows increasing global crop yields over the 56-year period (1961-2016), reflecting improvements in agricultural technology, crop genetics, and farming practices.

4. **Crop-Specific Patterns**: Different crops show varying average yields, with some crops (like potatoes) having higher yield potential than others.

5. **Practical Accuracy**: With an RMSE of 1.30 tons/hectare and MAE of 0.67 tons/hectare, the neural network model provides predictions accurate enough for practical agricultural planning and resource allocation decisions.

## Reproducibility

### Environment Setup

**Required Dependencies**:
```python
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
```

**Installation**:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

### Running the Notebook

1. **Ensure data file is present**: The notebook expects `yield.csv` in the same directory.

2. **Run the notebook**: Execute `work.ipynb` cell by cell, or run all cells.

3. **Expected outputs**:
   - Model files saved in `models/` directory:
     - `neural_network_crop_yield.pkl` (best model)
     - `crop_yield_scaler.pkl` (preprocessing scaler)
     - `crop_yield_feature_columns.pkl` (feature column order)
   - Visualizations saved in `outputs/` directory:
     - `crop_yield_model_comparison.png`
     - `crop_yield_predictions.png`
     - `crop_yield_by_crop.png`
     - `crop_yield_trends.png`

### Making Predictions

The notebook includes a `predict_crop_yield()` function for making predictions on new data:

```python
from work import predict_crop_yield

# Predict yield for a specific country, crop, and year
prediction = predict_crop_yield(
    country='United States',
    crop='Maize',
    year=2020
)

print(f"Predicted yield: {prediction:.2f} tons/hectare")
```

### Important Notes

1. **Feature Order**: The saved `feature_cols` preserves the exact column order used during training. This is critical for correct predictions—the model expects features in this specific sequence.

2. **Data Leakage Prevention**: 
   - Data is split BEFORE scaling
   - Scaler is fitted only on training data
   - Test data is transformed using the training scaler

3. **One-Hot Encoding**: 
   - During inference, missing feature columns (for countries/crops not in the input) are set to 0
   - All features must be present in the correct order

4. **Model Input**: The saved model expects scaled features. Always use the saved scaler to transform input data before prediction.

### Assumptions and Constraints

- **Data Range**: The model was trained on data from 1961-2016. Predictions for years outside this range may be less reliable.
- **Crop Types**: The model only supports the 10 crop types present in the training data.
- **Countries**: The model supports predictions for any of the 212 countries in the training data.
- **Feature Availability**: The model assumes only country, crop type, and year are available. Additional features (climate, soil, irrigation) could potentially improve predictions but are not included in this dataset.

## Project Structure

```
Project4/
│
├── work.ipynb                    # Main Jupyter notebook with complete analysis
├── yield.csv                     # Dataset (FAO crop yield data)
├── README.md                     # This file
│
├── models/                       # Saved models and preprocessing objects
│   ├── neural_network_crop_yield.pkl      # Best trained model
│   ├── crop_yield_scaler.pkl             # StandardScaler (fitted on training data)
│   └── crop_yield_feature_columns.pkl     # Feature column names in correct order
│
└── outputs/                      # Generated visualizations
    ├── crop_yield_predictions.png         # Predictions vs Actual scatter plot
    ├── crop_yield_by_crop.png            # Average yield by crop type
    ├── crop_yield_trends.png             # Yield trends over time (1961-2016)
    └── crop_yield_model_comparison.png   # Model performance comparison
```

## Author & Contact Information

**Author**: Babatunde Afeez Olabuntu

**Email**: olabuntubabatunde@gmail.com

**Collaboration**: Open to research and technical collaborations

**Personal Website**: https://olabuntu.github.io/MINE/

---

## Acknowledgments

- Food and Agriculture Organization (FAO) for providing the dataset
- Scikit-learn, XGBoost, and other open-source libraries
- The machine learning and computational biology communities for best practices and methodologies

---

**Last Updated**: 2024
