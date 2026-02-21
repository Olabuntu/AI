# Drug Response Prediction in Cancer Cell Lines

**Author:** Babatunde Afeez Olabuntu
**Email:** olabuntubabatunde@gmail.com  
**Collaboration:** Open to research and technical collaborations  
**Personal Website:** https://olabuntu.github.io/MINE/

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Biological / Genomics Relevance](#biological--genomics-relevance)
4. [Methods and Pipeline](#methods-and-pipeline)
5. [Models and Algorithms](#models-and-algorithms)
6. [Results and Findings](#results-and-findings)
7. [Reproducibility](#reproducibility)
8. [Project Structure](#project-structure)

---

## 🎯 Project Overview

This project develops a machine learning pipeline to predict drug response in cancer cell lines using the Genomics of Drug Sensitivity in Cancer (GDSC) dataset. The goal is to predict how sensitive a cancer cell line will be to a specific drug, measured as the Area Under the Curve (AUC) of the dose-response relationship.

**Problem Statement:** Predicting drug response is crucial for personalized cancer treatment. Traditional drug screening is expensive and time-consuming. By leveraging machine learning on large-scale drug sensitivity datasets, we can identify patterns that help predict which drugs are likely to be effective for specific cancer types, potentially accelerating treatment selection and drug discovery.

**Target Variable:** AUC (Area Under the Curve) - a continuous value ranging from 0 to 1, where higher values indicate greater drug sensitivity.

**Problem Type:** Regression (predicting continuous drug response values)

---

## 📊 Data Description

### Data Source

**GDSC (Genomics of Drug Sensitivity in Cancer) Database**

- **Institution:** Wellcome Sanger Institute
- **Dataset:** GDSC1 Fitted Dose Response Data
- **Website:** https://www.cancerrxgene.org/
- **Download:** https://www.cancerrxgene.org/downloads/bulk_download
- **License:** Research and educational purposes only

The GDSC database is a publicly available resource containing drug sensitivity measurements for hundreds of cancer cell lines tested against various anti-cancer compounds. This dataset represents one of the largest and most comprehensive collections of drug response data in cancer research.

### Dataset Structure

The dataset contains drug-cell line combinations with the following key features:

#### Cell Line Information
- **Cell_Line_Name:** Name of the cancer cell line (e.g., A549, MCF7, HeLa)
- **Cosmic_ID:** Unique identifier from the COSMIC (Catalogue of Somatic Mutations in Cancer) database
- **TCGA_Class:** Cancer type classification based on The Cancer Genome Atlas (TCGA) taxonomy (e.g., LUAD for lung adenocarcinoma, BRCA for breast cancer)

#### Drug Information
- **Drug_Name:** Name of the tested drug compound (e.g., Cisplatin, Olaparib, Paclitaxel)
- **Pathway_Name:** Biological pathway targeted by the drug (e.g., DNA replication, PI3K/MTOR signaling)
- **Putative_Target:** Molecular target of the drug (e.g., specific proteins or enzymes)

#### Response Metrics
- **AUC:** Area Under the Curve (0-1 range) - **Primary target variable** representing overall drug sensitivity
- **LN_IC50:** Log-transformed IC50 value (alternative response metric)
- **RMSE:** Root Mean Square Error of the dose-response curve fit
- **Z_Score:** Standardized drug response score

### Dataset Statistics

- **Total Samples:** ~333,000 drug-cell line combinations
- **Cell Lines:** ~1,000 unique cancer cell lines
- **Drugs:** ~400 unique anti-cancer drugs
- **Cancer Types:** 31 TCGA classifications
- **Features:** 721 features after encoding (categorical + numeric)

### Data Quality

- **Missing Values:** 
  - TCGA_Class: 0.19% missing (filled with 'UNKNOWN')
  - Putative_Target: 1.09% missing (filled with 'UNKNOWN')

- **Outliers:** 
  - Removed extreme outliers in target variable (AUC) using 3×IQR method
  - ~3,192 outliers removed (0.96% of data)

---

## 🧬 Biological / Genomics Relevance

### Connection to Cancer Biology

This project addresses a fundamental question in precision oncology: **Can we predict how a cancer cell will respond to a specific drug based on its characteristics?**

**Biological Context:**
- Different cancer types have distinct molecular profiles (mutations, gene expression patterns, pathway activations)
- Drugs target specific molecular pathways and proteins
- The interaction between a drug's mechanism of action and a cell's molecular profile determines drug sensitivity
- Understanding these relationships enables personalized treatment selection

### Genomics and Molecular Biology Applications

1. **Biomarker Discovery:** The model identifies which features (drug properties, cancer types, pathways) are most predictive of drug response, potentially revealing biomarkers for treatment selection.

2. **Drug Repurposing:** By predicting drug response across different cancer types, we can identify existing drugs that might be effective for new indications.

3. **Mechanism Understanding:** Feature importance analysis reveals which biological pathways and targets are most associated with drug sensitivity, contributing to our understanding of drug mechanisms.

4. **Precision Medicine:** The model can assist in selecting the most appropriate drug for a patient's specific cancer type, moving towards personalized treatment strategies.

5. **Drug Discovery:** Pharmaceutical companies can use these predictions to prioritize drug candidates for specific cancer types, potentially reducing the time and cost of drug development.

### Computational Biology Approach

This project demonstrates how computational methods can extract meaningful patterns from large-scale biological datasets. By combining:
- **Categorical features** (drug names, cancer types, pathways) representing discrete biological categories
- **Numeric features** (IC50 values) representing continuous biological measurements
- **Machine learning algorithms** that can capture complex, non-linear relationships

We create a predictive model that learns from thousands of drug-cell line interactions to make predictions about new combinations.

---

## 🔬 Methods and Pipeline

### Data Preprocessing Pipeline

#### 1. Data Loading and Cleaning
- Load GDSC Excel file containing drug response data
- Standardize column names for consistency
- Select biologically relevant features for modeling

#### 2. Target Variable Selection
   - **Primary Target:** AUC (Area Under Curve)
  - Range: 0-1, where higher values indicate greater sensitivity
  - Preferred over IC50 because it captures the entire dose-response relationship
- Remove rows with missing target values
- Convert to numeric format, handling any formatting issues
- Remove extreme outliers using IQR method (3×IQR threshold) to prevent model skewing

#### 3. Feature Engineering

   **Categorical Features Encoding:**
   - **One-Hot Encoding** for low cardinality features:
  - `Drug_Name` (378 categories)
     - `TCGA_Class` (31 categories)
  - `Pathway_Name` (24 categories)
  - `Putative_Target` (289 categories)
  - One-hot encoding preserves the nominal nature of these features without imposing artificial order

   - **Label Encoding** for high cardinality features:
  - `Cell_Line_Name` (958 categories) - encoded as a single feature to avoid creating 958 binary features
  - This is a practical trade-off: we lose the nominal property but keep the feature count manageable
   
   **Numeric Features:**
- `LN_IC50`: Log-transformed IC50 value, included as a feature since it's related to but distinct from AUC

**Missing Value Handling:**
- Categorical features: Filled with 'UNKNOWN' category before encoding
- Critical to handle missing values before encoding, as encoders cannot process NaN values

#### 4. Train-Test Split
   - **Split Ratio:** 80% training, 20% testing
   - **Random State:** 42 (for reproducibility)
- **Critical Design Decision:** Split performed BEFORE any learning-based preprocessing to prevent data leakage
  - All transformers (scaler, feature selector) are fitted ONLY on training data
  - Test data is never used to inform preprocessing decisions

#### 5. Feature Selection
   - **Method:** Mutual Information Regression
- **Rationale:** 
  - Captures both linear and non-linear relationships between features and target
  - More flexible than correlation-based methods (which only capture linear relationships)
  - Works well with tree-based models that can exploit non-linear patterns
   - **Features Selected:** Top 500 features (from 721 total)
- **Reduction:** ~30% feature reduction, reducing noise and computational complexity

#### 6. Feature Scaling
   - **Method:** RobustScaler
- **Rationale:**
  - Uses median and IQR instead of mean and standard deviation
  - More resistant to outliers than StandardScaler
  - Important for real-world biological data that may contain measurement outliers
- **Application:** Fitted on training data only, then applied to test data

### Key Design Decisions

1. **No Data Leakage:**
   - Train-test split before any preprocessing
   - All transformers fitted only on training data
   - Test data never influences training decisions

2. **Proper Encoding:**
   - One-hot encoding for nominal categorical data (preserves nominal property)
   - Label encoding for high cardinality features (practical compromise)
   - Handles unknown categories gracefully in test set

3. **Robust Preprocessing:**
   - RobustScaler for outlier-resistant scaling
   - Mutual information for comprehensive feature selection
   - Outlier removal using IQR method

4. **Feature Selection:**
   - Reduces dimensionality and noise
   - Improves model interpretability
   - Speeds up training without significant performance loss

---

## 🤖 Models and Algorithms

### Model Selection Strategy

Three regression models were trained and compared to identify the best approach:

1. **Linear Regression** (Baseline)
2. **Random Forest Regressor**
3. **XGBoost Regressor**

### Model 1: Linear Regression

**Type:** Linear model

**Strengths:**
- Fast training and prediction
- Highly interpretable (coefficients show feature importance)
- Good baseline for comparison
- No hyperparameters to tune

**Limitations:**
- Assumes linear relationships between features and target
- Cannot capture complex, non-linear patterns
- Lower performance on non-linear data

**Performance:** R² = 0.8197

**Why Included:** Provides a simple, interpretable baseline to compare against more complex models.

### Model 2: Random Forest Regressor

**Type:** Ensemble of decision trees

**Architecture:**
- 100 decision trees
- Maximum depth: 15 (prevents overfitting)
- Minimum samples to split: 5
- Minimum samples in leaf: 2

**Strengths:**
- Handles non-linear relationships naturally
- Provides feature importance scores
- Robust to outliers
- No strict assumptions about data distribution
- Can capture feature interactions

**Limitations:**
- Less interpretable than linear models
- Can overfit with many trees if not regularized
- Slower than linear models

**Performance:** R² = 0.7988

**Why Included:** Demonstrates the value of non-linear models and provides feature importance insights.

### Model 3: XGBoost Regressor ⭐ (Best Model)

**Type:** Gradient boosting ensemble

**Architecture (Initial):**
- 200 boosting rounds
- Maximum depth: 8
- Learning rate: 0.05
- Subsampling: 0.8 (row sampling)
- Column sampling: 0.8

**Architecture (After Hyperparameter Tuning):**
- 300 boosting rounds
- Maximum depth: 10
- Learning rate: 0.1
- Subsampling: 0.9

**Strengths:**
- Excellent performance on structured data
- Handles non-linear relationships effectively
- Built-in regularization prevents overfitting
- Provides feature importance
- Efficient implementation (fast training)
- Gradient boosting learns from previous mistakes, iteratively improving predictions

**Limitations:**
- More complex than linear models
- Requires hyperparameter tuning for optimal performance
- Longer training time than simpler models

**Performance:** 
- Initial: R² = 0.8487
- After Tuning: R² = 0.9027

**Why XGBoost Won:**
- Best at capturing complex drug-cell line interactions
- Handles high-dimensional feature space effectively
- Effective feature interactions through boosting mechanism
- Regularization prevents overfitting while maintaining high performance

### Hyperparameter Tuning

**Method:** Grid Search with 5-fold Cross-Validation

**Process:**
- Searched over parameter grid for best model (XGBoost)
- Optimized for R² score
- Evaluated on test set after tuning

**Results:**
- R² improved from 0.8487 to 0.9027 (+6.36% improvement)
- RMSE reduced from 0.0700 to 0.0561
- MAE reduced from 0.0465 to 0.0321

**Why Tuning Matters:** Hyperparameter tuning significantly improved model performance, demonstrating the importance of proper model configuration.

---

## 📈 Results and Findings

### Model Performance Comparison

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.8197 | 0.0764 | 0.0530 |
| Random Forest | 0.7988 | 0.0807 | 0.0533 |
| XGBoost (Original) | 0.8487 | 0.0700 | 0.0465 |
| **XGBoost (Tuned)** | **0.9027** | **0.0561** | **0.0321** |

### Best Model: XGBoost (Tuned)

**Final Performance Metrics:**
- **R² Score:** 0.9027 (90.27% variance explained)
- **RMSE:** 0.0561 (Root Mean Square Error)
- **MAE:** 0.0321 (Mean Absolute Error)

**Hyperparameters (Tuned):**
- `n_estimators`: 300
- `max_depth`: 10
- `learning_rate`: 0.1
- `subsample`: 0.9

**Improvement from Tuning:**
- R² improved from 0.8487 to 0.9027 (+6.36% improvement)
- RMSE reduced from 0.0700 to 0.0561
- MAE reduced from 0.0465 to 0.0321

### Model Interpretation

**R² Score of 0.9027 means:**
- The model explains 90.27% of the variance in drug response
- Only 9.73% of variance is unexplained
- Strong predictive capability for drug sensitivity

**RMSE of 0.0561 means:**
- On average, predictions are off by 0.0561 AUC units
- Given AUC ranges from 0 to 1, this is a relatively small error
- Indicates good prediction accuracy

**MAE of 0.0321 means:**
- Average absolute prediction error is 0.0321 AUC units
- Further confirms good prediction accuracy

### Key Findings

#### 1. Feature Importance

The most important features for predicting drug response include:
- **LN_IC50:** Most important feature (mutual information score: 0.76)
  - This makes biological sense as IC50 and AUC are related measures of drug sensitivity
- **Cell_Line_Encoded:** Second most important (score: 0.15)
  - Different cell lines have different baseline sensitivities
- **Pathway and Target Features:** Various pathway and target features show importance
  - Suggests that drug mechanism of action is predictive of response
- **TCGA Classification:** Cancer type features are important
  - Different cancer types respond differently to drugs

#### 2. Model Insights

- **Non-linear relationships are crucial:** XGBoost (non-linear) significantly outperformed Linear Regression, indicating that drug response involves complex, non-linear interactions between features.

- **Feature interactions matter:** Ensemble methods (Random Forest, XGBoost) captured complex drug-cell line interactions that linear models cannot represent.

- **Hyperparameter tuning is valuable:** 6.36% improvement in R² after tuning demonstrates the importance of proper model configuration.

#### 3. Data Insights

- Drug response varies significantly across cancer types
- Some drugs show consistent patterns across cell lines
- Cell line characteristics are important predictors
- Pathway and target information contribute meaningfully to predictions

#### 4. Practical Implications

- **Drug Selection:** Model can assist in selecting drugs for specific cancer types
- **Drug Prioritization:** Can identify potentially effective drug-cancer type combinations for further testing
- **Biomarker Identification:** Feature importance reveals which biological features are most predictive
- **Research Direction:** Results suggest areas for further investigation (e.g., specific pathway-target interactions)

### Visualizations Generated

All visualizations are saved in the `outputs/` directory:

1. **predictions_vs_actual.png:** Scatter plot showing model predictions vs true values
2. **error_distribution.png:** Histogram of prediction errors (residuals)
3. **feature_importance.png:** Top 20 most important features from Random Forest
4. **regression_model_comparison.png:** Bar charts comparing all models across metrics
5. **target_distribution.png:** Distribution of AUC values (histogram, box plot, Q-Q plot)
6. **drug_distribution.png:** Frequency of top 20 drugs in dataset
7. **tcga_distribution.png:** Frequency of top 15 TCGA cancer types

---

## 🔄 Reproducibility

### Environment Setup

#### Prerequisites

Python 3.7+ is required. Install the following packages:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl joblib
```

Or install all at once:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl joblib
```

#### Required Packages

- **pandas:** Data manipulation and analysis
- **numpy:** Numerical computing
- **scikit-learn:** Machine learning algorithms and preprocessing
- **xgboost:** XGBoost gradient boosting library
- **matplotlib:** Plotting and visualization
- **seaborn:** Statistical data visualization
- **openpyxl:** Reading Excel files
- **joblib:** Model serialization

### Data Download

#### Option A: Automatic Download (Recommended)

Use the provided download script:

```bash
python download_gdsc_data.py
```

This will download `GDSC1_fitted_dose_response.xlsx` to the project directory.

#### Option B: Manual Download

1. Visit: https://www.cancerrxgene.org/downloads/bulk_download
2. Download the `GDSC1-dataset` file
3. Save as `GDSC1_fitted_dose_response.xlsx` in the project directory

### Running the Notebook

1. **Open the notebook:**
   ```bash
   jupyter notebook work.ipynb
   ```
   Or use JupyterLab:
   ```bash
   jupyter lab work.ipynb
   ```

2. **Run all cells:**
   - Execute cells sequentially from top to bottom
   - The notebook will automatically:
     - Load and preprocess the data
     - Train all models
     - Generate visualizations
     - Save models and outputs

3. **Expected outputs:**
   - Models saved in `models/` directory
   - Visualizations saved in `outputs/` directory
   - Console output showing progress and results

### Reproducing Results

**Key Parameters for Reproducibility:**
- **Random State:** 42 (used throughout for train-test split, sampling, model initialization)
- **Data Sampling:** Set `USE_FULL_DATA = True` in cell 8 to use full dataset (currently set to False for faster experimentation)
- **Hyperparameter Tuning:** Grid search uses 5-fold cross-validation with random_state=42

**Note:** Results may vary slightly if:
- Different random seeds are used
- Different versions of libraries are installed
- Full dataset is used instead of sampled data

### Model Inference

To use the trained model for predictions on new data:

```python
import joblib
import numpy as np

# Load the model and preprocessors
model = joblib.load('models/xgboost_regression.pkl')
scaler = joblib.load('models/regression_scaler.pkl')
selector = joblib.load('models/regression_selector.pkl')
ohe = joblib.load('models/regression_onehot_encoder.pkl')

# Prepare your data (must match training feature format)
# ... prepare features ...

# Preprocess (same pipeline as training)
X_selected = selector.transform(X)
X_scaled = scaler.transform(X_selected)

# Predict
predictions = model.predict(X_scaled)
```

See the `predict_drug_response()` function in the notebook (cell 32) for a complete inference example.

### Assumptions and Constraints

1. **Data Format:** Input data must match the GDSC dataset structure
2. **Feature Availability:** All categorical features used in training must be present (or handled as 'UNKNOWN')
3. **Computational Resources:** 
   - Full dataset training requires significant memory (~8GB+ RAM recommended)
   - Hyperparameter tuning can take 10-30 minutes depending on dataset size
4. **Python Version:** Python 3.7+ required

---

## 📁 Project Structure

```
Project3/
├── README.md                          # This comprehensive documentation
├── work.ipynb                          # Main Jupyter notebook with all code
├── download_gdsc_data.py               # Script to download GDSC data
├── GDSC1_fitted_dose_response.xlsx    # GDSC dataset (downloaded)
│
├── models/                             # Saved models and preprocessors
│   ├── xgboost_regression.pkl          # Best trained model
│   ├── regression_scaler.pkl           # RobustScaler transformer
│   ├── regression_selector.pkl         # Feature selector
│   ├── regression_onehot_encoder.pkl   # OneHotEncoder for categorical features
│   ├── regression_cell_line_encoder.pkl  # LabelEncoder for cell lines
│   └── feature_info.json               # Feature metadata
│
└── outputs/                            # Visualizations and results
    ├── predictions_vs_actual.png       # Predictions vs true values
    ├── error_distribution.png          # Residual analysis
    ├── feature_importance.png          # Feature importance plot
    ├── regression_model_comparison.png # Model comparison
    ├── target_distribution.png         # AUC distribution
    ├── drug_distribution.png           # Drug frequency
    └── tcga_distribution.png           # Cancer type distribution
```

---

## 📚 References

1. **GDSC Database:**
   - Yang, W., et al. (2013). Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. Nucleic Acids Research, 41(D1), D955-D961.
   - Website: https://www.cancerrxgene.org/

2. **XGBoost:**
   - Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

3. **Feature Selection:**
   - Kraskov, A., et al. (2004). Estimating mutual information. Physical Review E, 69(6), 066138.

---

## 👤 Author & Contact

**Babatunde Afeez Olabuntu**

- **Email:** olabuntubabatunde@gmail.com
- **Collaboration:** Open to research and technical collaborations
- **Personal Website:** https://olabuntu.github.io/MINE/

This project demonstrates expertise in:
- Machine Learning and Data Science
- Cancer Genomics and Drug Response Prediction
- Data Preprocessing and Feature Engineering
- Model Selection and Hyperparameter Tuning
- Model Evaluation and Interpretation
- Computational Biology

---

## 📝 License

This project is for educational and research purposes. The GDSC data is provided by the Wellcome Sanger Institute for research use only. Please refer to their terms of use: https://www.cancerrxgene.org/

---

## 🙏 Acknowledgments

- **Wellcome Sanger Institute** for providing the GDSC dataset
- **GDSC Team** for maintaining this valuable resource
- **Open Source Community** for excellent ML libraries (scikit-learn, XGBoost, pandas, etc.)

---

**Last Updated:** 2024
**Project Status:** ✅ Complete - Model trained and evaluated successfully




