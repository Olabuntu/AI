# Breast Cancer Bone Relapse Prediction from Gene Expression Data

## Project Overview

This project addresses a critical challenge in breast cancer prognosis: predicting bone metastasis (relapse) in patients based on their gene expression profiles. Bone metastasis is a common and serious complication of breast cancer, occurring in approximately 70% of patients with advanced disease. Early identification of patients at high risk for bone relapse could enable more aggressive monitoring and preventive interventions, potentially improving patient outcomes.

The project uses machine learning to analyze high-dimensional RNA-seq gene expression data from 286 breast cancer patients to build predictive models. By identifying patterns in gene expression that correlate with bone relapse, this work contributes to the growing field of precision oncology, where treatment decisions are informed by molecular profiling.

## Data Description

### Data Source

The dataset is from the **Gene Expression Omnibus (GEO)**, a public repository for high-throughput gene expression data maintained by the National Center for Biotechnology Information (NCBI).

- **Dataset ID:** GSE2034
- **Source URL:** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034
- **Download Method:** Series Matrix File format from GEO database

The data was extracted from the GEO series matrix file using the provided `extract_to_csv.py` script, which processes the raw GEO format and converts it into structured CSV files suitable for analysis.

### Dataset Structure

- **Total Samples:** 286 breast cancer patients
- **Features:** 22,283 genes (gene expression levels measured via microarray)
- **Target Variable:** Bone relapse status
  - `0` = No bone relapse
  - `1` = Bone relapse occurred
- **Class Distribution:**
  - No relapse: 217 samples (75.9%)
  - Relapse: 69 samples (24.1%)

The dataset exhibits class imbalance, which is common in medical datasets where negative outcomes (relapse) are less frequent than positive outcomes (no relapse).

### Data Format

The processed data consists of:
- **Gene Expression Matrix:** Samples as rows, genes as columns (286 × 22,283)
- **Labels:** Sample IDs matched with binary relapse status

## Biological / Genomics Relevance

### Clinical Context

Breast cancer is the most common cancer in women worldwide, and bone metastasis represents a major cause of morbidity and mortality. When breast cancer cells spread to bone, they can cause pain, fractures, and other complications. Understanding which patients are at risk for bone relapse is crucial for:

1. **Risk Stratification:** Identifying high-risk patients who may benefit from more intensive monitoring
2. **Treatment Planning:** Informing decisions about adjuvant therapies and bone-protective treatments
3. **Biomarker Discovery:** Identifying genes and pathways involved in bone metastasis

### Genomics and Molecular Biology

This project leverages **transcriptomics**—the study of gene expression patterns—to understand disease mechanisms. Gene expression profiling using microarrays or RNA-seq provides a snapshot of which genes are active (expressed) in a tissue sample at a given time. In cancer, aberrant gene expression patterns can:

- Reveal dysregulated pathways (e.g., cell cycle, apoptosis, angiogenesis)
- Identify potential therapeutic targets
- Serve as prognostic or predictive biomarkers

The high dimensionality of gene expression data (thousands of genes) presents both opportunities and challenges. While it provides comprehensive molecular information, it also requires sophisticated computational approaches to extract meaningful patterns, especially when sample sizes are limited—a common scenario in clinical research.

### Computational Biology Connection

This work sits at the intersection of computational biology and machine learning. The application of ML to genomics data enables:

- **Pattern Recognition:** Identifying complex, non-linear relationships between gene expression and clinical outcomes
- **Dimensionality Reduction:** Selecting informative genes from thousands of candidates
- **Predictive Modeling:** Building models that can generalize to new patients

The methods used here are directly applicable to other genomics problems, such as predicting drug response, identifying disease subtypes, or discovering novel biomarkers.

## Methods and Pipeline

### Data Preprocessing

1. **Data Loading and Merging**
   - Load gene expression matrix and labels from CSV files
   - Merge by sample ID to ensure proper alignment
   - Remove metadata columns that were accidentally included

2. **Train-Test Split**
   - **Critical step:** Split data before any preprocessing to prevent data leakage
   - 80% training (228 samples) / 20% testing (58 samples)
   - Stratified split to maintain class balance in both sets
   - Random seed (42) for reproducibility

3. **Missing Value Handling**
   - Checked for missing values (none found in this dataset)
   - If present, would use median imputation fitted only on training data

### Feature Engineering and Selection

Given the high dimensionality (22,283 genes) and relatively small sample size (286 patients), feature selection is essential to:
- Reduce computational complexity
- Mitigate overfitting
- Improve model interpretability
- Focus on biologically relevant genes

**Two-Stage Feature Selection Approach:**

**Stage 1: Correlation Filter**
- Quick initial filter using Pearson correlation with target variable
- Threshold: correlation > 0.05
- Removes genes with minimal linear relationship to relapse status
- Result: 22,283 → 13,634 features (38.8% reduction)

**Stage 2: Mutual Information Selection**
- Selects top features using mutual information, which captures non-linear relationships
- Final selection: 1,750 most informative genes
- Total reduction: 22,283 → 1,750 features (92.1% reduction)

This two-stage approach balances computational efficiency with information retention, ensuring that the selected features maintain predictive power while making the problem computationally tractable.

### Feature Scaling

- **StandardScaler:** Normalizes features to zero mean and unit variance
- Essential for algorithms like SVM, logistic regression, and neural networks
- Fitted only on training data, then applied to test data

### Data Leakage Prevention

Throughout preprocessing, strict measures prevent data leakage:
- All preprocessing steps (feature selection, scaling) are fitted only on training data
- Test data is never used to inform training decisions
- Original test data is preserved for inference testing

## Models and Algorithms

Four machine learning models were trained and compared:

### 1. Logistic Regression
- **Type:** Linear classifier
- **Rationale:** Simple, interpretable baseline model
- **Hyperparameters:** C=1.0, max_iter=1000
- **Performance:** 74.14% accuracy

### 2. Support Vector Machine (SVM)
- **Type:** Non-linear classifier with RBF kernel
- **Rationale:** Effective for high-dimensional data, finds optimal decision boundaries
- **Hyperparameters:** C=1.0, gamma='scale', kernel='rbf'
- **Performance:** 75.86% accuracy

### 3. XGBoost
- **Type:** Gradient boosting ensemble
- **Rationale:** Often achieves best performance on structured data, handles non-linear relationships well
- **Hyperparameters:** n_estimators=100, max_depth=6, learning_rate=0.1
- **Performance:** 79.31% accuracy (best model)

### 4. Neural Network
- **Type:** Multi-layer perceptron
- **Architecture:** 2 hidden layers (128, 64 neurons)
- **Rationale:** Can learn complex non-linear patterns in gene expression
- **Hyperparameters:** hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=0.01
- **Performance:** 75.86% accuracy

**Note:** Random Forest was considered but skipped due to computational constraints with this high-dimensional dataset.

### Hyperparameter Tuning

The best-performing model (XGBoost) underwent hyperparameter optimization using:
- **GridSearchCV** with 5-fold cross-validation
- **Scoring metric:** AUC-ROC (appropriate for imbalanced data)
- **Parameter grid:** n_estimators [100, 200, 300], max_depth [3, 6, 9], learning_rate [0.01, 0.1, 0.2], subsample [0.8, 0.9, 1.0]

The tuned model did not outperform the original, so the original XGBoost model was retained as the final model.

## Results and Findings

### Model Performance Comparison

| Model | Accuracy | Notes |
|-------|----------|-------|
| **XGBoost** | **79.31%** | Best overall performance |
| SVM | 75.86% | Good performance, but poor recall on minority class |
| Neural Network | 75.86% | Balanced performance across classes |
| Logistic Regression | 74.14% | Baseline model, interpretable |

### Key Findings

1. **XGBoost achieved the best accuracy** (79.31%) among all models tested, demonstrating the effectiveness of gradient boosting for this genomics classification task.

2. **Class imbalance impact:** All models showed better performance on the majority class (no relapse) than the minority class (relapse). This is a common challenge in medical datasets and suggests that future work could benefit from techniques like SMOTE, class weighting, or cost-sensitive learning.

3. **Feature selection effectiveness:** The two-stage feature selection successfully reduced dimensionality by 92.1% (from 22,283 to 1,750 genes) while maintaining predictive power, demonstrating that most genes are not directly relevant to bone relapse prediction.

4. **Model interpretability vs. performance trade-off:** While XGBoost provides the best accuracy, logistic regression offers better interpretability, which can be valuable for understanding which genes are most predictive.

### Biological Insights

The selected 1,750 genes likely include:
- Genes involved in bone remodeling pathways
- Metastasis-related genes
- Cell adhesion and migration genes
- Angiogenesis factors

Future analysis could involve pathway enrichment analysis to identify biological processes associated with bone relapse.

### Limitations and Future Directions

- **Class imbalance:** The 75.9% vs. 24.1% imbalance affects minority class performance
- **Sample size:** With 286 samples, there's a risk of overfitting, especially with high-dimensional data
- **External validation:** Results should be validated on independent datasets
- **Feature interpretation:** Further analysis of selected genes could reveal biological mechanisms

## Reproducibility

### Environment Setup

**Requirements:**
- Python 3.8 or higher
- Jupyter Notebook

**Install Dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### Running the Analysis

1. **Prepare the Data:**
   ```bash
   python extract_to_csv.py
   ```
   This script extracts data from `Data Files/GSE2034_series_matrix.txt` and creates:
   - `Data Files/gene_expression.csv`
   - `Data Files/labels.csv`

2. **Run the Notebook:**
   ```bash
   jupyter notebook work.ipynb
   ```
   Execute all cells sequentially. The notebook will:
   - Load and explore the data
   - Perform preprocessing and feature selection
   - Train and compare multiple models
   - Perform hyperparameter tuning
   - Save models and create inference function

3. **Expected Outputs:**
   - `Models/` directory with saved models and preprocessors
   - `Outputs/` directory with visualizations (class distribution, model comparison, confusion matrix)

### Dependencies and Assumptions

**Key Dependencies:**
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- joblib >= 1.0.0

**Assumptions:**
- The GEO series matrix file (`GSE2034_series_matrix.txt`) is present in `Data Files/`
- Input data has exactly 22,283 features in the same order as training data
- Random seed (42) ensures reproducibility of train-test split

### Using the Trained Model for Inference

After training, you can use the saved model to predict on new samples:

```python
import joblib
import numpy as np

# Load the inference function from the notebook or define it
def predict_relapse(gene_expression_data, model_path, scaler_path, 
                    corr_indices_path, selector_mi_path):
    # ... (see work.ipynb for full implementation)
    pass

# Example usage
model_path = 'Models/xgboost_model.pkl'
scaler_path = 'Models/scaler.pkl'
corr_indices_path = 'Models/corr_indices.pkl'
selector_mi_path = 'Models/selector_mi.pkl'

# New sample must have shape (1, 22283) with features in same order
new_sample = np.array([...])  # Your gene expression data

prediction, probability = predict_relapse(
    new_sample, 
    model_path, 
    scaler_path, 
    corr_indices_path, 
    selector_mi_path
)

print(f"Prediction: {prediction} (0=no relapse, 1=relapse)")
print(f"Probabilities: [No relapse: {probability[0]:.4f}, Relapse: {probability[1]:.4f}]")
```

**Important:** Input data must have exactly 22,283 features in the same order as the training data.

## Project Structure

```
Project1/
│
├── README.md                    # This file
├── work.ipynb                   # Main analysis notebook
├── extract_to_csv.py           # Script to extract data from GEO file
│
├── Data Files/
│   ├── GSE2034_series_matrix.txt  # Original GEO data file
│   ├── gene_expression.csv         # Extracted gene expression matrix
│   └── labels.csv                  # Extracted labels (relapse status)
│
├── Models/                       # Created after training
│   ├── xgboost_model.pkl       # Trained XGBoost model
│   ├── scaler.pkl              # Feature scaler
│   ├── corr_indices.pkl        # Correlation filter indices
│   └── selector_mi.pkl         # Mutual information selector
│
└── Outputs/                      # Generated during analysis
    ├── class_distribution.png  # Class distribution visualization
    ├── model_comparison.png    # Model performance comparison
    └── confusion_matrix.png    # Confusion matrix for best model
```

## Author & Contact Information

**Author:** Babatunde Afeez Olabuntu

**Email:** olabuntubabatunde@gmail.com

**Collaboration:** Open to research and technical collaborations

**Personal Website:** https://olabuntu.github.io/MINE/

---

*This project is for educational and research purposes. Last updated: 2025*
