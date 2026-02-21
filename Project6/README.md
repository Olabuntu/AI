## Patient Subtype Discovery with Breast Cancer Wisconsin Data

Author: **Babatunde Afeez Olabuntu**  

---

### 1. Project overview

This project demonstrates **unsupervised learning (clustering)** to discover **patient subtypes** using the **Breast Cancer Wisconsin** dataset.  
Instead of treating this as a standard supervised classification task, we pretend the labels are unknown and:

- Cluster patients based only on their **cell nucleus measurements**  
- Compare the discovered clusters to the **true diagnosis** (malignant vs benign)  
- Analyze which features best distinguish the groups

The full analysis is implemented in `work.ipynb`.

---

### 2. Data description

- **Source**: `sklearn.datasets.load_breast_cancer()`  
- **Samples**: 569 patients  
- **Features**: 30 continuous features describing cell nuclei (e.g. mean radius, mean texture, mean perimeter, mean area, smoothness, etc.)  
- **Labels (ground truth)**:
  - `0` = **Malignant** (cancerous)
  - `1` = **Benign** (non‑cancerous)

In the notebook:

- The raw data are stored in a DataFrame `df` with:
  - 30 feature columns
  - A `Disease_Status` column containing 0/1 labels  
- The same data are saved to `data/breast_cancer_data.csv` for reproducibility.

---

### 3. Methodology

#### 3.1 Preprocessing

- **Standardization** using `StandardScaler`:
  - Each feature is transformed to have mean ≈ 0 and std ≈ 1.
  - This is critical because clustering algorithms are distance‑based and raw features have very different scales (e.g. area vs smoothness).

#### 3.2 Dimensionality reduction

- **PCA (2D)**:
  - PC1 explains ≈ **44.27%** of variance  
  - PC2 explains ≈ **18.97%**  
  - Total in 2D ≈ **63.24%**  
  - Used for 2D visualization of clusters.

- **UMAP (2D)**:
  - Embedding shape: **(569, 2)**  
  - Preserves local neighborhood structure better than PCA.  
  - Used to visualize cluster separation in a non‑linear way.

#### 3.3 Clustering algorithms

All clustering is performed on the **standardized** feature matrix `X_scaled`.

1. **K‑Means**
   - Search over **k = 2..6**.
   - For each k:
     - Fit K‑Means (`n_init=10`).
     - Record **inertia** (within‑cluster sum of squares).
     - Compute **silhouette score**.
   - Plot:
     - Elbow curve (k vs inertia).
     - Silhouette score vs k.
   - Select **optimal k** via **maximum silhouette score**.

2. **DBSCAN**
   - Density‑based clustering with **automatic eps selection**:
     - Use k‑nearest neighbors (k = `min_samples = 6`) to compute k‑distance.
     - Set eps to the **95th percentile** of k‑distances.
   - Fit DBSCAN and count:
     - Number of clusters (excluding noise label `-1`).
     - Fraction of noise points.

3. **Hierarchical (Agglomerative) clustering**
   - `AgglomerativeClustering` with `n_clusters = optimal_k` from K‑Means (i.e. k=2).
   - Produces an alternative 2‑cluster partition.

---

### 4. Results

#### 4.1 PCA variance

- **PC1**: 44.27%  
- **PC2**: 18.97%  
- **Total (2D)**: 63.24%  

This confirms that 2D visualizations (PCA plots) already capture a large portion of the structure, unlike earlier synthetic 2000‑feature data.

#### 4.2 K‑Means model selection (k = 2..6)

From the notebook output:

- Inertia decreases as k increases (as expected).
- **Silhouette scores:**
  - k=2: **0.3434**
  - k=3: 0.3144
  - k=4: 0.2833
  - k=5: 0.1582
  - k=6: 0.1604

**Conclusion:**

- **Optimal k = 2** by silhouette score.
- This matches the **true number of clinical classes** (malignant vs benign).

#### 4.3 K‑Means clustering performance

Using **k=2** from silhouette:

- Cluster distribution: **[375, 194]** patients.
- Final silhouette score: **0.3434**.

Cluster composition relative to true labels:

- **Cluster 0 (predominantly benign)**:
  - 375 patients total
  - Malignant (0): 36 (9.6%)
  - Benign (1): 339 (90.4%)

- **Cluster 1 (predominantly malignant)**:
  - 194 patients total
  - Malignant (0): 176 (90.7%)
  - Benign (1): 18 (9.3%)

This is strong separation: each cluster is ≈90% pure for one diagnosis type.

#### 4.4 DBSCAN results

- **min_samples**: 6  
- **eps**: ≈ 5.495 (95th percentile of k‑distances)  
- **Clusters found**: **1 cluster + noise**  
  - Noise points: **15** (≈ 2.6%)
  - Cluster labels: `[-1, 0]`

Because DBSCAN effectively finds only one non‑noise cluster, it **does not separate benign and malignant tumors** for these parameters, so it is not competitive here.

#### 4.5 Hierarchical clustering results

- `AgglomerativeClustering(n_clusters=2)` using Euclidean distance on `X_scaled`.
- Cluster distribution: **[184, 385]** patients.

When compared to the true labels (see metrics below), hierarchical clustering performs reasonably well, but **slightly worse than K‑Means**.

#### 4.6 External validation vs true labels

The notebook computes **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**:

- **K‑Means (k=2):**
  - ARI = **0.654**
  - NMI = **0.532**

- **Hierarchical (k=2):**
  - ARI = **0.575**
  - NMI = **0.457**

- **DBSCAN:**
  - Not evaluated (only one cluster + noise; ARI/NMI not meaningful).

**Model comparison summary:**

- **Best overall:** **K‑Means (k=2)**
  - Highest ARI and NMI
  - Clusters align closely with benign vs malignant labels.
- **Second best:** Hierarchical clustering
  - Reasonable alignment, but slightly worse than K‑Means.
- **Worst:** DBSCAN (with current parameters)
  - Fails to recover two clear clusters in this dataset.

---

### 5. Marker feature analysis

For each K‑Means cluster, the notebook computes:

1. Mean value of each feature within the cluster.  
2. Mean value across all **other** clusters.  
3. Absolute difference between these means.  
4. The top 10 features with largest differences (marker features).

Key findings:

- **Cluster 0 (BENIGN)**:
  - Marker features (lower in benign tumors):
    - `worst area`, `mean area`, `area error`
    - `worst perimeter`, `mean perimeter`
    - `worst radius`, `mean radius`
    - `worst texture`, `mean texture`
    - `perimeter error`
  - Interpretation: Benign tumors tend to have **smaller size‑related measurements** and less extreme worst‑case values.

- **Cluster 1 (MALIGNANT)**:
  - Same features as above, but **higher** in this cluster.
  - Interpretation: Malignant tumors are generally **larger** and have more irregular nuclei, reflected in higher area, perimeter, radius and related errors.

These marker features align well with medical intuition: larger, more irregular nuclei are characteristic of malignant tumors.

---

### 6. Visualizations

The notebook produces several key plots:

- **Elbow + Silhouette plot** (`elbow_method.png`):
  - Shows inertia vs k and silhouette vs k.
  - Confirms k=2 as the optimal number of clusters.

- **DBSCAN parameter plots** (`dbscan_parameter_selection.png`):
  - k‑distance graph and histogram of k‑distances.
  - Used to select eps; also shows why DBSCAN struggles here.

- **Clustering comparison plot** (`clustering_comparison.png`):
  - 2×2 grid:
    1. K‑Means in PCA space  
    2. DBSCAN in PCA space  
    3. Hierarchical in PCA space  
    4. K‑Means in UMAP space  
  - Visually confirms that K‑Means + UMAP separates benign vs malignant best.

---

### 7. How to run the notebook

1. Install required Python packages (versions from a typical scientific stack):

```bash
pip install numpy pandas scikit-learn matplotlib seaborn umap-learn
```

2. Open `work.ipynb` in Jupyter or VS Code / Cursor.

3. Run all cells from top to bottom:
   - Data loading
   - Scaling and PCA
   - K‑Means model selection
   - DBSCAN + Hierarchical clustering
   - Visualizations
   - Cluster analysis and marker feature identification
   - External metrics (ARI, NMI)

---

### 8. Conclusions

- The **Breast Cancer Wisconsin dataset** exhibits a clear two‑cluster structure corresponding to **benign vs malignant** tumors.
- **K‑Means with k=2**:
  - Achieves the best alignment with true labels (ARI ≈ 0.65, NMI ≈ 0.53).
  - Produces clusters that are ≈90% pure for each diagnosis class.
- **Hierarchical clustering** is a reasonable alternative but slightly worse.
- **DBSCAN** (with the chosen automatic eps and min_samples) does not recover the two classes well in this setting.
- Marker feature analysis highlights **size‑related and shape‑related measurements** (area, perimeter, radius, texture) as key drivers separating benign and malignant tumors.

Overall, the project demonstrates a complete workflow for:

- Unsupervised discovery of patient subgroups  
- Validation of clusters against real clinical labels  
- Interpretation of clusters via marker features in a medically meaningful way.


