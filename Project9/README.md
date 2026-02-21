# Variational Autoencoder for Gene Expression Data Generation

A deep learning project that uses Variational Autoencoders (VAEs) to learn the underlying distribution of gene expression data and generate realistic synthetic samples. This work addresses the challenge of data scarcity in genomics research by providing a method to augment datasets while preserving biological relationships.

## Project Overview

### What This Project Does

This project implements a Variational Autoencoder to:
- Learn the complex distribution of gene expression patterns from RNA-seq-like data
- Generate synthetic gene expression profiles that preserve key biological characteristics
- Compress high-dimensional gene expression data into a meaningful latent representation
- Enable downstream applications such as data augmentation, privacy-preserving data sharing, and hypothesis generation

### The Problem It Addresses

Genomics research, particularly in transcriptomics, faces several critical challenges:

1. **Data Scarcity**: High-quality RNA-seq datasets are expensive to generate and often have limited sample sizes, especially for rare diseases or specific conditions.

2. **Privacy Concerns**: Sharing real patient gene expression data raises privacy and ethical concerns, limiting collaborative research.

3. **Class Imbalance**: Many biological studies suffer from imbalanced datasets where certain conditions or cell types are underrepresented.

4. **Reproducibility**: Generating synthetic data that maintains biological relationships (gene co-expression, pathway activities) is crucial for valid downstream analysis.

### Why This Problem Matters

Synthetic gene expression data generation has significant implications for:
- **Biomedical Research**: Augmenting training datasets for machine learning models in drug discovery and disease classification
- **Method Development**: Testing and validating new computational methods without privacy concerns
- **Educational Purposes**: Teaching genomics and machine learning concepts with realistic but synthetic data
- **Hypothesis Generation**: Exploring the latent space to understand relationships between gene expression patterns and phenotypes

## Data Description

### Data Source

This project uses **synthetically generated gene expression data** designed to mimic real RNA-seq characteristics. The data generation process was carefully crafted to capture essential properties of actual transcriptomic datasets.

### Why Synthetic Data?

I chose to generate synthetic data for several reasons:
1. **Reproducibility**: Synthetic data allows complete control over the data generation process, ensuring reproducible results
2. **Educational Value**: Demonstrates VAE capabilities without requiring access to restricted real datasets
3. **Privacy**: Avoids any ethical concerns associated with real patient data
4. **Controlled Complexity**: Allows systematic exploration of how VAE architecture choices affect performance on data with known structure

### Data Structure

The generated dataset contains:
- **1,000 samples** (simulating patients, cells, or experimental conditions)
- **500 genes** (features representing gene expression levels)
- **Binary labels**: Disease status (0 or 1) based on expression patterns

### Key Features

The synthetic data generation process incorporates:

1. **Gene Co-expression Modules**: Groups of genes that are correlated, simulating biological pathways or co-regulated gene sets. This mimics the modular structure found in real transcriptomic data where genes in the same pathway tend to be expressed together.

2. **Log-Normal Distribution**: Gene expression values follow a log-normal distribution, which accurately reflects the distribution of RNA-seq counts after normalization.

3. **Global Factors**: Batch effects and cell-type-specific variations that affect all genes, simulating technical and biological variability in real experiments.

4. **Technical Noise**: Realistic sequencing noise to simulate measurement error.

5. **Non-Negative Values**: All expression values are non-negative, as gene expression cannot be negative.

The data is saved as `gene_expression_data.csv` with:
- Rows: Samples (SAMPLE_0001, SAMPLE_0002, ...)
- Columns: Gene identifiers (GENE_00001, GENE_00002, ...) plus a `Disease_Status` column

## Biological / Genomics Relevance

### Connection to Genomics and Molecular Biology

This project directly addresses fundamental questions in computational biology:

#### 1. **Transcriptomics and Gene Expression Analysis**

RNA-seq (RNA sequencing) is a cornerstone technology in modern genomics that measures the abundance of RNA transcripts. Gene expression data:
- Reveals which genes are active in different cell types, tissues, or conditions
- Helps identify disease biomarkers and therapeutic targets
- Enables understanding of cellular responses to stimuli, drugs, or environmental changes

#### 2. **Biological Pathways and Co-expression**

In real biological systems, genes don't act in isolation. They form:
- **Co-expression modules**: Groups of genes that are expressed together, often because they participate in the same biological pathway
- **Regulatory networks**: Genes controlled by the same transcription factors show correlated expression patterns
- **Functional relationships**: Genes with similar functions often have coordinated expression

This project explicitly models these relationships through gene modules, making the synthetic data biologically meaningful.

#### 3. **Latent Space Representation**

The VAE learns a compressed 32-dimensional representation of 500-dimensional gene expression profiles. This latent space:
- Captures the essential biological variation in the data
- Can reveal hidden biological structure (e.g., cell type differences, disease states)
- Enables interpolation between expression states, which could model biological transitions

#### 4. **Applications in Biomedical Research**

The generated synthetic data and learned representations can be used for:
- **Data Augmentation**: Increasing training set size for disease classification models
- **Privacy-Preserving Research**: Sharing synthetic data instead of real patient data
- **Hypothesis Generation**: Exploring the latent space to identify novel gene expression patterns
- **Method Validation**: Testing new analysis pipelines on realistic but controlled data

### Computational Biology Context

This work sits at the intersection of:
- **Deep Learning**: Using neural networks to model complex biological distributions
- **Dimensionality Reduction**: Compressing high-dimensional genomic data while preserving biological information
- **Generative Modeling**: Creating new data samples that follow learned biological patterns
- **Statistical Genomics**: Validating synthetic data quality using established statistical tests

## Methods and Pipeline

### Data Preprocessing

1. **Standardization**: Gene expression values are standardized using `StandardScaler` to have zero mean and unit variance. This ensures all genes contribute equally to the model, regardless of their baseline expression levels.

2. **Rationale**: For generative models like VAEs, using the full dataset for scaling is appropriate because we're learning the data distribution rather than making predictions on unseen test data.

### Feature Engineering

The synthetic data generation itself can be considered a form of feature engineering:
- **Module-based structure**: Creates correlated gene groups that the VAE must learn
- **Global factors**: Introduces sample-level variation that affects all genes
- **Noise modeling**: Adds realistic technical variation

### Statistical and Computational Methods

#### 1. **Variational Autoencoder Architecture**

The VAE uses an encoder-decoder structure:
- **Encoder**: Compresses 500-dimensional gene expression → 32-dimensional latent space
- **Latent Space**: Represents each sample as a probability distribution (mean and variance)
- **Decoder**: Reconstructs gene expression from the latent representation

#### 2. **Reparameterization Trick**

Enables gradient-based optimization through stochastic sampling by expressing the latent variable as a deterministic transformation of a standard normal distribution.

#### 3. **Loss Function**

Combines two objectives:
- **Reconstruction Loss**: Measures how well the model reconstructs input data (MSE)
- **KL Divergence**: Regularizes the latent distribution to be close to standard normal, ensuring smooth latent space

#### 4. **Training Strategy**

- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Learning Rate Scheduling**: Reduces learning rate when loss plateaus
- **Gradient Clipping**: Prevents exploding gradients for training stability

### Methodological Choices

**Why VAE over other generative models?**
- VAEs provide explicit latent representations, useful for biological interpretation
- The probabilistic latent space enables uncertainty quantification
- Training is more stable than GANs for this data type

**Why 32-dimensional latent space?**
- Balances compression (reducing 500 → 32) with reconstruction quality
- Allows meaningful biological structure to be captured without overfitting

**Why batch normalization and dropout?**
- Batch normalization stabilizes training with high-dimensional genomic data
- Dropout prevents overfitting, important when learning from limited samples

## Models and Algorithms

### Model Architecture

**Variational Autoencoder (VAE)**

```
Input: 500 genes (gene expression values)
  ↓
Encoder: 500 → 256 → 128 → 64
  ↓
Latent Space: 32 dimensions (mean + log variance)
  ↓
Decoder: 64 → 128 → 256 → 500
  ↓
Output: 500 genes (reconstructed expression)
```

**Key Components:**
- **Encoder Layers**: 3 fully connected layers with batch normalization, LeakyReLU activation, and dropout (0.2)
- **Latent Space**: 32-dimensional continuous space with learned mean and variance
- **Decoder Layers**: Symmetric to encoder, reconstructing gene expression from latent codes
- **Total Parameters**: 347,316 trainable parameters

### Why This Architecture?

1. **Deep but Not Too Deep**: Three hidden layers provide sufficient capacity without excessive complexity
2. **Batch Normalization**: Essential for stable training with high-dimensional genomic data
3. **LeakyReLU**: Prevents dead neurons that can occur with standard ReLU in deep networks
4. **Dropout**: Regularization technique that helps prevent overfitting to the training data
5. **Symmetric Encoder-Decoder**: Common design choice that ensures balanced compression and reconstruction

### Training Algorithm

**Optimizer**: Adam (Adaptive Moment Estimation)
- Learning rate: 0.001
- Adaptive learning rate scheduling based on loss plateau

**Loss Function**: 
```
L_total = L_reconstruction + β × L_KL

where:
- L_reconstruction = MSE(reconstructed, original)
- L_KL = KL divergence between learned and standard normal distribution
- β = 1.0 (standard VAE, can be tuned for beta-VAE)
```

**Training Procedure**:
1. Forward pass: Encode → Sample latent → Decode
2. Compute reconstruction and KL losses
3. Backpropagate gradients
4. Clip gradients (max norm = 1.0)
5. Update parameters
6. Monitor loss and save best model

## Results and Findings

### Model Performance

**Training Metrics:**
- **Initial Loss**: 26,736 (epoch 10)
- **Final Loss**: 23,794 (epoch 50)
- **Loss Reduction**: ~11% improvement over training
- **Reconstruction Loss**: 22,387 (final)
- **KL Divergence**: 1,406 (final)
- **Training Stability**: Smooth convergence without oscillations

### Synthetic Data Quality

**Statistical Validation:**

1. **Correlation Structure Preservation**:
   - Gene-gene correlation matrix similarity: **0.77** (p < 0.001)
   - This indicates the model successfully learned and preserved relationships between genes

2. **Distribution Matching**:
   - Mean correlation (real vs synthetic): **0.34** (p = 7.33e-15)
   - Standard deviation correlation: **0.83** (p = 3.97e-130)
   - Average Kolmogorov-Smirnov statistic: **0.30** (acceptable range)

3. **Variance Analysis**:
   - Variance ratio (synthetic/real): **0.39**
   - This variance collapse is a known limitation of standard VAEs, where the model tends to generate samples with less variability than the training data

### Biological Insights

1. **Latent Space Structure**: The learned 32-dimensional representation successfully captures biological variation, as evidenced by:
   - Clear separation of disease status groups in t-SNE visualization
   - Smooth interpolation between expression states

2. **Gene Co-expression Preservation**: The model maintains correlation patterns between genes, suggesting it learned meaningful biological relationships rather than just memorizing individual gene values.

3. **Generative Capability**: The VAE can produce novel gene expression profiles that:
   - Follow the learned distribution
   - Maintain realistic gene-gene relationships
   - Exhibit appropriate expression levels

### Practical Implications

**Strengths:**
- ✅ Excellent preservation of gene correlation structure (0.77 similarity)
- ✅ Good relative variance patterns (0.83 correlation)
- ✅ Stable training with consistent improvement
- ✅ Biologically meaningful latent representations

**Limitations:**
- ⚠️ Variance collapse: Synthetic data has ~39% of real variance (common VAE limitation)
- ⚠️ Moderate mean alignment (0.34 correlation) - room for improvement
- ⚠️ Distribution matching is acceptable but not perfect (KS = 0.30)

**Scientific Interpretation:**
The results demonstrate that VAEs can learn meaningful biological structure from gene expression data, particularly gene-gene relationships. The variance collapse suggests that while the model captures the overall distribution shape and correlations, it tends to generate more conservative (less variable) samples. This is a well-documented limitation of VAEs and indicates potential for future improvements using alternative architectures or training strategies.

## Reproducibility

### Environment Setup

**Required Python Packages:**
```bash
pip install torch numpy pandas matplotlib scikit-learn scipy jupyter
```

**Python Version**: Python 3.7 or higher recommended

**Hardware Requirements**:
- **CPU**: Training takes ~5-10 minutes on modern CPUs
- **GPU**: Optional but recommended; reduces training time to ~1-2 minutes (CUDA-compatible GPU required)

### Running the Notebook

1. **Open the Notebook**:
   ```bash
   jupyter notebook work.ipynb
   ```

2. **Execute Cells Sequentially**:
   - Run all cells from top to bottom
   - The notebook will:
     - Generate synthetic gene expression data
     - Train the VAE model
     - Generate visualizations
     - Perform statistical evaluations
     - Save all outputs

3. **Expected Outputs**:
   - `gene_expression_data.csv`: Generated dataset
   - `best_vae_model.pth`: Best model checkpoint
   - `gene_expression_vae.pth`: Final trained model
   - `training_curves.png`: Training loss visualization
   - `gene_distribution_comparison.png`: Distribution comparisons
   - `correlation_comparison.png`: Correlation matrix heatmaps
   - `latent_space_visualization.png`: Latent space projections
   - `latent_interpolation.png`: Interpolation examples

### Reproducibility Guarantees

**Random Seeds**: 
- Data generation uses `random_state=42`
- t-SNE uses `random_state=42`
- NumPy random seed is set in the data generation function

**Deterministic Behavior**:
- With the same random seeds, the notebook produces identical results
- Model training uses PyTorch's default random initialization (may vary slightly between runs)

### Dependencies and Assumptions

**Key Assumptions**:
1. Data is standardized before training (handled automatically)
2. Batch size of 64 is appropriate for the dataset size
3. 32-dimensional latent space is sufficient for 500 genes
4. Beta = 1.0 provides good balance between reconstruction and regularization

**Potential Issues**:
- **CUDA/GPU**: If CUDA is not available, the code automatically falls back to CPU (slower but functional)
- **Memory**: The full dataset (1000 samples × 500 genes) requires minimal memory (~2MB)
- **Visualization**: Some plots may appear differently depending on matplotlib backend

### Reproducing Results

To exactly reproduce the results shown:
1. Use Python 3.7+ with the exact package versions (if needed, use a virtual environment)
2. Run all cells in order without modification
3. Ensure sufficient disk space for saved models (~1-2 MB) and images (~500 KB each)

## Author & Contact Information

**Author**: Babatunde Afeez Olabuntu

**Email**: olabuntubabatunde@gmail.com

**Collaboration**: Open to research and technical collaborations

**Personal Website**: https://olabuntu.github.io/MINE/

---

## Project Files

```
Project9/
├── work.ipynb                          # Main analysis notebook
├── gene_expression_data.csv            # Generated synthetic dataset
├── gene_expression_vae.pth             # Trained model weights
├── best_vae_model.pth                  # Best model checkpoint
├── training_curves.png                 # Training loss visualization
├── gene_distribution_comparison.png    # Distribution comparisons
├── correlation_comparison.png          # Gene correlation matrices
├── latent_space_visualization.png      # Latent space analysis
├── latent_interpolation.png            # Interpolation visualization
└── README.md                           # This file
```

## Citation

If you use this code or methodology, please cite:

- **Variational Autoencoder**: Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*.

## License

This project is for educational and research purposes.

---

*Last updated: 2024*
