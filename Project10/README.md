# VAE vs GAN Comparison for Synthetic Gene Expression Generation

## Project Overview

This project compares two generative deep learning models—Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN)—for generating synthetic gene expression data. The goal is to determine which approach produces more realistic and useful synthetic genomics data that can be used for downstream analysis, data augmentation, or privacy-preserving research.

Gene expression data is fundamental to understanding biological processes, disease mechanisms, and drug responses. However, real datasets are often limited in size, contain sensitive patient information, or lack diversity. Generative models offer a solution by creating synthetic data that preserves statistical properties and biological patterns while enabling unlimited data generation for research purposes.

## Data Description

### Data Source

The project uses **synthetically generated gene expression data** designed to mimic real RNA-seq (RNA sequencing) characteristics. The data generation process was carefully designed to capture key biological patterns found in actual genomics datasets.

### Data Generation Process

The synthetic data is generated using a custom algorithm that incorporates:

- **Co-expression modules**: Groups of genes that are expressed together, simulating biological pathways or regulatory networks. Genes within a module have correlated expression patterns (correlation strength: 0.6-0.9).

- **Global variation factors**: Systematic variation across samples that simulates batch effects, cell type differences, or other experimental confounders commonly found in real sequencing data.

- **Log-normal distribution**: The data is transformed to follow a log-normal distribution, which matches the statistical properties of real RNA-seq data.

- **Technical noise**: Realistic sequencing variability is added to simulate measurement error.

### Dataset Structure

- **Samples**: 1,000 synthetic samples (representing patients, cells, or experimental conditions)
- **Features**: 500 genes (reduced from typical RNA-seq datasets for computational efficiency while maintaining biological realism)
- **Format**: Gene expression matrix (samples × genes) with continuous, non-negative values
- **Labels**: Binary disease status labels derived from expression patterns

The generated data is saved as `Project10/data/gene_expression_data.csv` with gene identifiers (GENE_00001, GENE_00002, etc.) and sample IDs (SAMPLE_0001, SAMPLE_0002, etc.).

## Biological / Genomics Relevance

### Context in Genomics Research

Gene expression profiling is a cornerstone of modern genomics and precision medicine. RNA-seq technology measures the abundance of RNA transcripts, providing insights into:

- **Disease mechanisms**: Identifying genes and pathways dysregulated in disease
- **Drug discovery**: Understanding drug response and identifying therapeutic targets
- **Biomarker development**: Finding expression signatures that predict outcomes
- **Personalized medicine**: Tailoring treatments based on individual expression profiles

### Why Generative Models Matter

Generative models for genomics data address several critical challenges:

1. **Data scarcity**: Many rare diseases or specific conditions have limited sample sizes. Synthetic data can augment training sets for machine learning models.

2. **Privacy protection**: Real patient data contains sensitive information. Synthetic data can preserve statistical properties without revealing individual identities.

3. **Data sharing**: Synthetic datasets can be shared more freely, enabling collaboration and reproducibility.

4. **Hypothesis testing**: Researchers can generate data under specific biological assumptions to test analytical methods.

5. **Class imbalance**: Synthetic data can balance underrepresented classes in classification tasks.

### Computational Biology Applications

This comparison directly informs computational biologists choosing between VAE and GAN approaches for:
- Creating synthetic cohorts for method validation
- Augmenting training data for predictive models
- Generating in-silico experiments
- Preserving privacy while enabling data sharing

## Methods and Pipeline

### Data Preprocessing

1. **Standardization**: Gene expression values are standardized to zero mean and unit variance using `StandardScaler`. This is crucial for neural network training, as it prevents any single gene from dominating due to scale differences.

2. **Dataset Creation**: The preprocessed data is wrapped in a PyTorch `Dataset` class and loaded in batches of 64 samples with shuffling enabled.

### Pipeline Overview

The analysis follows a systematic pipeline:

1. **Data Generation**: Create synthetic gene expression data with biological realism
2. **Preprocessing**: Standardize features for neural network compatibility
3. **Model Definition**: Implement VAE and GAN architectures
4. **Training**: Train both models on the same dataset
5. **Generation**: Generate synthetic samples from both models
6. **Evaluation**: Compare models using multiple quantitative metrics
7. **Visualization**: Create visualizations to assess quality
8. **Results**: Save models, metrics, and figures

## Models and Algorithms

### Variational Autoencoder (VAE)

**Architecture**:
- **Encoder**: Two fully connected layers (500 → 128 → 128) with ReLU activations
- **Latent Space**: 32-dimensional continuous latent representation
- **Decoder**: Three fully connected layers (32 → 128 → 128 → 500) with ReLU activations
- **Total Parameters**: ~174,000

**Key Features**:
- Learns a probabilistic latent space with mean (μ) and log-variance (log σ²) parameters
- Uses the reparameterization trick to enable differentiable sampling
- Loss function combines reconstruction error (MSE) with KL divergence regularization

**Training**:
- Optimizer: Adam (learning rate: 0.001)
- Epochs: 50
- Loss: Reconstruction Loss + β × KL Divergence (β = 1.0)

**Why VAE**: VAEs provide a structured latent space that can be sampled from, making them well-suited for generating new data. The KL divergence term regularizes the latent space to be close to a standard normal distribution, enabling smooth interpolation and generation.

### Generative Adversarial Network (GAN)

**Generator Architecture**:
- Four fully connected layers (32 → 128 → 128 → 128 → 500) with ReLU activations
- No activation on final layer to match StandardScaled data range
- **Total Parameters**: ~102,000

**Discriminator Architecture**:
- Four fully connected layers with LeakyReLU (0.2) activations
- Dropout (0.3) after first two layers for regularization
- Sigmoid output for binary classification (real vs. fake)
- **Total Parameters**: ~89,000

**Training**:
- Optimizer: Adam (learning rate: 0.0002, betas: (0.5, 0.999))
- Epochs: 100
- Techniques:
  - Label smoothing (0.9 for real, 0.1 for fake) to prevent discriminator overconfidence
  - Gradient clipping (max norm: 1.0) to stabilize training

**Why GAN**: GANs learn to generate data through adversarial training, where a generator and discriminator compete. This can produce highly realistic samples, though training can be less stable than VAEs.

## Results and Findings

### Quantitative Comparison

The models were evaluated using five key metrics:

1. **Mean MSE**: How well each model captures the mean expression levels across genes
   - VAE: 0.000007 (better)
   - GAN: 0.673524

2. **Std MSE**: How well each model captures expression variance
   - VAE: 0.989515
   - GAN: 0.814004 (better)

3. **Wasserstein Distance**: Overall distribution similarity to real data (lower is better)
   - VAE: 0.801164
   - GAN: 0.142642 (better)

4. **Training Stability**: Loss variance in final epochs (lower is better)
   - VAE: 0.000344 (more stable)
   - GAN Generator: 0.021635

5. **Sample Diversity**: Average pairwise distance between generated samples (higher is better)
   - VAE: 0.163666
   - GAN: 3.174470 (more diverse)

### Key Findings

- **VAE Advantages**:
  - Superior at capturing mean expression levels
  - More stable training with consistent loss convergence
  - Provides interpretable latent space for downstream analysis

- **GAN Advantages**:
  - Better at matching overall data distribution (lower Wasserstein distance)
  - Generates more diverse samples
  - Better captures expression variance patterns

- **Trade-offs**:
  - VAE offers stability and interpretability but may generate less diverse samples
  - GAN produces more diverse and distributionally accurate samples but requires more careful training

### Biological Interpretation

The results suggest that:
- For applications requiring **statistical fidelity** (matching mean expression), VAE may be preferable
- For applications requiring **diversity** and **distributional accuracy**, GAN may be better
- The choice depends on the specific downstream application and research goals

### Visualizations

The project includes four key visualizations (saved in `Project10/figures/`):

1. **Training Curves**: Shows loss evolution for both models
2. **PCA Comparison**: 2D projection showing how generated samples compare to real data
3. **Distribution Comparison**: Histograms comparing expression distributions for individual genes
4. **Comparison Summary**: Table summarizing all quantitative metrics

## Reproducibility

### Environment Setup

**Required Python Packages**:
```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
scipy>=1.7.0
tqdm>=4.62.0
```

**Installation**:
```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn scipy tqdm
```

### Running the Analysis

1. **Open the notebook**: `work.ipynb`
2. **Run all cells**: Execute cells sequentially from top to bottom
3. **Expected runtime**: 
   - VAE training: ~2-5 minutes (CPU) or ~30 seconds (GPU)
   - GAN training: ~5-10 minutes (CPU) or ~1-2 minutes (GPU)
   - Total: ~10-20 minutes on CPU, ~2-3 minutes on GPU

### Output Files

After running the notebook, the following structure will be created:

```
Project10/
├── data/
│   └── gene_expression_data.csv          # Generated synthetic data
├── models/
│   └── models.pth                         # Trained model checkpoints
├── figures/
│   ├── training_curves.png                # Training loss plots
│   ├── pca_comparison.png                 # PCA visualization
│   ├── distribution_comparison.png         # Gene expression distributions
│   └── comparison_summary.png             # Metrics summary table
└── results/
    └── comparison_results.csv             # Quantitative metrics
```

### Reproducibility Notes

- **Random seeds**: The data generation uses `random_state=42` for reproducibility
- **Model checkpoints**: Saved models include all necessary configuration for reloading
- **Preprocessing**: The `StandardScaler` is saved with models to ensure consistent preprocessing
- **Hardware**: Results may vary slightly between CPU and GPU due to floating-point precision differences

### Assumptions and Constraints

- The analysis assumes synthetic data generation is acceptable (no real patient data)
- Models are trained on a relatively small dataset (1,000 samples, 500 genes) for computational efficiency
- Results are specific to the data generation process used and may not generalize to all RNA-seq datasets
- Training hyperparameters were chosen for stability and may not be optimal for all use cases

## Author & Contact Information

**Author**: Babatunde Afeez Olabuntu

**Email**: olabuntubabatunde@gmail.com

**Collaboration**: Open to research and technical collaborations

**Personal Website**: https://olabuntu.github.io/MINE/

---

## License

This project is provided for research and educational purposes. Please cite appropriately if used in academic work.

