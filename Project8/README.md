# DNA Promoter Prediction using Deep Learning

## Project Overview

This project addresses the computational challenge of identifying promoter regions in DNA sequences using deep learning approaches. Promoters are critical regulatory elements that control gene expression by serving as binding sites for RNA polymerase and transcription factors. Accurate promoter prediction is essential for understanding gene regulation, annotating genomes, and advancing synthetic biology applications.

The problem is particularly challenging because promoter sequences exhibit high variability, with different types of promoters (e.g., TATA-box containing vs. TATA-less) showing distinct sequence patterns. Traditional sequence alignment methods often fail to capture the complex, context-dependent features that distinguish promoters from non-promoter regions.

This work compares four different neural network architectures (LSTM, Bidirectional LSTM, GRU, and CNN-LSTM hybrid) to identify the most effective approach for promoter classification in human DNA sequences.

## Data Description

### Data Source

The dataset consists of two FASTA files:
- **`human_TATA_5000.fa`**: Contains 2,067 human TATA-box promoter sequences (positive class)
- **`human_nonprom_big.fa`**: Contains 27,731 non-promoter sequences (negative class)

These sequences represent real genomic data from human chromosomes, focusing on well-characterized TATA-box promoters as the positive class due to their clear biological signal for transcription initiation.

### Data Structure

- **Sequence Format**: Raw DNA sequences in FASTA format
- **Sequence Length**: Variable (standardized to 200 base pairs during preprocessing)
- **Encoding**: Nucleotides are mapped to integers (A=1, C=2, G=3, T=4, unknown=5, padding=0)
- **Class Distribution**: Balanced dataset with equal numbers of promoter and non-promoter sequences (2,067 samples each, total 4,134 sequences)
- **Train/Test Split**: 80/20 stratified split (3,307 training, 827 test samples)

### Preprocessing

1. **Sequence Standardization**: All sequences are standardized to 200bp using center-cropping for longer sequences or symmetric padding with 'N' for shorter sequences
2. **Integer Encoding**: DNA sequences are converted to integer arrays for neural network input
3. **Class Balancing**: The dataset is balanced to prevent the model from learning a trivial majority-class predictor
4. **Stratified Splitting**: Train/test split maintains proportional class representation

The processed dataset is saved as `dna_dataset_processed.npz` for reproducibility.

## Biological / Genomics Relevance

### Promoter Biology

Promoters are DNA sequences located upstream of genes that serve as recognition sites for RNA polymerase II and transcription factors. They play a crucial role in:

- **Gene Expression Regulation**: Controlling when and where genes are transcribed
- **Cellular Differentiation**: Different cell types express different genes based on promoter accessibility
- **Disease Mechanisms**: Mutations in promoter regions can lead to aberrant gene expression and disease
- **Evolutionary Studies**: Promoter evolution shapes phenotypic diversity

### TATA-Box Promoters

This project specifically focuses on TATA-box containing promoters, which are characterized by the consensus sequence TATAWAAR (where W = A or T, R = A or G). These promoters:

- Represent approximately 10-20% of human promoters
- Are associated with tissue-specific and developmentally regulated genes
- Provide a well-defined biological signal for machine learning models

### Computational Genomics Applications

Accurate promoter prediction has numerous applications:

1. **Genome Annotation**: Identifying gene starts in newly sequenced genomes
2. **Regulatory Element Discovery**: Finding novel promoters in non-coding regions
3. **Synthetic Biology**: Designing promoters with desired expression characteristics
4. **Disease Research**: Identifying promoter mutations associated with genetic disorders
5. **Comparative Genomics**: Understanding promoter evolution across species

## Methods and Pipeline

### Data Preprocessing Pipeline

1. **FASTA Parsing**: Sequences are extracted from FASTA files, ignoring header lines
2. **Length Standardization**: 
   - Longer sequences: Center-cropped to preserve the most informative central region
   - Shorter sequences: Symmetrically padded with 'N' to maintain positional context
3. **Integer Encoding**: Each nucleotide is mapped to an integer token for neural network processing
4. **Dataset Balancing**: Classes are balanced to ensure equal representation
5. **Train/Test Split**: Stratified 80/20 split maintains class proportions

### Feature Engineering

The primary feature engineering approach is learned embeddings:
- **Embedding Layer**: Converts integer-encoded nucleotides into dense vector representations
- **Embedding Dimension**: 64-dimensional vectors capture nucleotide relationships
- **Learned Representations**: The embedding layer learns meaningful nucleotide relationships during training

No hand-crafted features (e.g., k-mer frequencies, GC content) are used, allowing the models to discover relevant patterns directly from the sequence data.

### Training Methodology

- **Optimizer**: Adam optimizer with initial learning rate of 0.001
- **Learning Rate Scheduling**: StepLR scheduler reduces learning rate by 10x every 5 epochs
- **Loss Function**: CrossEntropyLoss for binary classification
- **Early Stopping**: Monitors validation loss with patience of 10 epochs to prevent overfitting
- **Gradient Clipping**: Maximum gradient norm of 1.0 prevents exploding gradients in RNN architectures
- **Batch Size**: 32 samples per batch balances memory usage and gradient stability
- **Maximum Epochs**: 50 epochs (often stopped early due to early stopping)

## Models and Algorithms

### Model 1: Unidirectional LSTM

**Architecture**:
- Embedding layer (vocab_size=6 → 64 dimensions)
- 2-layer LSTM (hidden_dim=128, unidirectional)
- Fully connected layers (128 → 64 → 2 classes)
- Dropout (0.3) for regularization

**Rationale**: Serves as a baseline to compare against bidirectional architectures. Processes sequences left-to-right only, capturing sequential dependencies in one direction.

**Performance**: 74.37% test accuracy

### Model 2: Bidirectional LSTM (BiLSTM)

**Architecture**:
- Same as Model 1, but with bidirectional=True
- Final hidden state concatenates forward and backward representations (256 dimensions)

**Rationale**: Captures context from both sequence directions, which should help identify promoter motifs that depend on surrounding sequence context. The bidirectional approach is particularly useful for DNA sequences where regulatory signals can be context-dependent.

**Performance**: 73.04% test accuracy

### Model 3: GRU (Gated Recurrent Unit)

**Architecture**:
- Embedding layer (vocab_size=6 → 64 dimensions)
- 2-layer bidirectional GRU (hidden_dim=128)
- Fully connected layers (256 → 64 → 2 classes)
- Dropout (0.3) for regularization

**Rationale**: GRU is computationally more efficient than LSTM while often achieving similar performance. The simplified gating mechanism (reset and update gates vs. LSTM's three gates) may help with faster convergence and reduced overfitting.

**Performance**: 95.16% test accuracy (best model)

### Model 4: CNN-LSTM Hybrid

**Architecture**:
- Embedding layer (vocab_size=6 → 64 dimensions)
- 1D Convolutional layer (kernel_size=7, 128 filters) - acts as motif detector
- MaxPooling (kernel_size=2) - reduces sequence length
- 1-layer bidirectional LSTM (hidden_dim=128) - processes detected motifs
- Fully connected layers (256 → 64 → 2 classes)
- Dropout (0.3) for regularization

**Rationale**: Combines the strengths of both architectures:
- **CNN component**: Detects local sequence motifs (short patterns) through convolutional filters
- **LSTM component**: Models longer-range dependencies between detected motifs
- This hybrid approach is well-suited for DNA sequences where both local patterns (motifs) and their spatial arrangement matter

**Performance**: 82.10% test accuracy

### Model Selection

All models were trained with identical hyperparameters and training procedures to ensure fair comparison. The GRU model achieved the best performance (95.16% accuracy) and was saved as `dna_best_model.pth`.

## Results and Findings

### Model Performance Comparison

| Model | Test Accuracy | Precision (Promoter) | Recall (Promoter) | F1-Score (Promoter) |
|-------|---------------|---------------------|-------------------|---------------------|
| LSTM | 74.37% | 0.83 | 0.61 | 0.70 |
| BiLSTM | 73.04% | - | - | - |
| **GRU** | **95.16%** | **0.95** | **0.95** | **0.95** |
| CNN-LSTM | 82.10% | - | - | - |

### Key Findings

1. **GRU Outperforms LSTM**: The GRU model achieved significantly higher accuracy (95.16%) compared to both unidirectional and bidirectional LSTMs. This suggests that:
   - The simplified gating mechanism in GRU may be better suited for this sequence length and dataset size
   - GRU's reduced complexity may help prevent overfitting
   - The bidirectional GRU effectively captures context from both directions

2. **Bidirectional Processing Helps**: Both BiLSTM and bidirectional GRU models benefit from processing sequences in both directions, capturing context-dependent regulatory signals.

3. **Hybrid Architecture Shows Promise**: The CNN-LSTM hybrid achieved 82.10% accuracy, demonstrating that combining local motif detection with sequence-level modeling can be effective, though not optimal for this specific task.

4. **Baseline LSTM Performance**: The unidirectional LSTM serves as a reasonable baseline (74.37%), showing that even simple sequential models can learn meaningful promoter patterns.

### Biological Insights

The high performance of the GRU model (95.16% accuracy) suggests that:
- Promoter sequences contain learnable patterns that distinguish them from non-promoter regions
- The 200bp window captures sufficient information for accurate classification
- Deep learning models can effectively identify TATA-box promoters without explicit motif knowledge

### Practical Implications

The best-performing model (GRU) can be used for:
- **Genome Annotation**: Identifying promoter regions in newly sequenced genomes
- **Regulatory Element Discovery**: Screening genomic regions for potential promoters
- **Functional Genomics**: Understanding gene regulation mechanisms

## Reproducibility

### Environment Setup

**Required Python Packages**:
```
torch >= 1.0.0
numpy >= 1.18.0
scikit-learn >= 0.22.0
```

**Installation**:
```bash
pip install torch numpy scikit-learn
```

### Running the Notebook

1. **Ensure Data Files Are Present**:
   - `human_TATA_5000.fa` (promoter sequences)
   - `human_nonprom_big.fa` (non-promoter sequences)

2. **Execute the Notebook**:
   - Open `work.ipynb` in Jupyter Notebook or JupyterLab
   - Run all cells sequentially
   - The notebook will:
     - Load and preprocess the FASTA files
     - Train all four models
     - Evaluate and compare model performance
     - Save the best model checkpoint

3. **Using Preprocessed Data** (Optional):
   - If `dna_dataset_processed.npz` exists, you can modify the notebook to load it directly instead of reprocessing FASTA files

### Expected Runtime

- **Data Preprocessing**: ~1-2 minutes
- **Model Training** (per model): ~5-15 minutes on GPU, ~30-60 minutes on CPU
- **Total Runtime**: ~1-2 hours on GPU for all four models

### Hardware Requirements

- **Minimum**: CPU with 4GB RAM
- **Recommended**: GPU (CUDA-compatible) with 4GB+ VRAM for faster training
- The code automatically detects and uses GPU if available

### Reproducing Results

To reproduce the exact results:
- Use random seed 42 (set in train_test_split)
- Ensure all models use the same hyperparameters as specified in the notebook
- The processed dataset (`dna_dataset_processed.npz`) ensures consistent train/test splits

### Model Checkpoints

The following model checkpoints are saved:
- `dna_lstm_classifier_lstm.pth`: Unidirectional LSTM model
- `dna_lstm_classifier_bilstm.pth`: Bidirectional LSTM model
- `dna_lstm_classifier_gru.pth`: GRU model (best performer)
- `dna_lstm_classifier_cnnlstm.pth`: CNN-LSTM hybrid model
- `dna_best_model.pth`: Best performing model (GRU)

### Assumptions and Constraints

1. **Sequence Length**: All sequences are standardized to 200bp. Longer or shorter sequences in real applications may require retraining or different preprocessing.

2. **TATA-Box Focus**: The model is trained specifically on TATA-box promoters. Performance on TATA-less promoters may differ.

3. **Human-Specific**: The model is trained on human sequences. Transfer to other species may require retraining or fine-tuning.

4. **Binary Classification**: The model distinguishes promoters from non-promoters. Multi-class classification (e.g., different promoter types) would require architectural modifications.

## Author & Contact Information

**Author**: Babatunde Afeez Olabuntu

**Email**: olabuntubabatunde@gmail.com

**Collaboration**: Open to research and technical collaborations

**Personal Website**: https://olabuntu.github.io/MINE/

---

## License

This project is provided for research and educational purposes. Please cite appropriately if used in academic work.

