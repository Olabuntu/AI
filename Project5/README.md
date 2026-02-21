# DNA to Protein Sequence Translation using Deep Learning

## Project Overview

This project implements and compares multiple deep learning architectures for translating DNA sequences to protein sequences. The task of predicting protein sequences from DNA is fundamental in computational biology, as it mimics the biological process of translation where messenger RNA (mRNA) is decoded to produce proteins.

The project addresses the challenge of learning the mapping from DNA codons (triplets of nucleotides) to amino acids, which is governed by the standard genetic code. While this mapping is deterministic, the goal here is to train neural networks to learn this relationship from data, which has applications in:

- **Protein structure prediction**: Understanding how DNA sequences translate to proteins helps predict protein structure and function
- **Genetic variant analysis**: Predicting how mutations in DNA affect protein sequences
- **Synthetic biology**: Designing DNA sequences that produce desired proteins
- **Bioinformatics pipelines**: Automated protein sequence prediction from genomic data

## Data Description

### Data Source

The dataset used in this project is **synthetically generated** using the standard genetic code. Synthetic data was chosen because:

1. It provides a controlled environment to evaluate model performance
2. It ensures we have ground truth labels (exact protein sequences for each DNA sequence)
3. It allows us to generate large-scale datasets for training deep learning models
4. It eliminates concerns about data quality, annotation errors, or missing labels

### Data Generation Process

The synthetic data generation follows these steps:

1. **Genetic Code Mapping**: Uses the complete standard genetic code with all 64 codons mapped to their corresponding amino acids (20 standard amino acids + 3 stop codons)

2. **Sequence Generation**: 
   - Randomly selects valid codons (excluding stop codons) to build DNA sequences
   - Each sequence contains between 10-100 codons (30-300 nucleotides)
   - Ensures proper codon structure by building sequences codon-by-codon

3. **Translation**: Each DNA sequence is translated to its corresponding protein sequence using the standard genetic code

4. **Dataset Size**: 20,000 DNA-protein sequence pairs were generated for this project

### Dataset Structure

The generated dataset (`data/dna_protein_pairs.csv`) contains the following columns:

- `dna_sequence`: DNA sequence string (A, T, G, C nucleotides)
- `protein_sequence`: Corresponding protein sequence (20 standard amino acids)
- `dna_length`: Length of DNA sequence in nucleotides
- `protein_length`: Length of protein sequence in amino acids

**Key Statistics:**
- DNA length range: 30-300 nucleotides
- Protein length range: 10-100 amino acids
- All sequences follow proper codon boundaries (DNA length is always a multiple of 3)

## Biological / Genomics Relevance

### The Central Dogma of Molecular Biology

This project directly addresses the **translation** step in the central dogma of molecular biology:

```
DNA → RNA → Protein
```

In biological systems:
1. **Transcription**: DNA is transcribed to messenger RNA (mRNA)
2. **Translation**: mRNA is translated to proteins using ribosomes and transfer RNA (tRNA)

This project focuses on the translation step, learning the codon-to-amino-acid mapping that occurs during protein synthesis.

### Biological Significance

1. **Genetic Code Understanding**: The standard genetic code is universal (with minor variations) across most organisms. Learning this mapping computationally helps understand how genetic information is encoded.

2. **Protein Function**: Proteins are the workhorses of cells, performing functions ranging from structural support to enzymatic catalysis. Predicting protein sequences from DNA is crucial for understanding gene function.

3. **Mutation Analysis**: Single nucleotide polymorphisms (SNPs) or mutations can change codons, potentially altering the resulting protein. Models that learn translation can help predict the effects of genetic variants.

4. **Synthetic Biology Applications**: In designing synthetic biological systems, researchers need to predict what proteins will be produced from engineered DNA sequences.

5. **Computational Biology Tools**: This type of model can be integrated into larger bioinformatics pipelines for genome annotation and protein prediction.

## Methods and Pipeline

### Data Preprocessing

1. **Vocabulary Creation**:
   - **DNA vocabulary**: 4 nucleotides (A, T, G, C) + special tokens (`<PAD>`, `<UNK>`) = 6 tokens
   - **Protein vocabulary**: 20 standard amino acids + special tokens (`<PAD>`, `<UNK>`, `<START>`, `<END>`) = 24 tokens

2. **Sequence Encoding**:
   - DNA sequences are encoded as integer sequences using the DNA vocabulary
   - Protein sequences are encoded with `<START>` and `<END>` tokens for sequence generation
   - Sequences are padded or truncated to fixed maximum lengths

3. **Length Determination**:
   - Maximum lengths are set using the 95th percentile of sequence lengths to handle most sequences while avoiding outliers
   - DNA length is rounded to the nearest multiple of 3 to preserve codon boundaries
   - Protein length includes space for START and END tokens

4. **Data Splitting**:
   - Training set: 64% (12,800 samples)
   - Validation set: 16% (3,200 samples)
   - Test set: 20% (4,000 samples)

### Feature Engineering

The main feature engineering step is the proper handling of sequence boundaries:

- **Codon alignment**: Ensuring DNA sequences maintain codon structure
- **Sequence tokens**: Adding START/END tokens to protein sequences for proper sequence generation
- **Padding strategy**: Using padding tokens to handle variable-length sequences in batches

## Models and Algorithms

### Architecture Overview

All models use an **encoder-decoder architecture**, which is well-suited for sequence-to-sequence tasks:

1. **Encoder**: Processes the input DNA sequence and creates a contextual representation
2. **Decoder**: Generates the output protein sequence autoregressively (one amino acid at a time)

### Model Variants

Four different architectures were implemented and compared:

#### 1. RNN (Recurrent Neural Network)
- **Encoder**: Bidirectional RNN
- **Decoder**: Unidirectional RNN
- **Parameters**: ~2.4M
- **Characteristics**: Simple recurrent architecture, baseline model

#### 2. LSTM (Long Short-Term Memory)
- **Encoder**: Bidirectional LSTM
- **Decoder**: Unidirectional LSTM
- **Parameters**: ~2.4M
- **Characteristics**: Better at capturing long-range dependencies than RNN
- **Configuration**: 
  - Embedding dimension: 128
  - Hidden dimension: 256
  - Number of layers: 3

#### 3. GRU (Gated Recurrent Unit)
- **Encoder**: Bidirectional GRU
- **Decoder**: Unidirectional GRU
- **Parameters**: ~2.4M
- **Characteristics**: Similar to LSTM but with fewer parameters

#### 4. Transformer
- **Encoder**: Transformer encoder with self-attention
- **Decoder**: Transformer decoder with self-attention and cross-attention
- **Parameters**: Variable
- **Characteristics**: Attention mechanism allows direct connections between distant positions
- **Configuration**:
  - Embedding dimension: 256
  - Number of attention heads: 8
  - Number of layers: 4
  - Feedforward dimension: 1024

### Training Strategy

1. **Loss Function**: Cross-entropy loss, ignoring padding tokens

2. **Optimization**:
   - Optimizer: Adam with initial learning rate of 0.001
   - Learning rate scheduling: ReduceLROnPlateau (reduces LR by factor of 0.5 when validation loss plateaus)
   - Gradient clipping: Maximum gradient norm of 1.0 to prevent exploding gradients

3. **Training Procedure**:
   - Maximum epochs: 50
   - Early stopping: Patience of 10 epochs based on validation loss
   - Teacher forcing: During training, the decoder uses the actual target sequence; during inference, it uses its own predictions

4. **Regularization**:
   - Early stopping prevents overfitting
   - Gradient clipping stabilizes training

## Results and Findings

### Model Performance Comparison

| Model | Test Loss | Accuracy | Correct/Total Tokens |
|-------|-----------|----------|---------------------|
| RNN | 2.5753 | 20.16% | 45,258/224,480 |
| **LSTM** | **1.3033** | **64.43%** | **144,626/224,480** |
| GRU | 2.0885 | 49.30% | 110,677/224,480 |
| Transformer | 2.9073 | 9.65% | 21,655/224,480 |

### Key Findings

1. **Best Model**: LSTM achieved the best performance with 64.43% token-level accuracy and the lowest test loss (1.3033)

2. **LSTM Advantages**: 
   - LSTM's ability to maintain long-term dependencies through its cell state mechanism makes it well-suited for learning codon-to-amino-acid mappings
   - The bidirectional encoder captures context from both directions of the DNA sequence

3. **RNN Performance**: Basic RNN showed poor performance (20.16%), likely due to vanishing gradient problems with longer sequences

4. **GRU Performance**: GRU performed moderately (49.30%), better than RNN but worse than LSTM, suggesting that the additional complexity of LSTM's cell state is beneficial for this task

5. **Transformer Performance**: Surprisingly, the Transformer model performed worst (9.65%). This could be due to:
   - Insufficient training data for the model's capacity
   - The deterministic nature of the genetic code may not require the complex attention mechanisms that Transformers excel at
   - Potential overfitting or training instability

### Biological Insights

The 64.43% accuracy achieved by the LSTM model indicates that:

- The model successfully learned significant patterns in the codon-to-amino-acid mapping
- There is room for improvement, suggesting that the model may benefit from:
  - Larger training datasets
  - Different architectures or hyperparameters
  - Additional features or context

### Example Predictions

The model demonstrates reasonable translation capabilities. For example:

- **Input DNA**: `ACTATTGATTGCGCAGGAGTCACTATTGAGGGCAACGAACAATGGGCTCTAATCACCGTC...`
- **True Protein**: `TIDCAGVTIEGNEQWALITV...`
- **Predicted Protein**: `TIDCAGVTIDEKGQWEAILV...`

The predictions show that the model correctly identifies many codons and produces biologically plausible protein sequences, though some errors occur, particularly in longer sequences.

## Reproducibility

### Environment Setup

#### Required Dependencies

```bash
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
```

#### Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install torch numpy pandas scikit-learn
```

3. Ensure you have a CUDA-capable GPU for faster training (optional but recommended)

### Running the Notebook

1. Open `work.ipynb` in Jupyter Notebook or JupyterLab

2. Run all cells sequentially. The notebook will:
   - Generate synthetic DNA-protein pairs
   - Preprocess and encode the data
   - Train all four model architectures
   - Compare model performance
   - Save the best model

3. **Expected Runtime**:
   - Data generation: ~1-2 minutes
   - Model training: ~30-60 minutes per model (depending on hardware)
   - Total: ~2-4 hours for all models

### Reproducing Results

To reproduce the exact results:

1. **Random Seeds**: The notebook uses fixed random seeds (42) for reproducibility, but these are not explicitly set in all cells. For exact reproducibility, ensure seeds are set before data generation and model initialization.

2. **Data**: The synthetic data generation is deterministic given the random seed, so running the notebook will produce the same dataset.

3. **Model Checkpoints**: The trained LSTM model is saved in `model/dna_protein_translator_lstm.pth` and can be loaded for inference without retraining.

### Using the Trained Model

To use the saved model for inference on new DNA sequences:

```python
import torch
from work import predict_protein, DNAProteinTranslatorLSTM

# Load checkpoint
checkpoint = torch.load('model/dna_protein_translator_lstm.pth', map_location='cpu')

# Reconstruct model (see notebook for full code)
model = DNAProteinTranslatorLSTM(...)
model.load_state_dict(checkpoint['model_state_dict'])

# Predict protein from DNA
dna_sequence = "ATGCGATCGATCGATCG"
protein = predict_protein(
    dna_sequence, 
    model, 
    checkpoint['dna_vocab'], 
    checkpoint['protein_vocab'],
    checkpoint['max_dna_length'],
    checkpoint['max_protein_length_with_tokens']
)
print(f"Predicted protein: {protein}")
```

### Assumptions and Constraints

1. **Input Format**: DNA sequences must contain only A, T, G, C nucleotides
2. **Sequence Length**: DNA sequences longer than the maximum length (291 nucleotides) will be truncated
3. **Codon Alignment**: The model assumes proper codon boundaries; sequences not aligned to codons may produce suboptimal results
4. **Standard Genetic Code**: The model is trained on the standard genetic code; alternative genetic codes (e.g., mitochondrial) may not work correctly

## Project Structure

```
Project5/
├── README.md                          # This file
├── work.ipynb                         # Main notebook with all code
├── data/
│   └── dna_protein_pairs.csv         # Generated synthetic dataset
└── model/
    └── dna_protein_translator_lstm.pth  # Trained LSTM model checkpoint
```

## Author & Contact Information

**Author**: Babatunde Afeez Olabuntu

**Email**: olabuntubabatunde@gmail.com

**Collaboration**: Open to research and technical collaborations

**Personal Website**: https://olabuntu.github.io/MINE/

## Acknowledgments

This project implements standard encoder-decoder architectures for sequence-to-sequence learning, adapted for the biological domain of DNA-to-protein translation. The work builds upon foundational research in neural machine translation and applies it to computational biology.

## License

This project is provided for educational and research purposes.

