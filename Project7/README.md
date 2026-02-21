# Medical Text Classification using Transformer Models

## Project Overview

This project implements a medical text classification system using state-of-the-art transformer models. The goal is to automatically categorize medical literature abstracts and clinical texts into predefined classes, which is essential for organizing biomedical information, supporting literature review processes, and enabling efficient retrieval of relevant medical documents.

The problem of medical text classification is critical in computational biology and biomedical informatics because:
- The volume of published medical literature grows exponentially each year
- Manual categorization is time-consuming and inconsistent
- Automated classification enables better information retrieval and knowledge discovery
- It supports downstream applications like literature mining, evidence-based medicine, and clinical decision support systems

## Data Description

The dataset consists of medical text abstracts in tab-separated format, where each entry contains a label and corresponding text. The data is organized into:

- **Training data** (`data/train.dat`): Contains 14,438 labeled examples used for model training
- **Test data** (`data/test.dat`): Contains 14,442 unlabeled examples for final evaluation

### Dataset Structure

Each training example follows the format:
```
<label>\t<text>
```

Where:
- `label` is an integer from 1-5 representing the document category
- `text` is the medical text content (abstracts, clinical notes, or research summaries)

The dataset contains **5 classes** representing different categories of medical literature. The labels are normalized to 0-4 during preprocessing for model compatibility.

### Data Source

The dataset appears to be a collection of medical literature abstracts and clinical texts. The texts cover various medical domains including cardiology, nephrology, oncology, neurology, and other clinical specialties. Each text represents a complete abstract or clinical note that requires classification into one of the five predefined categories.

## Biological / Genomics Relevance

This project addresses a fundamental challenge in computational biology and biomedical informatics: **automated organization and classification of biomedical literature**. The work connects to several important areas:

### Biomedical Literature Mining
- Enables efficient categorization of research papers, clinical notes, and medical abstracts
- Supports systematic reviews and meta-analyses by quickly identifying relevant literature
- Facilitates knowledge discovery by organizing vast amounts of biomedical text

### Clinical Decision Support
- Can be applied to classify clinical notes and patient records
- Supports evidence-based medicine by linking clinical questions to relevant research
- Enables automated triage and routing of medical documents

### Computational Biology Applications
- Supports genomics research by organizing literature related to gene function, pathways, and disease associations
- Enables automated annotation of biological databases
- Facilitates integration of textual knowledge with genomic data

### Research Impact
The ability to automatically classify medical texts is crucial for:
- **Literature curation**: Organizing research papers for databases like PubMed, Gene Ontology, and disease-specific repositories
- **Information retrieval**: Improving search and recommendation systems for biomedical researchers
- **Knowledge integration**: Connecting textual knowledge with structured biological data (genes, proteins, pathways)

## Methods and Pipeline

### Data Preprocessing

1. **Data Loading**: The `.dat` files are parsed, handling both labeled and unlabeled formats
2. **Label Normalization**: Original labels (1-5) are mapped to consecutive integers (0-4) for model compatibility
3. **Train/Validation Split**: Training data is split 80/20 using stratified sampling to maintain class distribution
4. **Data Persistence**: Processed datasets are saved to CSV files for reproducibility

### Feature Engineering

- **Tokenization**: Text is tokenized using model-specific tokenizers (BERT, BioBERT, etc.)
- **Sequence Length**: Maximum sequence length of 256 tokens is used, with truncation and padding as needed
- **No explicit feature engineering**: Transformer models learn representations directly from raw text

### Methodological Choices

- **Stratified splitting**: Ensures balanced class distribution in train/validation sets
- **Early stopping**: Prevents overfitting by monitoring validation accuracy
- **Model checkpointing**: Saves best model based on validation performance
- **Label mapping preservation**: Maintains bidirectional mapping between original and normalized labels for interpretability

## Models and Algorithms

### Transformer Models Evaluated

The project compares four transformer-based models:

1. **BERT-base-uncased**: General-purpose BERT model trained on English Wikipedia and BooksCorpus
   - Baseline for comparison
   - 110M parameters

2. **DistilBERT-base-uncased**: Distilled version of BERT
   - Smaller and faster while maintaining competitive performance
   - 66M parameters

3. **BioBERT-v1.1** (`dmis-lab/biobert-v1.1`): BERT pre-trained on biomedical literature
   - Trained on PubMed abstracts and full-text articles
   - Expected to perform well on medical text due to domain-specific pre-training

4. **Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`): BERT trained on clinical notes
   - Pre-trained on MIMIC-III clinical notes
   - Specialized for clinical text understanding

### Training Configuration

- **Learning rate**: 2e-5 (standard for transformer fine-tuning)
- **Batch size**: 16 (optimized for available hardware)
- **Epochs**: 3 (with early stopping patience of 2)
- **Max sequence length**: 256 tokens
- **Optimization**: AdamW with weight decay (0.01)
- **Evaluation**: Per-epoch validation with accuracy as the primary metric

### Why These Models?

- **General-purpose models** (BERT, DistilBERT) provide baselines and demonstrate transfer learning capabilities
- **Domain-specific models** (BioBERT, ClinicalBERT) leverage pre-training on biomedical/clinical text, which should improve performance on medical classification tasks
- **Comparison approach** allows identification of the most effective model for this specific task

## Results and Findings

The project trains and evaluates all four models, comparing their validation accuracy to identify the best-performing model. Key results include:

### Model Performance Comparison

All models are evaluated on the validation set (20% of training data), and performance metrics include:
- **Accuracy**: Overall classification accuracy
- **Per-class metrics**: Precision, recall, and F1-score for each category
- **Classification report**: Detailed breakdown of model performance across all classes

### Expected Insights

1. **Domain-specific models** (BioBERT, ClinicalBERT) are expected to outperform general-purpose models due to their biomedical pre-training
2. **Model efficiency**: DistilBERT may provide a good balance between accuracy and computational cost
3. **Class-specific performance**: Some medical categories may be easier to distinguish than others

### Practical Implications

The best-performing model can be used for:
- **Automated literature categorization**: Classifying new medical abstracts as they are published
- **Clinical note organization**: Organizing patient records and clinical documentation
- **Research support**: Supporting systematic reviews and meta-analyses
- **Database curation**: Automating annotation of biomedical databases

## Reproducibility

### Environment Setup

1. **Python Dependencies**:
   ```bash
   pip install torch transformers datasets pandas numpy scikit-learn
   ```

2. **Required Libraries**:
   - `torch`: PyTorch for deep learning
   - `transformers`: Hugging Face transformers library
   - `datasets`: Hugging Face datasets
   - `pandas`, `numpy`: Data manipulation
   - `scikit-learn`: Evaluation metrics and data splitting

### Running the Notebook

1. **Data Preparation**: Ensure `data/train.dat` and `data/test.dat` are in the `data/` directory
2. **Execute Notebook**: Run all cells in `work.ipynb` sequentially
3. **Output Locations**:
   - Processed data: `processed_data/` directory
   - Trained models: `model_results/` directory
   - Each model is saved in its own subdirectory

### Reproducing Results

- **Random seed**: Fixed random state (42) for train/validation split ensures reproducibility
- **Model checkpoints**: All trained models are saved and can be reloaded for inference
- **Processed data**: Intermediate processed datasets are saved for consistency

### Hardware Requirements

- **GPU recommended**: Training on GPU is 10-50x faster than CPU
- **Memory**: At least 8GB RAM recommended (batch size may need adjustment for lower memory)
- **Storage**: ~2-3GB for model checkpoints and processed data

### Assumptions and Constraints

- **Internet connection**: Required for first-time model downloads from Hugging Face Hub
- **Data format**: Assumes tab-separated `.dat` files with specific format
- **Label consistency**: All labels in training data must be present in the label mapping

## Author & Contact Information

**Author**: Babatunde Afeez Olabuntu

**Email**: olabuntubabatunde@gmail.com

**Collaboration**: Open to research and technical collaborations

**Personal Website**: https://olabuntu.github.io/MINE/

---

## Project Structure

```
Project7/
├── data/
│   ├── train.dat          # Training data with labels
│   └── test.dat            # Test data (unlabeled)
├── processed_data/
│   ├── train_processed.csv # Processed training set
│   ├── val_processed.csv   # Processed validation set
│   ├── test_processed.csv  # Processed test set
│   └── label_mapping.json  # Label normalization mappings
├── model_results/          # Trained model checkpoints
│   └── [model_name]/       # Individual model directories
├── work.ipynb              # Main analysis notebook
└── README.md               # This file
```

## License

This project is provided for research and educational purposes.

