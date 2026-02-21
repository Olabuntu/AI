# Plant Disease Classification using Deep Learning

A deep learning project for classifying plant diseases from leaf images using convolutional neural networks and transfer learning techniques. This project addresses the critical agricultural challenge of early disease detection, which is essential for maintaining crop health and maximizing agricultural productivity.

## Project Overview

### What This Project Does

This project implements an automated plant disease classification system capable of identifying 38 different plant diseases across 14 crop species from leaf images. The system uses deep learning techniques, including custom convolutional neural networks and transfer learning with pre-trained ResNet models, to achieve high accuracy in disease detection.

### The Problem

Early detection of plant diseases is crucial for agricultural productivity and food security. Traditional methods of disease identification require expert knowledge and can be time-consuming, making them impractical for large-scale farming operations. Late detection often leads to significant crop losses and increased pesticide use.

### Why This Problem Matters

- **Food Security**: Plant diseases cause an estimated 10-16% of global crop losses annually
- **Economic Impact**: Early detection can save farmers significant costs by enabling targeted treatment
- **Environmental Benefits**: Precise disease identification reduces unnecessary pesticide application
- **Scalability**: Automated systems can help monitor large agricultural areas efficiently

## Data Description

### Data Source

The project uses the **PlantVillage Dataset**, a publicly available dataset widely used in plant disease classification research.

- **Source**: PlantVillage Dataset
- **Download Link**: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- **Alternative Source**: https://github.com/spMohanty/PlantVillage-Dataset
- **Total Images**: ~54,305 images
- **Image Format**: Color RGB images (JPG/JPEG/PNG)
- **Image Size**: Variable (resized to 224×224 for training)

### Dataset Structure

The dataset contains 38 classes representing different plant species and their disease states:

- **Apple**: Apple_scab, Black_rot, Cedar_apple_rust, healthy
- **Blueberry**: healthy
- **Cherry**: Powdery_mildew, healthy
- **Corn (Maize)**: Cercospora_leaf_spot, Common_rust, Northern_Leaf_Blight, healthy
- **Grape**: Black_rot, Esca_(Black_Measles), Leaf_blight, healthy
- **Orange**: Haunglongbing_(Citrus_greening)
- **Peach**: Bacterial_spot, healthy
- **Pepper (Bell)**: Bacterial_spot, healthy
- **Potato**: Early_blight, Late_blight, healthy
- **Raspberry**: healthy
- **Soybean**: healthy
- **Squash**: Powdery_mildew
- **Strawberry**: Leaf_scorch, healthy
- **Tomato**: Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot, Spider_mites, Target_Spot, Tomato_mosaic_virus, Tomato_Yellow_Leaf_Curl_Virus, healthy

### Data Split

The dataset is split into three sets to ensure proper model evaluation:

- **Training Set**: 70% (~37,998 images) - Used for model training
- **Validation Set**: 15% (~8,145 images) - Used for hyperparameter tuning and early stopping
- **Test Set**: 15% (~8,162 images) - Held-out set for final performance evaluation

The split ensures balanced representation across all classes, with special handling for classes with fewer than 7 images to maintain data integrity.

## Biological / Genomics Relevance

### Connection to Plant Pathology and Agricultural Science

This project directly addresses challenges in **plant pathology** and **agricultural biotechnology**:

1. **Disease Identification**: The model can identify specific pathogens and disease patterns that affect crop health, which is fundamental to plant pathology research.

2. **Precision Agriculture**: Automated disease detection supports precision agriculture initiatives, enabling data-driven farming decisions.

3. **Crop Monitoring**: The system can be integrated into agricultural monitoring systems to provide real-time disease surveillance across large farming operations.

4. **Research Applications**: The methodology can be extended to study disease progression, resistance patterns, and treatment efficacy.

5. **Biological Data Analysis**: This work demonstrates how computational methods can extract meaningful biological information from visual data, similar to how genomics tools extract information from sequence data.

### Real-World Applications

- **Field Monitoring**: Farmers can use mobile applications powered by this model to identify diseases in the field
- **Research**: Plant pathologists can use this as a tool for rapid disease screening
- **Education**: Agricultural extension services can use this to train farmers in disease identification
- **Breeding Programs**: Early disease detection helps identify resistant plant varieties

## Methods and Pipeline

### Data Preprocessing

1. **Data Organization**: Images are organized from the raw PlantVillage dataset into train/validation/test splits (70/15/15)

2. **Image Transformations**:
   - **Training**: Resize to 224×224, random horizontal flips, random rotations (±15°), color jitter (brightness/contrast variation), normalization using ImageNet statistics
   - **Validation/Test**: Resize to 224×224, normalization only (no augmentation for consistent evaluation)

3. **Normalization**: Images are normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) to match pre-trained model expectations

### Feature Engineering

The project leverages transfer learning, which uses features learned from ImageNet (a large general image dataset). These pre-trained features are particularly effective for plant images because:

- ImageNet contains many natural images with similar visual patterns
- The early layers learn general features (edges, textures, shapes) that are transferable
- This approach requires less training data than training from scratch

### Methodological Choices

1. **70/15/15 Split**: Chosen to provide sufficient validation data for early stopping while maintaining a large training set

2. **ImageNet Normalization**: Used because we employ pre-trained models that were trained on ImageNet

3. **Data Augmentation**: Applied only during training to increase dataset diversity and prevent overfitting

4. **Early Stopping**: Implemented to prevent overfitting and save computational resources

## Models and Algorithms

### Model 1: Custom CNN

A convolutional neural network designed from scratch as a baseline comparison.

**Architecture**:
- 3 convolutional layers (32, 64, 128 filters)
- MaxPooling layers (2×2) after each convolution
- 2 fully connected layers (512 hidden units, 38 outputs)
- Dropout (0.5) for regularization
- ReLU activation functions

**Training Configuration**:
- Optimizer: Adam (learning rate: 0.001)
- Loss Function: CrossEntropyLoss
- Learning Rate Scheduler: StepLR (reduces LR by 10× every 5 epochs)
- Maximum Epochs: 20
- Early Stopping Patience: 5 epochs
- Batch Size: 32

**Results**: Best Validation Accuracy: 68.00%

**Why This Model**: I designed this as a baseline to demonstrate the value of transfer learning. Training from scratch requires more data and computational resources.

### Model 2: Transfer Learning (ResNet18)

A pre-trained ResNet18 model fine-tuned for plant disease classification.

**Architecture**:
- Base Model: ResNet18 (pre-trained on ImageNet)
- Frozen Layers: Most layers frozen to preserve pre-trained features
- Trainable Layers: Last 10 parameter groups fine-tuned
- Final Layer: Replaced with Linear(512 → 38) for 38 classes

**Training Configuration**:
- Optimizer: Adam (learning rate: 0.001)
- Loss Function: CrossEntropyLoss
- Learning Rate Scheduler: StepLR (step_size=5, gamma=0.1)
- Maximum Epochs: 15
- Early Stopping Patience: 5 epochs
- Batch Size: 32

**Results**: Best Validation Accuracy: 97.60%

**Why This Model**: Transfer learning leverages features learned from millions of images, making it much more effective than training from scratch. This is especially valuable when working with limited domain-specific data.

### Model Comparison

| Model | Architecture | Validation Accuracy | Training Time | Notes |
|-------|-------------|-------------------|---------------|-------|
| Custom CNN | 3 Conv + 2 FC | 68.00% | Longer | Trained from scratch |
| **ResNet18 (Transfer Learning)** | **Pre-trained ResNet18** | **97.60%** | **Shorter** | **Best Model** |

**🏆 Best Model: Transfer Learning (ResNet18)**

The ResNet18 model significantly outperforms the custom CNN, demonstrating the power of transfer learning for this task. The pre-trained features provide a strong foundation that requires only fine-tuning rather than learning from scratch.

## Results and Findings

### Performance Metrics

- **Best Model**: ResNet18 (Transfer Learning)
- **Validation Accuracy**: 97.60%
- **Test Accuracy**: 97.70% (on held-out test set)

### Key Insights

1. **Transfer Learning Superiority**: The pre-trained ResNet18 model achieved approximately 30% higher accuracy than the custom CNN, highlighting the value of leveraging pre-trained features learned from large-scale image datasets.

2. **Efficient Training**: Transfer learning required fewer epochs (15 vs 20) to achieve better results, making it more computationally efficient.

3. **Excellent Generalization**: The model shows excellent generalization with test accuracy (97.70%) matching validation accuracy (97.60%), indicating minimal overfitting. This suggests the model will perform well on new, unseen images.

4. **Class Performance**: The classification report shows high precision and recall across most disease classes, with some classes achieving perfect 1.00 scores.

### Biological Insights

- The model successfully distinguishes between visually similar diseases (e.g., different types of blight)
- Healthy vs. diseased classification is highly accurate across all plant species
- The system can identify subtle visual patterns that may not be immediately obvious to human observers

### Practical Implications

- **Field Deployment**: The high accuracy suggests the model is ready for real-world deployment
- **Scalability**: The efficient training process allows for easy updates with new disease types
- **Reliability**: Consistent performance across train/val/test sets indicates robust learning

## Reproducibility

### Environment Setup

#### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- CUDA-capable GPU (optional, for faster training - the project works on CPU but will be slower)

#### Installation

1. **Clone or download this repository**

2. **Install required packages**:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision
pip install pillow scikit-learn
pip install pandas numpy
pip install matplotlib seaborn
pip install jupyter ipykernel
pip install kagglehub
```

3. **Download the dataset**:

Option A: Use the provided script (requires Kaggle API setup):
```bash
python scripts/dowload_data.py
```

Option B: Download manually from:
- https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- Extract to `raw_data/plantvillage dataset/color/`

#### Kaggle API Setup (for automated download)

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account → API → Create New API Token
3. Place the `kaggle.json` file in `~/.kaggle/` directory
4. Run: `python scripts/dowload_data.py`

### Running the Notebook

1. **Start Jupyter Notebook**:
```bash
jupyter notebook
```

2. **Open and run `notebooks/work.ipynb`**:
   - The notebook is designed to run sequentially from top to bottom
   - Each cell builds on the previous ones
   - Run all cells to reproduce the complete pipeline

3. **Expected Runtime**:
   - Data organization: ~5-10 minutes (depending on dataset size)
   - Custom CNN training: ~2-4 hours on CPU, ~30-60 minutes on GPU
   - Transfer Learning training: ~1-2 hours on CPU, ~15-30 minutes on GPU

### Reproducing Results

To reproduce the exact results:

1. **Use the same random seed**: The notebook sets `random.seed(42)` and `torch.manual_seed(42)` for reproducibility

2. **Use the same data split**: The first cell organizes data with `random_state=42` in `train_test_split`

3. **Use the same model configuration**: All hyperparameters are specified in the notebook

4. **Note on subset training**: The notebook includes code to create balanced subsets (5000 train, 1000 val, 1000 test) for faster experimentation. To use the full dataset, comment out or skip the subset creation cell.

### Dependencies

All required packages are listed in `requirements.txt`:

- **torch>=1.9.0**: Deep learning framework
- **torchvision>=0.10.0**: Pre-trained models and image transforms
- **Pillow>=8.0.0**: Image processing
- **scikit-learn>=0.24.0**: Data splitting and evaluation metrics
- **pandas>=1.3.0, numpy>=1.21.0**: Data manipulation
- **matplotlib>=3.4.0, seaborn>=0.11.0**: Visualization
- **jupyter>=1.0.0, ipykernel>=6.0.0**: Notebook environment
- **kagglehub>=0.2.0**: Dataset download (optional)

### Assumptions and Constraints

1. **Data Availability**: The project assumes access to the PlantVillage dataset
2. **Computational Resources**: Full training on CPU is feasible but slow; GPU recommended for faster iteration
3. **Image Format**: Assumes images are in JPG, JPEG, or PNG format
4. **Directory Structure**: Assumes data is organized in class-specific subdirectories

### Output Files

After running the notebook, you'll find:

- **`models/plant_disease_model.pth`**: Saved model weights (best performing model)
- **`config/class_to_idx.json`**: Class name to index mapping
- **`config/idx_to_class.json`**: Index to class name mapping
- **`outputs/visualizations/training_history.png`**: Training curves
- **`outputs/visualizations/model_comparison.png`**: Model performance comparison

## Author & Contact Information

**Author**: Babatunde Afeez Olabuntu

**Email**: olabuntubabatunde@gmail.com

**Collaboration**: Open to research and technical collaborations

**Personal Website**: https://olabuntu.github.io/MINE/

## Project Structure

```
Project2/
│
├── notebooks/
│   └── work.ipynb                    # Main Jupyter notebook (complete workflow)
│
├── data/                             # Processed dataset (train/val/test splits)
│   ├── train/                         # Training set (~37,998 images)
│   ├── val/                           # Validation set (~8,145 images)
│   └── test/                          # Test set (~8,162 images)
│
├── raw_data/                         # Original dataset (before processing)
│   └── plantvillage dataset/
│       └── color/                    # Color images (~39,000 images)
│
├── scripts/
│   └── dowload_data.py               # Script to download dataset from Kaggle
│
├── models/
│   └── plant_disease_model.pth      # Saved best model weights (ResNet18)
│
├── config/
│   ├── class_to_idx.json             # Class name → index mapping
│   └── idx_to_class.json             # Index → class name mapping
│
├── outputs/
│   ├── visualizations/               # Training curves and comparisons
│   └── sample_images/                # Sample test images
│
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## Acknowledgments

- PlantVillage team for the comprehensive dataset
- PyTorch team for the excellent deep learning framework
- Kaggle for hosting the dataset
- The open-source community for tools and libraries

## License

This project is for educational and research purposes. The PlantVillage dataset is publicly available for research use.

---

**Last Updated**: 2024
