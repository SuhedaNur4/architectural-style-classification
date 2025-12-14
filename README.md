# Architectural Style Classification with EfficientNet-B3

This project focuses on classifying architectural styles from images using a
transfer learning approach based on **EfficientNet-B3** and **PyTorch**.

The main goal is to build a robust image classification pipeline that includes
data preparation, model training, evaluation, error analysis, and visual
explanations using Grad-CAM.

---

## Dataset

- The dataset consists of architectural images grouped by style.
- Class names are inferred automatically from folder names.
- Some image files may be corrupted or missing.

To handle this safely, a custom dataset class (`SafeImageFolder`) is implemented
to **skip corrupted images without interrupting training**.


### Dataset Sources and Curation

The dataset was constructed by combining images from multiple public sources:

- **Architectural Styles Dataset (Kaggle)**  
  The base dataset and class names (25 architectural styles) were obtained from:  
  https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset

- **Architectural Styles Periods Dataset (Kaggle)**  
  An additional dataset providing period-based architectural categorizations:  
  https://www.kaggle.com/datasets/gustavoachavez/architectural-styles-periods-dataset

Overlapping architectural styles between the two datasets were carefully
identified and merged to ensure label consistency.

The final dataset was manually curated before applying offline data
augmentation.

---

##  Offline Data Augmentation

Offline data augmentation was performed **separately** before training.

- Original dataset size: ~18,000 images  
- After offline augmentation: ~75,000 images  

Augmentation was applied **outside** this notebook to:
- Keep the training pipeline clean
- Avoid repeated augmentation during runtime
- Improve class balance and model generalization

This notebook focuses on **training, evaluation, and analysis** using the
augmented dataset.

---

##  Model Architecture

- Backbone: **EfficientNet-B3** (ImageNet pretrained)
- Transfer learning is applied by:
  - Keeping pretrained feature extractor
  - Replacing the final classification layer with a new `Linear` layer
- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`

---

## Training Pipeline

The training pipeline includes:

- Train / Validation / Test split
- Batch loading with `DataLoader`
- Training and validation loops per epoch
- Best model selection based on **validation loss**
- Model checkpoint saving

Training metrics tracked:
- Training loss & accuracy
- Validation loss & accuracy

---

## Evaluation & Analysis

After training, the model is evaluated on the test set using:

- Test accuracy and loss
- Confusion Matrix
- Classification Report (Precision / Recall / F1-score)
- Error analysis showing the most frequently confused class pairs

All plots and analysis outputs are saved for further inspection.

> Plots were generated from a completed training run and saved for reference.


---

## Grad-CAM Visualization

To improve interpretability, **Grad-CAM** is used to visualize
which regions of an image influence the model’s predictions.

- Grad-CAM is applied to the last convolutional block of EfficientNet-B3
- Heatmaps are overlaid on original images
- Sample visualizations are generated from the test set

---

##  Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📌 Notes

- The notebook is structured with clear sections using Markdown for readability.
- Training outputs and visualizations are intentionally separated from core logic.
- The repository is designed for clarity, reproducibility, and educational value.
