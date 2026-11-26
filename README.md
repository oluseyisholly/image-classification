# Fashion-MNIST Image Classification

## CNN vs Random Forest

**Course:** DLBAIPCV01 – Project: Computer Vision  
**Task:** Task 2 – Classifying Fashion Products

## Project Overview

This project compares two machine-learning approaches for classifying Fashion-MNIST clothing images:

1. Convolutional Neural Network (CNN)
2. Random Forest Classifier

The models are evaluated using training and test accuracy, precision, recall, F1-score, training time, confusion matrices, per-class performance, and misclassification analysis.

The final production recommendation should be based on the actual experimental results obtained after running the notebook.

## Dataset

The project uses Fashion-MNIST, containing 70,000 grayscale images in 10 clothing categories:

- 60,000 training images
- 10,000 test images
- Image size: 28 × 28 pixels
- 10 balanced classes

| ID | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

The dataset is loaded directly through Keras, so no dataset file needs to be committed.

## Repository Structure

```text
fashion-mnist-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── fashion_mnist_classification.ipynb
├── results/
│   └── README.md
└── report/
    └── README.md
```

After running the notebook, place the generated result files in `results/` and the final university report in `report/`.

## Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    max_depth=100,
    random_state=42,
    n_jobs=-1
)
```

## CNN Architecture

- Input: 28 × 28 × 1
- Conv2D: 32 filters, 3 × 3, ReLU
- MaxPooling2D: 2 × 2
- Conv2D: 64 filters, 3 × 3, ReLU
- MaxPooling2D: 2 × 2
- Flatten
- Dense: 128 units, ReLU
- Output: 10 units, Softmax

Training configuration:

- Optimizer: Adam
- Loss: Sparse Categorical Cross-Entropy
- Epochs: 10
- Batch size: 128
- Validation split: 10%

## Running the Project

1. Open `notebooks/fashion_mnist_classification.ipynb` in Google Colab.
2. Run all cells from top to bottom.
3. Review the generated CNN and Random Forest metrics.
4. Save the generated CSV and PNG files into `results/`.
5. Complete the final report with the actual experimental results.

## Expected Result Files

```text
model_comparison.csv
cnn_classification_report.csv
random_forest_classification_report.csv
cnn_train_confusion_matrix.png
cnn_test_confusion_matrix.png
rf_train_confusion_matrix.png
rf_test_confusion_matrix.png
cnn_accuracy_curve.png
cnn_loss_curve.png
misclassified_examples.png
```

## Important

No final model-performance values are hard-coded in this repository. Accuracy, precision, recall, F1-score, training times, confusion matrices, weakest classes and the production recommendation must come from the executed notebook.

## References

- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747.
- TensorFlow. *Basic classification: Classify images of clothing*.
