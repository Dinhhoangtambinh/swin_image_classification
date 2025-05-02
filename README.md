# swin_image_classification
A complete pipeline for classifying images into predefined categories using a pre-trained Swin Transformer model, fine-tuned on a custom dataset. It includes data preparation, augmentation, model training, evaluation, and visualization of results.

## Features
- **Data Augmentation**: Addresses class imbalance by generating additional images for underrepresented classes using Albumentations.
- **Pre-trained Model**: Utilizes a Swin Transformer (`swin_base_patch4_window7_224`) pre-trained on ImageNet, fine-tuned for the specific task.
- **Mixed Precision Training**: Employs AMP for faster training and reduced memory usage.
- **Early Stopping**: Monitors validation loss to prevent overfitting.
- **Progressive Unfreezing**: Progressive Unfreezing: Gradually unfreezes layers during training to leverage pre-trained weights while fine-tuning, improving adaptation to the target dataset.
- **Label Smoothing**: Applies label smoothing to the CrossEntropyLoss to reduce overconfidence and enhance generalization.
- **Differential Learning Rates and Weight Decay**: Uses distinct learning rates and weight decay for the classifier head and backbone, with separate handling for bias/norm parameters to optimize training process.
- **Comprehensive Evaluation**: Computes accuracy, precision, recall, F1-score, and generates a classification report and confusion matrix.
- **Visualizations**: Plots training/validation curves and confusion matrix for performance insights.

## Dataset

The dataset should be organized into three directories: `train`, `val`, and `test`, each containing subdirectories for each class.

Example structure:
```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

## Results

Sample results from a run:
- Accuracy: 0.9330
- Precision: 0.9278
- Recall: 0.9203
- F1-score: 0.9226

Detailed classification reports and visualizations are generated within the notebook.

## Results

Sample results from a run:
- Accuracy: 0.9330
- Precision: 0.9278
- Recall: 0.9203
- F1-score: 0.9226

Detailed classification reports and visualizations are generated within the notebook.
