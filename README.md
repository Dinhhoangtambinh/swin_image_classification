# swin_image_classification
This project provides a full pipeline for image classification using a Swin Transformer model. It fine-tunes Swin on a custom dataset, the workflow includes data preparation, augmentation, model training, evaluation, and visualization of results.

## Features
- **Data Augmentation**: Uses Albumentations to balance the dataset by generating more samples for underrepresented classes.
- **Pre-trained Model**: Utilizes a Swin Transformer (`swin_base_patch4_window7_224`) pre-trained on ImageNet, fine-tuned for the specific task.
- **Mixed Precision Training**: Speeds up training and saves memory using AMP.
- **Early Stopping**: Stops training when validation loss stops improving, helping to avoid overfitting.
- **Progressive Unfreezing**: Gradually unfreezes layers during training to better adapt the model to the new dataset while making use of learned features.
- **Label Smoothing**: Adds label smoothing to the loss function to reduce overconfidence and improve generalization.
- **Differential Learning Rates and Weight Decay**: Applies different learning rates and weight decay settings to the backbone and classifier head to preserve the learned knowledge in the backbone, ensure training stability.
- **Comprehensive Evaluation**: Reports key metrics like accuracy, precision, recall, F1-score, and includes a classification report and confusion matrix.
- **Visualizations**: Includes plots of training and validation metrics as well as the confusion matrix to help interpret results.

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

## Future Work

- Hyperparameter tuning for improved performance.
- Experimentation with other transformer models.
- Additional data augmentation techniques.
- Exploration of alternative optimization strategies.
