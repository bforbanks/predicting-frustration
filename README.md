# Predicting Frustration from Heart-Rate Signal with Machine Learning Models

## Overview

This repository contains the code and data used for the paper [Predicting frustration from heart-rate signal with Machine Learning Models](Predicting_frustration_from_heart_rate_signal_with_Machine_Learning_Models.pdf). The goal of the project is to predict frustration levels from heart rate signals using various machine learning models, focusing on their generalizability to new individuals.

## Table of Contents

- [Dependencies](#dependencies)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [References](#references)


## Dependencies

The project uses the following Python packages:

- numpy
- pandas
- scikit-learn
- imbalanced-learn
- statsmodels
- tqdm

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Usage

To script was developed and run using iPython. 

It is recommended to run the script in an iPython environment.

## Data Preprocessing

1. **Dataset Description**: The dataset is a subset of the EmoPairCompete dataset and contains 168 observations with 11 attributes. The features used for prediction include HR Mean, HR Median, HR Std, HR Min, HR Max, and HR AUC. The target variable is the self-rated frustration level.

2. **Binarization of Target Variable**: Frustration levels are categorized as 'frustrated' (5 and above) and 'not frustrated' (below 5), leading to a class imbalance of 14% frustrated and 86% not-frustrated.

3. **Mitigating Class Imbalance and Scaling**: Synthetic Minority Over-sampling Technique (SMOTE) is applied to the training data to balance the classes. Subsequently, the data is standardized by removing the mean and scaling to unit variance.

## Model Training and Evaluation

The following machine learning models were trained and evaluated:

- Random Forest (RF)
- Logistic Regression (LR)
- K-Nearest Neighbors (KNN)
- AdaBoost (ADA)

Additionally, three baseline classifiers were used for comparison:

- Stratified Dummy Classifier (Base[S])
- Positive Class Classifier (Base[+])
- Negative Class Classifier (Base[-])

### Cross-Validation

A two-layer cross-validation approach was used. The outer layer uses Leave-One-Group-Out cross-validation, ensuring generalization to new individuals. The inner layer performs a grid search for hyperparameter tuning.

### Performance Metrics

The following metrics were used to evaluate the models:

- F1 Score
- Balanced Accuracy
- Precision
- Recall
- Negative Predictive Value (NPV)
- Matthews Correlation Coefficient (MCC)
- ROC-AUC

### Statistical Tests

Cochran's Q test was used to check for differences in performance between models. Pairwise McNemar tests were performed for pairwise comparisons, with a Bonferroni-adjusted significance threshold.

## Results

The performance of the models was as follows (refer to the paper for detailed metrics):

- **Possible Best Performing Model**: K-Nearest Neighbors (KNN)
- **Worst Performing Model**: Logistic Regression (LR)
- **Baseline Performance**: The stratified dummy classifier (Base[S]) performed comparably to some machine learning models.

## Acknowledgments

The dataset used in this project is a subset of the EmoPairCompete dataset.


## References

- Bowyer, K. W., Chawla, N. V., Hall, L. O., & Kegelmeyer, W. P. (2011). SMOTE: Synthetic Minority Over-sampling Technique. CoRR, abs/1106.1813. Available at: [http://arxiv.org/abs/1106.1813](http://arxiv.org/abs/1106.1813)

- Baldi, P., Brunak, S., Chauvin, Y., Andersen, C. A. F., & Nielsen, H. (2000). Assessing the accuracy of prediction algorithms for classification: an overview. Bioinformatics, 16(5), 412-424. Available at: [https://doi.org/10.1093/bioinformatics/16.5.412](https://doi.org/10.1093/bioinformatics/16.5.412)

- Das, S., Lund, N. L., Ramos González, C., & Clemmensen, L. H. (2024). EmoPairCompete - Physiological Signals Dataset for Emotion and Frustration Assessment under Team and Competitive Behaviors. In ICLR 2024 Workshop on Learning from Time Series For Health. Available at: [https://openreview.net/forum?id=BvgAzJX40Z](https://openreview.net/forum?id=BvgAzJX40Z)

