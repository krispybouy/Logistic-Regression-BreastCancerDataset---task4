# Task 4: Classification with Logistic Regression (Breast Cancer Dataset)

This repository contains my work for Task 4, where I built a binary classification model using Logistic Regression to predict whether a tumor is malignant or benign based on features from the Breast Cancer Wisconsin dataset.

## Overview

The main objective was to train and evaluate a logistic regression model that could classify tumors using the given features extracted from digitized breast mass images.

### Data Preprocessing

I began by loading the data.csv file and cleaning it up before model training.
Here’s what I did step by step:

Missing Values: Checked for null values using .isna().sum(). Found that the column Unnamed: 32 was completely empty, so I removed it.

Label Encoding: Converted the target column diagnosis from categorical values (M/B) to numeric labels (1 for malignant, 0 for benign) using LabelEncoder.

Feature Standardization: Applied StandardScaler to all numerical columns (except the ID column) to bring them to a similar scale, which helps logistic regression converge better.

### Sigmoid Function — Core of Logistic Regression

Logistic regression relies on the sigmoid (logistic) function to convert linear predictions into probabilities between 0 and 1:

σ(z) = 1 / (1 + e^(-z))
Here, z represents the linear combination of input features.

Why it’s important:
- Maps any real number into the range (0, 1) — ideal for probability outputs
- Has an S-shaped curve, meaning small and large inputs saturate towards 0 and 1
- When z = 0, σ(z) = 0.5 — the default classification cutoff
- As z increases, σ(z) approaches 1 (more confident in the positive class)
- As z decreases, σ(z) approaches 0 (more confident in the negative class)

This mapping makes logistic regression a natural choice for binary classification problems, as it outputs probabilities rather than hard labels.

### Model Development

The dataset was split into training (80%) and testing (20%) subsets using train_test_split, with random_state=42 for consistent results.
I then trained a logistic regression model using the standardized features.

### Model Evaluation
Default Threshold (0.5)

Initial results with the default threshold gave strong performance:
ROC-AUC: 0.996 — nearly perfect discrimination between malignant and benign samples
Precision: 0.986 — 97.5% of malignant predictions were correct
Recall: 0.986 — the model successfully identified 90.7% of actual malignant cases
The ROC curve confirmed the model’s robustness, with an area under the curve of 0.99.

### Threshold Adjustment for Medical Use

Why change the threshold?
In cancer diagnosis, false negatives (missing a malignant case) are much riskier than false positives (incorrectly predicting malignancy).
A false negative could delay treatment, while a false positive usually just leads to extra testing — which, though inconvenient, is far safer.

Tuned Threshold (0.3)
To reduce false negatives, I lowered the decision threshold from 0.5 to 0.3, which made the model more sensitive:
Recall: 1.00
Precision: 0.9730
Recall: Improved to 0.9535 — correctly identifying around 95.4% of malignant cases

This trade-off slightly reduces precision but significantly improves patient safety by minimizing the chance of missed diagnoses.
