# Imbalanced Credit Card Fraud Detection: A Comparative Analysis of SVM, Naïve Bayes, and K-NN

<!-- Anda bisa membuat banner kustom di situs seperti canva.com dan menempelkan link gambarnya di sini -->
![Project Banner](https://i.imgur.com/example.png) -->

This project provides an in-depth comparative analysis of Support Vector Machine (SVM), Naïve Bayes, and K-Nearest Neighbors (K-NN) for the task of detecting fraudulent transactions in a highly imbalanced credit card dataset. It serves as a capstone project for the "Introduction to Data Mining" course from BISA AI Academy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

---

### Table of Contents
*   [1. Project Overview](#1-project-overview)
*   [2. Dataset](#2-dataset)
*   [3. Methodology](#3-methodology)
*   [4. Repository Structure](#4-repository-structure)
*   [5. Getting Started](#5-getting-started)
*   [6. Results and Evaluation](#6-results-and-evaluation)
*   [7. Conclusion and Future Work](#7-conclusion-and-future-work)
*   [8. Acknowledgements](#8-acknowledgements)

---

### 1. Project Overview

The core objective of this project is to build and rigorously evaluate several supervised machine learning models to accurately identify fraudulent credit card transactions. Given that fraudulent transactions represent only **0.17%** of the dataset, this project places a strong emphasis on strategies for handling **severe class imbalance**. The models chosen for comparison are **Support Vector Machine (SVM)**, **Gaussian Naïve Bayes**, and **K-Nearest Neighbors (K-NN)**, allowing for an analysis of their performance in this challenging scenario.

Different resampling techniques were strategically employed: **SMOTE (oversampling)** was used for K-NN and Naïve Bayes, while **RandomUnderSampler (undersampling)** was applied for the computationally intensive SVM to ensure efficient training.

---

### 2. Dataset

The dataset used is the "Credit Card Fraud Detection" dataset from Kaggle, originating from a research collaboration between Worldline and the Machine Learning Group of ULB.

*   **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
*   **Content:** The dataset contains anonymized credit card transactions by European cardholders over two days.
*   **Key Characteristics:**
    *   Total Transactions (after cleaning): 283,726
    *   Fraudulent Transactions (Positive Class): 473 (0.17%)
    *   **Features**: 30 numerical features. `V1` to `V28` are principal components from a PCA transformation. `Time` and `Amount` are the only original features.
    *   **Target Variable**: `Class` (1 for fraud, 0 for genuine).
    *   **Main Challenge**: The dataset is **critically unbalanced**, making standard accuracy a misleading performance metric.

---

### 3. Methodology

1.  **Exploratory Data Analysis (EDA):** Visualizing data distributions, analyzing feature characteristics (Time and Amount), and examining the correlation matrix to understand the data's underlying structure.
2.  **Data Pre-processing:**
    *   **Duplicate Removal:** Identified and removed 1,081 duplicate rows.
    *   **Feature Scaling:** Applied `StandardScaler` to the `Time` and `Amount` features to standardize their range, a critical step for distance-based algorithms like SVM and K-NN.
3.  **Data Splitting:** The dataset was split into training (80%) and testing (20%) sets using a stratified approach to maintain the class proportion in both subsets.
4.  **Handling Imbalance (Applied on Training Set ONLY):**
    *   **SMOTE (Synthetic Minority Over-sampling Technique):** Used to oversample the minority class for training the K-NN and Naïve Bayes models.
    *   **RandomUnderSampler:** Used to undersample the majority class for training the SVM model to manage its computational complexity.
5.  **Modeling and Evaluation:**
    *   Three models—**K-NN, Gaussian Naïve Bayes, and SVM (linear kernel)**—were trained on their respective resampled datasets.
    *   Models were evaluated on the **original, untouched test set** using metrics appropriate for imbalanced classification: Confusion Matrix, Precision, Recall, F1-Score, and **AUPRC (Area Under the Precision-Recall Curve)**.

---

### 4. Repository Structure
```
bisa-ai-datamining-capstone-fraud-detection/
│
├── data/                 # Dataset files
├── notebooks/            # Jupyter/Colab notebooks for analysis
├── reports/              # Generated visualizations
│   └── figures/
├── .gitignore            # Git ignore file
├── README.md             # Project documentation (this file)
└── requirements.txt      # Python dependencies
```

---

### 5. Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/bisa-ai-datamining-capstone-fraud-detection.git
    cd bisa-ai-datamining-capstone-fraud-detection
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the data:**
    Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` inside the `data/raw/` directory.

4.  **Run the notebook:**
    Open and run the `bisa-ai-datamining-capstone-fraud-detection.ipynb` notebook located in the `notebooks/` directory.

---

### 6. Results and Evaluation

The models' performance on the unseen test data is summarized below. The primary goal is to maximize **Recall** (identifying as many frauds as possible) while maintaining a reasonable **AUPRC**, which reflects a good balance between precision and recall.

| Model                        | Resampling       | Precision | **Recall** | F1-Score | **AUPRC** | AUC-ROC |
| ---------------------------- | ---------------- | --------- | ---------- | -------- | --------- | ------- |
| K-Nearest Neighbors          | SMOTE (Oversample) | 0.4699    | 0.8211     | 0.5977   | 0.5221    | 0.9100  |
| Gaussian Naïve Bayes         | SMOTE (Oversample) | 0.0540    | 0.8105     | 0.1012   | 0.0789    | 0.9447  |
| **Support Vector Machine**   | **Undersample**  | 0.0505    | **0.8737** | 0.0955   | **0.5815**| 0.9522  |

![Model Performance Comparison](https://github.com/LatiefDataVisionary/bisa-ai-datamining-capstone-fraud-detection/blob/main/reports/Model%20Performance%20Comparison.png)


#### **Key Observations:**

*   **Best Recall & AUPRC:** The **Support Vector Machine (SVM)** model, despite its lower precision, achieved the highest Recall (**87.37%**) and the best AUPRC (**0.5815**). This indicates it was the most effective at identifying the majority of actual fraudulent transactions.
*   **K-NN Performance:** The K-NN model showed a good balance with the highest precision (**46.99%**) and a strong recall, making it a viable alternative.
*   **Naïve Bayes Challenge:** The Gaussian Naïve Bayes model, while having high recall, suffered from extremely low precision, leading to a large number of false positives.

#### **Performance Visualizations:**

The Precision-Recall and ROC curves below provide a visual comparison of the models' trade-offs. The SVM model's PR curve covers the largest area, confirming its superior AUPRC.

![Precision-Recall Curve](https://github.com/LatiefDataVisionary/bisa-ai-datamining-capstone-fraud-detection/blob/main/reports/Precision-Recall%20Curve%20Comparison.png)
![ROC Curve](https://github.com/LatiefDataVisionary/bisa-ai-datamining-capstone-fraud-detection/blob/main/reports/ROC%20Curve%20Comparison.png)

---

### 7. Conclusion and Future Work

#### **Conclusion**

In the context of fraud detection where minimizing missed frauds (**maximizing Recall**) is paramount, the **Support Vector Machine (SVM) model trained with RandomUnderSampler is the most recommended model** from this comparative study. It successfully identified over 87% of the fraudulent transactions in the test set and demonstrated the best precision-recall trade-off as indicated by the AUPRC score. The strategic use of undersampling was also key to making the SVM training computationally feasible.

#### **Future Work**

To further enhance this project, the following steps could be taken:
*   **Hyperparameter Tuning:** Fine-tune the hyperparameters for SVM and K-NN to potentially boost their precision and F1-Scores.
*   **Advanced Models:** Implement more sophisticated algorithms like LightGBM, XGBoost, or ensemble methods.
*   **Feature Engineering:** Explore the creation of new features from the existing data to provide more predictive power.
*   **Cost-Sensitive Learning:** Implement models where the cost of a False Negative (missing a fraud) is explicitly set higher than a False Positive.

---

### 8. Acknowledgements

This project utilizes a dataset provided by:
*   The Machine Learning Group (MLG) of ULB (Université Libre de Bruxelles) and Worldline.
*   Full credit and acknowledgements go to the original creators of this invaluable dataset.
