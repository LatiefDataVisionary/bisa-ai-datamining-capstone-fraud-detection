# Imbalanced Credit Card Fraud Detection: A Comparative Analysis of SVM, Naïve Bayes, and K-NN

![Banner](https://i.imgur.com/your-banner-image-url.png) <!-- Opsional: Anda bisa buat banner keren di canva.com -->

A comprehensive analysis and implementation of machine learning models to detect fraudulent credit card transactions on a highly imbalanced dataset. This project is a capstone for the "Introduction to Data Mining" course from BISA AI Academy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

---

### Table of Contents
*   [1. Project Overview](#1-project-overview)
*   [2. Dataset](#2-dataset)
*   [3. Objectives](#3-objectives)
*   [4. Methodology](#4-methodology)
*   [5. Repository Structure](#5-repository-structure)
*   [6. Getting Started](#6-getting-started)
*   [7. Results and Evaluation](#7-results-and-evaluation)
*   [8. Acknowledgements](#8-acknowledgements)

---

### 1. Project Overview

The primary goal of this project is to build and evaluate several supervised machine learning models to accurately identify fraudulent credit card transactions. Given the nature of the dataset—where fraudulent transactions are extremely rare—this project places a strong emphasis on handling **class imbalance**. The models explored are **Support Vector Machine (SVM)**, **Naïve Bayes**, and **K-Nearest Neighbors (K-NN)**.

---

### 2. Dataset

The dataset used in this project is the "Credit Card Fraud Detection" dataset from Kaggle, sourced from a research collaboration by Worldline and the Machine Learning Group of ULB.

*   **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
*   **Content:** The dataset contains anonymized credit card transactions made by European cardholders over a two-day period in September 2013.
*   **Key Characteristics:**
    *   Total Transactions: 284,807
    *   Fraudulent Transactions: 492 (0.172% of the total)
    *   **Features**: It contains 30 numerical features. `V1` to `V28` are the result of a PCA transformation to protect user privacy. `Time` and `Amount` are the only features that have not been transformed.
    *   **Target Variable**: `Class` (1 for fraudulent transactions, 0 for genuine transactions).
    *   **Challenge**: The dataset is **highly unbalanced**, making accuracy a poor performance metric.

---

### 3. Objectives

*   To perform a thorough Exploratory Data Analysis (EDA) to understand the data distribution and feature characteristics.
*   To correctly pre-process the data, including feature scaling and addressing the severe class imbalance.
*   To build, train, and compare three classification models: SVM, Naïve Bayes, and K-NN.
*   To evaluate the models using appropriate metrics for imbalanced datasets, such as Precision-Recall AUC, F1-Score, and the Confusion Matrix.

---

### 4. Methodology

1.  **Exploratory Data Analysis (EDA):** Analyzing feature distributions and the relationship between features and the target variable.
2.  **Data Pre-processing:**
    *   **Scaling:** Applying `StandardScaler` to the `Amount` and `Time` features to ensure all features have a similar scale, which is crucial for distance-based algorithms like SVM and K-NN.
    *   **Handling Imbalance:** Implementing a resampling technique (e.g., **SMOTE** - Synthetic Minority Over-sampling Technique or **RandomUnderSampler**) to create a balanced dataset for model training.
3.  **Modeling:**
    *   **Support Vector Machine (SVM):** A powerful model that finds an optimal hyperplane to separate classes.
    *   **Gaussian Naïve Bayes:** A probabilistic classifier based on Bayes' theorem with an assumption of feature independence.
    *   **K-Nearest Neighbors (K-NN):** A non-parametric algorithm that classifies a data point based on the majority class of its 'k' nearest neighbors.
4.  **Evaluation:**
    *   Splitting the data into training and testing sets.
    *   Using a **Confusion Matrix**, **Precision**, **Recall**, **F1-Score**, and **Area Under the Precision-Recall Curve (AUPRC)** to evaluate model performance, as recommended for imbalanced classification.

---

### 5. Repository Structure
```
credit-card-fraud-detection/
│
├── data/                 # Dataset files
├── notebooks/            # Jupyter/Colab notebooks for analysis and modeling
├── reports/              # Generated visualizations and figures
│   └── figures/
├── .gitignore            # Git ignore file
├── README.md             # Project documentation (this file)
└── requirements.txt      # Python dependencies
```

---

### 6. Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/LatiefDataVisionary/bisa-ai-datamining-capstone-fraud-detection
.git
    cd credit-card-fraud-detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the data:**
    Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` inside the `data/raw/` directory.

5.  **Run the notebook:**
    Open and run the `1-EDA-Preprocessing-and-Modeling.ipynb` notebook located in the `notebooks/` directory using Jupyter Notebook or Google Colab.

---

### 7. Results and Evaluation

*(This section will be updated with the final model performance metrics, comparisons, and key findings after the analysis is complete.)*

**Placeholder for a results table:**

| Model                 | AUPRC | F1-Score | Recall | Precision |
| --------------------- | ----- | -------- | ------ | --------- |
| K-Nearest Neighbors   | -     | -        | -      | -         |
| Gaussian Naïve Bayes  | -     | -        | -      | -         |
| Support Vector Machine| -     | -        | -      | -         |


---

### 8. Acknowledgements

This project uses a dataset provided by:
*   The Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles).
*   The work is a result of a research collaboration with Worldline.
*   Please cite their original works if you use this dataset in your research.
