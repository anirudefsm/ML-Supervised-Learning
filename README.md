
# MLM Project 2: Supervised Learning Models on Import-Export Dataset

## Project Overview

This project explores supervised learning models to analyze an import-export dataset. Using Python libraries such as Pandas and NumPy, it focuses on data preprocessing, exploratory analysis, and the application of various classification models to derive insights and managerial recommendations.

## Project Contents

-   **Project Information**
-   **Dataset Description**
-   **Data Sampling**
-   **Project Objectives**
-   **Exploratory Data Analysis**
-   **Machine Learning Models**
-   **Findings and Recommendations**

----------

## Dataset Description

-   **Data Source**: [Kaggle Import-Export Dataset](https://www.kaggle.com/datasets/chakilamvishwas/imports-exports-15000)
-   **Columns**:
    -   Transaction_ID, Country, Product, Import_Export, Quantity, Value, Date, Category, Port, Customs_Code, Weight, Shipping_Method, Supplier, Customer, Invoice_Number, Payment_Terms
-   **Data Type**: Panel Data (longitudinal)
-   **Sampling**: A subset of 5,001 entries was used for analysis.

----------

## Objectives

-   Classify data into segments, clusters, or classes using supervised learning.
-   Identify key features and thresholds for classification.
-   Determine the best classification model based on performance metrics.

----------

## Methodology

### 1. **Data Preprocessing**

-   No missing values.
-   Ordinal encoding for categorical variables.
-   Min-Max scaling for numeric features.

### 2. **Exploratory Data Analysis**

-   Descriptive statistics (mean, median, standard deviation, skewness, kurtosis).
-   Data visualization (bar plots, heatmaps, histograms).
-   Inferential statistics (tests for normality, correlation).

### 3. **Supervised Learning Models**

-   Logistic Regression
-   Support Vector Machines (SVM)
-   Stochastic Gradient Descent (SGD)
-   Decision Trees
-   K-Nearest Neighbors (KNN)
-   Naive Bayes
-   Bagging (Random Forest)
-   Boosting (XGBoost)

### 4. **Model Evaluation Metrics**

-   Confusion Matrix (Accuracy, Precision, Recall, F1-Score, AUC)
-   K-Fold Cross-Validation
-   Runtime and memory usage

----------

## Findings

1.  **Data Insights**
    
    -   No missing values in the dataset.
    -   Potential outliers detected but retained for analysis.
    -   Important features include Country, Product, Quantity, Value, Weight, Shipping_Method, Port, and Customs_Code.
2.  **Model Performance**
    
    -   Logistic Regression showed balanced performance with quick runtime.
    -   Random Forest exhibited higher accuracy but required more computational resources.
    -   Decision Tree provided interpretability and moderate accuracy.
3.  **Recommendations**
    
    -   Address class imbalance using oversampling or undersampling techniques.
    -   Enhance feature engineering to improve model accuracy.
    -   Use Logistic Regression for quick insights and Random Forest for more accurate, resource-intensive analyses.

----------

## Requirements

-   Python 3.x
-   Libraries: Pandas, NumPy, Matplotlib, Scikit-learn, Seaborn, XGBoost

----------

## How to Run

1.  Clone the repository.
    
    ```bash
    git clone https://github.com/your-username/mlm-project2.git
    
    ```
    
2.  Install required libraries.
    
    ```bash
    pip install -r requirements.txt
    
    ```
    
3.  Run the notebook file.
    
    ```bash
    jupyter notebook MLM_Project2_055003.ipynb
    
    ```
    

----------

## Author

-   **Anirudh Gupta** (055003)
