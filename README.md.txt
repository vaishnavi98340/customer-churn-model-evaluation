# Telecom Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn using Machine Learning.

Customer churn means a customer stops using the services of a telecom company. The objective of this project is to analyze customer information and build classification models that can predict whether a customer is likely to churn.

Two Machine Learning algorithms are used:

* Logistic Regression
* Decision Tree Classifier

The project also includes WOE (Weight of Evidence) and IV (Information Value) for feature transformation and feature selection.

---

## Dataset

The project uses a Telecom Customer Churn dataset containing customer demographic, service, contract, billing, and revenue-related information.

The target variable is:

```text
Target
```

Target values:

```text
0 → No Churn
1 → Churn
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Machine Learning Workflow

The project follows this workflow:

```text
Data Loading
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
WOE Encoding
      ↓
Information Value
      ↓
Feature Selection
      ↓
Logistic Regression
      ↓
Decision Tree
      ↓
Model Evaluation
      ↓
Model Comparison
```

---

## 1. Data Loading

The dataset is loaded using Pandas.

The dataset is inspected using:

* First few rows
* Dataset shape
* Column information
* Missing value information

---

## 2. Data Cleaning

The following data cleaning operations are performed:

* Missing value detection
* Missing value treatment
* Duplicate removal
* Removal of unnecessary columns

The following columns are removed:

* `Customer ID`
* `Churn Category`
* `Churn Reason`
* `Customer Status`

These columns are not used as input features for the churn prediction model.

---

## 3. Exploratory Data Analysis

EDA is performed to understand the data before building the models.

The project includes:

### Target Distribution

Shows the number of customers who churned and did not churn.

### Correlation Heatmap

Shows the relationship between numerical variables.

### Histograms

Used to understand the distribution of numerical variables.

### Boxplots

Used to identify the distribution and possible outliers in numerical variables.

---

## 4. Feature Engineering

Categorical variables are converted into numerical variables using One-Hot Encoding.

The target variable is encoded using Label Encoding.

This makes the dataset suitable for Machine Learning algorithms.

---

## 5. WOE

Weight of Evidence (WOE) is used to transform features based on their relationship with the target variable.

WOE is calculated manually in the project, so the `category_encoders` library is not required.

---

## 6. Information Value

Information Value (IV) is calculated to measure the predictive power of the features.

Features with an IV value of at least `0.02` are selected for modeling.

---

## 7. Logistic Regression

Logistic Regression is used to predict customer churn.

The model performance is evaluated using:

* Accuracy
* Classification Report
* Confusion Matrix

---

## 8. Decision Tree

A Decision Tree Classifier is used as the second Machine Learning model.

The Decision Tree is also visualized using a tree diagram.

The model is evaluated using:

* Accuracy
* Classification Report
* Confusion Matrix

---

## 9. Model Comparison

The performance of both models is compared using accuracy.

```text
Logistic Regression Accuracy
            vs
Decision Tree Accuracy
```

The model with better accuracy is selected as the better-performing model for this project.

---

## Project Structure

```text
Telecom-Customer-Churn/
│
├── telecom_customer_churn.csv
│
├── churn_prediction.py
│
├── requirements.txt
│
├── workflow.md
│
├── README.md
│
└── decision_tree.png
```

---

## Installation

Clone the repository and install the required libraries.

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the Python file:

```bash
python churn_prediction.py
```

The program performs:

1. Data loading
2. Data cleaning
3. EDA
4. Feature engineering
5. WOE calculation
6. IV calculation
7. Logistic Regression
8. Decision Tree
9. Model evaluation
10. Decision Tree visualization

---

## Results

The project compares Logistic Regression and Decision Tree based on their test accuracy.

The final output displays:

* Logistic Regression accuracy
* Decision Tree accuracy
* Classification reports
* Confusion matrices
* Decision Tree visualization

The actual accuracy values depend on the dataset and the execution environment.

---

## Key Learning

This project demonstrates the complete Machine Learning workflow for a binary classification problem.

Important concepts covered:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* One-Hot Encoding
* WOE
* Information Value
* Feature Selection
* Logistic Regression
* Decision Tree
* Model Evaluation

---

## Conclusion

The Telecom Customer Churn Prediction project demonstrates how customer data can be processed and used to build Machine Learning classification models.

Logistic Regression and Decision Tree are trained and compared to determine which model performs better for predicting customer churn.

```
```
