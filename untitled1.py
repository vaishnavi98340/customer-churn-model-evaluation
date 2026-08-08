# ==========================================================
# TELECOM CUSTOMER CHURN PREDICTION
# ==========================================================


# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ==========================================================
# 2. DATA LOADING
# ==========================================================

df = pd.read_csv(r"C:\Users\vaishnavi\Downloads\telecom_customer_churn.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())


# ==========================================================
# 3. DATA CLEANING
# ==========================================================

print("\nMissing Values:")
print(df.isnull().sum())


# Remove unnecessary columns

df.drop(
    columns=[
        "Customer ID",
        "Churn Category",
        "Churn Reason",
        "Customer Status"
    ],
    errors="ignore",
    inplace=True
)


# Fill categorical missing values

for col in df.select_dtypes(
    include="object"
).columns:

    df[col] = df[col].fillna(
        df[col].mode()[0]
    )


# Fill numerical missing values

for col in df.select_dtypes(
    include=["int64", "float64"]
).columns:

    if col != "Target":

        df[col] = df[col].fillna(
            df[col].median()
        )


# Remove duplicate rows

df.drop_duplicates(
    inplace=True
)


print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

print("\nDuplicates Removed")


# ==========================================================
# 4. EDA
# ==========================================================

print("\n================ EDA ================")

print("\nDescriptive Statistics:")
print(df.describe())


# Target Distribution

plt.figure(figsize=(6, 4))

sns.countplot(
    x=df["Target"]
)

plt.title(
    "Customer Churn Distribution"
)

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Number of Customers"
)

plt.show()


# Correlation Heatmap

numeric_df = df.select_dtypes(
    include=np.number
)

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Correlation Heatmap"
)

plt.show()


# Histograms

numeric_df.hist(
    figsize=(12, 10)
)

plt.tight_layout()

plt.show()


# Boxplots

for col in numeric_df.columns:

    if col != "Target":

        plt.figure(
            figsize=(5, 3)
        )

        sns.boxplot(
            x=df[col]
        )

        plt.title(
            "Boxplot - " + col
        )

        plt.show()


# ==========================================================
# 5. FEATURE ENGINEERING
# ==========================================================

print("\n================ FEATURE ENGINEERING ================")


# Encode Target

label_encoder = LabelEncoder()

df["Target"] = label_encoder.fit_transform(
    df["Target"]
)


# One Hot Encoding

categorical_columns = df.select_dtypes(
    include="object"
).columns


df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


print("\nFeature Engineering Completed")

print(
    "New Dataset Shape:",
    df.shape
)


# ==========================================================
# 6. X AND Y
# ==========================================================

X = df.drop(
    "Target",
    axis=1
)

y = df["Target"]


# ==========================================================
# 7. TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)


# ==========================================================
# 8. WOE + IV
# ==========================================================

print("\n================ WOE + IV ================")


def calculate_woe_iv(
    X_train,
    y_train
):

    X_woe = pd.DataFrame(
        index=X_train.index
    )

    iv_values = {}


    total_good = (
        y_train == 0
    ).sum()

    total_bad = (
        y_train == 1
    ).sum()


    for column in X_train.columns:

        temp = pd.DataFrame({

            "feature": X_train[column],

            "target": y_train

        })


        grouped = temp.groupby(
            "feature"
        )["target"].agg(
            ["count", "sum"]
        )


        grouped["bad"] = grouped["sum"]

        grouped["good"] = (
            grouped["count"]
            -
            grouped["bad"]
        )


        # Avoid zero values

        grouped["good"] = (
            grouped["good"]
            .replace(0, 0.5)
        )

        grouped["bad"] = (
            grouped["bad"]
            .replace(0, 0.5)
        )


        # Distribution

        grouped["dist_good"] = (
            grouped["good"]
            /
            total_good
        )

        grouped["dist_bad"] = (
            grouped["bad"]
            /
            total_bad
        )


        # WOE

        grouped["WOE"] = np.log(

            grouped["dist_good"]
            /
            grouped["dist_bad"]

        )


        # IV

        grouped["IV"] = (

            grouped["dist_good"]
            -
            grouped["dist_bad"]

        ) * grouped["WOE"]


        iv_values[column] = (
            grouped["IV"].sum()
        )


        # WOE Mapping

        mapping = grouped[
            "WOE"
        ].to_dict()


        X_woe[column] = (

            X_train[column]
            .map(mapping)
            .fillna(0)

        )


    iv_table = pd.DataFrame(

        iv_values.items(),

        columns=[
            "Feature",
            "IV"
        ]

    ).sort_values(

        by="IV",

        ascending=False

    )


    return X_woe, iv_table


# Calculate WOE and IV

X_train_woe, iv_table = calculate_woe_iv(
    X_train,
    y_train
)


print("\nWOE Completed")

print(
    X_train_woe.head()
)


# ==========================================================
# 9. IV TABLE
# ==========================================================

print("\n================ INFORMATION VALUE ================")

print(iv_table)


# Select useful features

selected_features = iv_table[
    iv_table["IV"] >= 0.02
]["Feature"].tolist()


# If no feature passes IV threshold

if len(selected_features) == 0:

    selected_features = list(
        X_train_woe.columns
    )


print("\nSelected Features:")

print(
    selected_features
)


# ==========================================================
# 10. WOE TRANSFORMATION FOR TEST DATA
# ==========================================================

X_test_woe = pd.DataFrame(
    index=X_test.index
)


total_good = (
    y_train == 0
).sum()

total_bad = (
    y_train == 1
).sum()


for column in selected_features:

    temp = pd.DataFrame({

        "feature": X_train[column],

        "target": y_train

    })


    grouped = temp.groupby(
        "feature"
    )["target"].agg(
        ["count", "sum"]
    )


    grouped["bad"] = grouped["sum"]

    grouped["good"] = (
        grouped["count"]
        -
        grouped["bad"]
    )


    grouped["good"] = (
        grouped["good"]
        .replace(0, 0.5)
    )

    grouped["bad"] = (
        grouped["bad"]
        .replace(0, 0.5)
    )


    grouped["dist_good"] = (
        grouped["good"]
        /
        total_good
    )

    grouped["dist_bad"] = (
        grouped["bad"]
        /
        total_bad
    )


    grouped["WOE"] = np.log(

        grouped["dist_good"]
        /
        grouped["dist_bad"]

    )


    mapping = grouped[
        "WOE"
    ].to_dict()


    X_test_woe[column] = (

        X_test[column]
        .map(mapping)
        .fillna(0)

    )


# Final datasets

X_train_final = X_train_woe[
    selected_features
]

X_test_final = X_test_woe[
    selected_features
]


# ==========================================================
# 11. LOGISTIC REGRESSION
# ==========================================================

print("\n================ LOGISTIC REGRESSION ================")


lr = LogisticRegression(
    max_iter=1000
)


lr.fit(
    X_train_final,
    y_train
)


pred_lr = lr.predict(
    X_test_final
)


lr_accuracy = accuracy_score(
    y_test,
    pred_lr
)


print(
    "\nLogistic Regression Accuracy:",
    lr_accuracy
)


print(
    "\nLogistic Regression Classification Report:"
)

print(
    classification_report(
        y_test,
        pred_lr
    )
)


print(
    "\nLogistic Regression Confusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        pred_lr
    )
)


# ==========================================================
# 12. DECISION TREE
# ==========================================================

print("\n================ DECISION TREE ================")


dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=4
)


dt.fit(
    X_train_final,
    y_train
)


pred_dt = dt.predict(
    X_test_final
)


dt_accuracy = accuracy_score(
    y_test,
    pred_dt
)


print(
    "\nDecision Tree Accuracy:",
    dt_accuracy
)


print(
    "\nDecision Tree Classification Report:"
)

print(
    classification_report(
        y_test,
        pred_dt
    )
)


print(
    "\nDecision Tree Confusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        pred_dt
    )
)


# ==========================================================
# 13. DECISION TREE DIAGRAM
# ==========================================================

plt.figure(
    figsize=(25, 12)
)


plot_tree(

    dt,

    feature_names=X_train_final.columns,

    class_names=[
        "No Churn",
        "Churn"
    ],

    filled=True,

    rounded=True,

    fontsize=9

)


plt.title(
    "Decision Tree - Customer Churn Prediction"
)

plt.show()


# ==========================================================
# 14. MODEL COMPARISON
# ==========================================================

print("\n================ MODEL COMPARISON ================")

print(
    "Logistic Regression Accuracy:",
    round(lr_accuracy, 4)
)

print(
    "Decision Tree Accuracy:",
    round(dt_accuracy, 4)
)


if lr_accuracy > dt_accuracy:

    print(
        "\nBest Model: Logistic Regression"
    )

else:

    print(
        "\nBest Model: Decision Tree"
    )


# ==========================================================
# END
# ==========================================================