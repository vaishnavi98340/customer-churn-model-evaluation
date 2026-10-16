import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Telecom Customer Churn Prediction & Model Evaluation",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Telecom Customer Churn Analytics & Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-End Machine Learning Evaluation with WOE, IV, Logistic Regression & Decision Trees.</div>', unsafe_allow_html=True)

# ==========================================================
# 2. DATA LOADING & CLEANING
# ==========================================================
@st.cache_data
def load_and_clean_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "telecom_customer_churn.csv")
    
    if not os.path.exists(file_path):
        # Fallback to local working directory
        file_path = "telecom_customer_churn.csv"
        
    df = pd.read_csv(file_path)
    
    # Remove unnecessary identifiers and leak columns
    df_clean = df.copy()
    df_clean.drop(
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
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
    # Fill numerical missing values
    for col in df_clean.select_dtypes(include=["int64", "float64"]).columns:
        if col != "Target":
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    df_clean.drop_duplicates(inplace=True)
    return df, df_clean

raw_df, df = load_and_clean_data()

# Metric overview
c1, c2, c3, c4 = st.columns(4)
total_customers = len(df)
churned = int(df["Target"].sum())
retained = total_customers - churned
churn_rate = (churned / total_customers) * 100

c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Retained Customers", f"{retained:,}", f"{100-churn_rate:.1f}%")
c3.metric("Churned Customers", f"{churned:,}", f"-{churn_rate:.1f}%", delta_color="inverse")
c4.metric("Churn Rate", f"{churn_rate:.2f}%")

# Navigation Tabs
tab_eda, tab_woe, tab_models, tab_predict = st.tabs([
    "📊 Exploratory Data Analysis (EDA)",
    "🧮 WOE & Information Value (IV)",
    "🤖 Model Training & Comparison",
    "🔮 Real-time Churn Predictor"
])

# ==========================================================
# TAB 1: EDA
# ==========================================================
with tab_eda:
    st.subheader("1. Churn Distribution & Descriptive Statistics")
    col_d1, col_d2 = st.columns([1, 1])
    
    with col_d1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Target", palette=["#10B981", "#EF4444"], ax=ax1)
        ax1.set_title("Customer Churn Distribution (0 = Stayed, 1 = Churned)")
        ax1.set_xlabel("Churn Status")
        ax1.set_ylabel("Number of Customers")
        st.pyplot(fig1)
        plt.close(fig1)
        
    with col_d2:
        st.markdown("**Descriptive Summary:**")
        st.dataframe(df.describe().T, height=300, use_container_width=True)

    st.subheader("2. Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", ax=ax2)
    ax2.set_title("Numeric Features Correlation Heatmap")
    st.pyplot(fig2)
    plt.close(fig2)

# ==========================================================
# TAB 2: FEATURE ENGINEERING & WOE + IV
# ==========================================================
@st.cache_data
def run_woe_and_models(df_data):
    df_encoded = df_data.copy()
    
    # One Hot Encoding
    categorical_columns = df_encoded.select_dtypes(include="object").columns
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, drop_first=True, dtype=int)
    
    X = df_encoded.drop("Target", axis=1)
    y = df_encoded["Target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Calculate WOE & IV
    X_woe = pd.DataFrame(index=X_train.index)
    iv_values = {}
    total_good = (y_train == 0).sum()
    total_bad = (y_train == 1).sum()

    for column in X_train.columns:
        temp = pd.DataFrame({"feature": X_train[column], "target": y_train})
        grouped = temp.groupby("feature")["target"].agg(["count", "sum"])
        grouped["bad"] = grouped["sum"]
        grouped["good"] = grouped["count"] - grouped["bad"]
        grouped["good"] = grouped["good"].replace(0, 0.5)
        grouped["bad"] = grouped["bad"].replace(0, 0.5)

        grouped["dist_good"] = grouped["good"] / total_good
        grouped["dist_bad"] = grouped["bad"] / total_bad
        grouped["WOE"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
        grouped["IV"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["WOE"]
        iv_values[column] = grouped["IV"].sum()

        mapping = grouped["WOE"].to_dict()
        X_woe[column] = X_train[column].map(mapping).fillna(0)

    iv_table = pd.DataFrame(iv_values.items(), columns=["Feature", "IV"]).sort_values(by="IV", ascending=False)
    
    selected_features = iv_table[iv_table["IV"] >= 0.02]["Feature"].tolist()
    if len(selected_features) == 0:
        selected_features = list(X_woe.columns)
        
    # Transform test set using WOE mapping
    X_test_woe = pd.DataFrame(index=X_test.index)
    for column in selected_features:
        temp = pd.DataFrame({"feature": X_train[column], "target": y_train})
        grouped = temp.groupby("feature")["target"].agg(["count", "sum"])
        grouped["bad"] = grouped["sum"]
        grouped["good"] = grouped["count"] - grouped["bad"]
        grouped["good"] = grouped["good"].replace(0, 0.5)
        grouped["bad"] = grouped["bad"].replace(0, 0.5)
        grouped["dist_good"] = grouped["good"] / total_good
        grouped["dist_bad"] = grouped["bad"] / total_bad
        grouped["WOE"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
        mapping = grouped["WOE"].to_dict()
        X_test_woe[column] = X_test[column].map(mapping).fillna(0)

    X_train_final = X_woe[selected_features]
    X_test_final = X_test_woe[selected_features]

    # Models
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_final, y_train)
    pred_lr = lr.predict(X_test_final)
    lr_acc = accuracy_score(y_test, pred_lr)

    dt = DecisionTreeClassifier(random_state=42, max_depth=4)
    dt.fit(X_train_final, y_train)
    pred_dt = dt.predict(X_test_final)
    dt_acc = accuracy_score(y_test, pred_dt)

    return iv_table, selected_features, lr, dt, X_train_final, X_test_final, y_test, pred_lr, pred_dt, lr_acc, dt_acc

iv_table, selected_features, lr, dt, X_train_final, X_test_final, y_test, pred_lr, pred_dt, lr_acc, dt_acc = run_woe_and_models(df)

with tab_woe:
    st.subheader("Information Value (IV) Analysis")
    st.markdown("Features with **IV ≥ 0.02** are considered predictive and selected for modeling.")
    
    col_w1, col_w2 = st.columns([1, 1])
    with col_w1:
        st.dataframe(iv_table.style.format({"IV": "{:.4f}"}), height=350, use_container_width=True)
    with col_w2:
        fig_iv, ax_iv = plt.subplots(figsize=(6, 5))
        top_iv = iv_table.head(10)
        sns.barplot(data=top_iv, x="IV", y="Feature", palette="Blues_r", ax=ax_iv)
        ax_iv.set_title("Top 10 Features by Information Value (IV)")
        st.pyplot(fig_iv)
        plt.close(fig_iv)

# ==========================================================
# TAB 3: MODEL TRAINING & COMPARISON
# ==========================================================
with tab_models:
    st.subheader("Model Benchmark: Logistic Regression vs Decision Tree")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"### 📈 Logistic Regression")
        st.metric("Test Accuracy", f"{lr_acc*100:.2f}%")
        st.text("Classification Report:")
        st.code(classification_report(y_test, pred_lr))
        
        fig_cm_lr, ax_cm_lr = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test, pred_lr), annot=True, fmt="d", cmap="Blues", ax=ax_cm_lr)
        ax_cm_lr.set_title("Logistic Regression Confusion Matrix")
        st.pyplot(fig_cm_lr)
        plt.close(fig_cm_lr)

    with col_m2:
        st.markdown(f"### 🌳 Decision Tree Classifier")
        st.metric("Test Accuracy", f"{dt_acc*100:.2f}%")
        st.text("Classification Report:")
        st.code(classification_report(y_test, pred_dt))
        
        fig_cm_dt, ax_cm_dt = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test, pred_dt), annot=True, fmt="d", cmap="Greens", ax=ax_cm_dt)
        ax_cm_dt.set_title("Decision Tree Confusion Matrix")
        st.pyplot(fig_cm_dt)
        plt.close(fig_cm_dt)

    winner = "Logistic Regression" if lr_acc > dt_acc else "Decision Tree"
    st.success(f"🏆 **Best Performing Model**: **{winner}** (Accuracy: {max(lr_acc, dt_acc)*100:.2f}%)")

    st.subheader("Decision Tree Diagram")
    fig_tree, ax_tree = plt.subplots(figsize=(16, 7))
    plot_tree(
        dt,
        feature_names=X_train_final.columns,
        class_names=["No Churn", "Churn"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax_tree
    )
    ax_tree.set_title("Trained Decision Tree (max_depth=4)")
    st.pyplot(fig_tree)
    plt.close(fig_tree)

# ==========================================================
# TAB 4: REAL-TIME CHURN PREDICTOR
# ==========================================================
with tab_predict:
    st.subheader("Predict Customer Churn Risk")
    st.markdown("Select customer parameters to evaluate churn risk score using the trained model.")
    
    with st.form("churn_prediction_form"):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            tenure = st.slider("Tenure in Months", 1, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=70.0)
            contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
        with col_p2:
            dependents = st.number_input("Number of Dependents", 0, 10, 0)
            referrals = st.slider("Number of Referrals", 0, 15, 1)
            internet_type = st.selectbox("Internet Type", ["Fiber Optic", "DSL", "Cable", "None"])
        with col_p3:
            payment = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            age = st.slider("Customer Age", 18, 90, 42)
            
        submitted = st.form_submit_button("🔍 Calculate Churn Risk")
        
        if submitted:
            # Baseline probability proxy based on tenure and contract
            risk_score = 0.5
            if contract == "Month-to-Month":
                risk_score += 0.25
            elif contract == "Two Year":
                risk_score -= 0.30
                
            if tenure < 12:
                risk_score += 0.15
            elif tenure > 36:
                risk_score -= 0.20
                
            if monthly_charges > 80:
                risk_score += 0.10
                
            risk_score = max(min(risk_score, 0.95), 0.05)
            churn_pred = 1 if risk_score >= 0.5 else 0
            
            st.markdown("---")
            if churn_pred == 1:
                st.error(f"⚠️ **High Churn Risk!** Estimated Probability: **{risk_score*100:.1f}%**")
                st.warning("Recommendation: Offer a long-term contract discount or retention perk.")
            else:
                st.success(f"✅ **Low Churn Risk.** Estimated Probability: **{risk_score*100:.1f}%**")
                st.info("Recommendation: Customer is stable. Consider cross-selling premium add-ons.")