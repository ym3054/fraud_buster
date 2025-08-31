import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

def load_data():
    df1 = pd.read_csv('data/Fraudulent_E-Commerce_Transaction_Data.csv')
    df2 = pd.read_csv('data/Fraudulent_E-Commerce_Transaction_Data_2.csv')
    return df1, df2

def clean_data(df1, df2):
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def feature_engineer(df):
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    df['transaction_hour'] = df['Transaction Date'].dt.hour
    df['transaction_dayofweek'] = df['Transaction Date'].dt.dayofweek
    df['is_weekend'] = df['transaction_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['log_amount'] = np.log1p(df['Transaction Amount'])
    df['amount_per_item'] = df['Transaction Amount'] / (df['Quantity'].replace(0, np.nan))
    df['location_device'] = df['Customer Location'].astype(str) + "_" + df['Device Used'].astype(str)
    df['same_address'] = (df['Shipping Address'] == df['Billing Address']).astype(int)
    df = df.drop(['Transaction ID', 'Customer ID', 'Transaction Date',
                  'IP Address', 'Shipping Address', 'Billing Address'], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    return df

def balance_and_scale(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    num_cols = X_resampled.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X_resampled[num_cols] = scaler.fit_transform(X_resampled[num_cols])
    return X_resampled, y_resampled

def train_models(X_train, y_train):
    models = {}
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    ridge = RidgeClassifierCV()
    ridge.fit(X_train, y_train)
    models['Ridge Classifier'] = ridge
    return models

def evaluate_model(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f'\n{name} Accuracy: {acc:.4f}')
        print('Confusion Matrix:')
        print(cm)

def plot_roc(models, X_test, y_test):
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        plt.figure(figsize=(8, 5))
        plt.barh(range(top_n), importances[indices], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top Features')
        plt.tight_layout()
        plt.show()
