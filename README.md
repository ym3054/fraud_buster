# 🛡️ Fraud Buster ML

**Fraud Buster ML** is a complete machine learning pipeline for detecting fraudulent e-commerce transactions. It demonstrates data preprocessing, feature engineering, class imbalance handling with SMOTE, model training, evaluation, and visualization.

---

## 📁 Project Structure

```
Fraud-Buster-ML/
├── utils.py                       # All preprocessing, modeling, and visualization functions
├── Fraud_Buster_Refactored.ipynb # Main notebook using utils.py functions
```

## 📈 Data Source
- Kaggle: https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions/data
---

## 📊 Features

- Clean & combine datasets
- Time-based feature engineering (hour, weekday, weekend)
- Address and location feature transformation
- Handling class imbalance with `SMOTE`
- Train 3 models:
  - Random Forest
  - Gradient Boosting
  - Ridge Classifier
- Evaluation metrics:
  - Accuracy
  - Confusion Matrix
  - ROC AUC Curve
  - Feature Importance

---

## 🧪 How to Run

1. **Install dependencies**  
```bash
pip install -r requirements.txt
```

2. **Place the data files**  
Download or move the following into `data/`:
- `Fraudulent_E-Commerce_Transaction_Data.csv`
- `Fraudulent_E-Commerce_Transaction_Data_2.csv`

3. **Launch Jupyter Notebook**  
```bash
jupyter notebook Fraud_Buster_Refactored.ipynb
```

---

## 📈 Example Outputs

- ROC Curve for all models
- Feature importance plot (from Random Forest)

---

## 📌 Requirements

- pandas
- numpy
- matplotlib
- scikit-learn
- imbalanced-learn
- jupyter

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📝 License

This project is open source under the MIT License.

---

## 🙌 Acknowledgements

Built as part of a machine learning capstone project for fraud detection.

