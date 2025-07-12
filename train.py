import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Đọc dữ liệu
df_train = pd.read_csv("data/processed/train.csv")
df_test = pd.read_csv("data/processed/test.csv")

# Bỏ các dòng có NaN trong target (phòng ngừa)
df_train = df_train.dropna(subset=["target"])
df_test = df_test.dropna(subset=["target"])

# Tách feature và nhãn
X_train, y_train = df_train.drop("target", axis=1), df_train["target"]
X_test, y_test = df_test.drop("target", axis=1), df_test["target"]

# Huấn luyện mô hình
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Dự đoán
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:, 1]

# Tính ROC AUC có kiểm tra số lớp
if len(set(y_test)) > 1:
    roc_auc = roc_auc_score(y_test, proba)
else:
    warnings.warn("Only one class present in y_test. ROC AUC is undefined.", UndefinedMetricWarning)
    roc_auc = None

# Tính metrics
metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred, zero_division=0),
    "recall": recall_score(y_test, pred, zero_division=0),
    "roc_auc": roc_auc
}

# Ghi metrics
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Tạo thư mục model
os.makedirs("model", exist_ok=True)

# Lưu model
joblib.dump(clf, "model/heart_clf.pkl")
