import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib, json

df_train = pd.read_csv("data/processed/train.csv")
df_test = pd.read_csv("data/processed/test.csv")
X_train, y_train = df_train.drop("target",1), df_train["target"]
X_test, y_test = df_test.drop("target",1), df_test["target"]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:,1]

metrics = {
  "accuracy": accuracy_score(y_test,pred),
  "precision": precision_score(y_test,pred),
  "recall": recall_score(y_test,pred),
  "roc_auc": roc_auc_score(y_test,proba)
}

with open("metrics.json","w") as f: json.dump(metrics,f)

import os
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/heart_clf.pkl")
