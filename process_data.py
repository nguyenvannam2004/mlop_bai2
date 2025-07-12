import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/heart.csv")
df = df.dropna()
X = df.drop(columns=["target"])
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
train_X, test_X, train_y, test_y = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
pd.DataFrame(train_X, columns=X.columns).assign(target=train_y).to_csv("data/processed/train.csv", index=False)
pd.DataFrame(test_X, columns=X.columns).assign(target=test_y).to_csv("data/processed/test.csv", index=False)
