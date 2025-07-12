import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Tải dữ liệu từ internet nếu chưa có
url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
df = pd.read_csv(url)

# Lưu lại dữ liệu gốc
os.makedirs("data", exist_ok=True)
df.to_csv("data/heart.csv", index=False)

# Xử lý dữ liệu
df = df.dropna()  # Bỏ các dòng có NaN nếu có
X = df.drop(columns=["target"])
y = df["target"]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia train/test CÓ stratify để giữ tỉ lệ class
train_X, test_X, train_y, test_y = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Tạo thư mục output
os.makedirs("data/processed", exist_ok=True)

# Ghi dữ liệu ra file
pd.DataFrame(train_X, columns=X.columns).assign(target=train_y).to_csv("data/processed/train.csv", index=False)
pd.DataFrame(test_X, columns=X.columns).assign(target=test_y).to_csv("data/processed/test.csv", index=False)

# Kiểm tra phân bố nhãn
print("Train label distribution:")
print(train_y.value_counts())
print("Test label distribution:")
print(test_y.value_counts())
