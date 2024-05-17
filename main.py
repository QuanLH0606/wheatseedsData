import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Đọc dữ liệu và gán tên cột
column_names = [
    'Area', 'Perimeter', 'Compactness', 'Length_of_kernel', 
    'Width_of_kernel', 'Asymmetry_coefficient', 'Length_of_kernel_groove', 'Class'
]
data = pd.read_csv('wheat-seeds.csv', header=None, names=column_names)

# Khám phá dữ liệu cơ bản
print("Kích thước của tập dữ liệu:", data.shape)
print("\nKiểu dữ liệu của các cột:\n", data.dtypes)
print("\nXem trước 5 hàng đầu tiên của tập dữ liệu:\n", data.head())
print("\nThông tin tổng quan về tập dữ liệu:\n", data.info())
print("\nThống kê mô tả của tập dữ liệu:\n", data.describe())

# Kiểm tra giá trị thiếu
print("\nSố lượng giá trị thiếu trong mỗi cột:\n", data.isnull().sum())

# Xử lý giá trị thiếu - Thay thế bằng giá trị trung bình của cột
data.fillna(data.mean(), inplace=True)

# Kiểm tra dữ liệu trùng lặp
duplicate_rows = data.duplicated().sum()
print("Số lượng hàng trùng lặp:", duplicate_rows)

# Loại bỏ dữ liệu trùng lặp
data = data.drop_duplicates()
print("Kích thước của tập dữ liệu sau khi loại bỏ trùng lặp:", data.shape)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop('Class', axis=1))
data_scaled = pd.DataFrame(data_scaled, columns=data.columns[:-1])

# Thêm lại cột nhãn vào dữ liệu đã chuẩn hóa
data_scaled['Class'] = data['Class'].values

print("Dữ liệu đã được chuẩn hóa:\n", data_scaled.head())

# Xử lý outliers
z_scores = np.abs(stats.zscore(data_scaled.drop('Class', axis=1)))
outliers = (z_scores > 3).any(axis=1)

# Loại bỏ các hàng có outliers
data_cleaned = data_scaled[~outliers]
print("Kích thước của tập dữ liệu sau khi loại bỏ outliers:", data_cleaned.shape)

# Tách tập dữ liệu thành tập huấn luyện và tập kiểm tra
X = data_cleaned.drop('Class', axis=1)
y = data_cleaned['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Kích thước của tập huấn luyện:", X_train.shape)
print("Kích thước của tập kiểm tra:", X_test.shape)

# Tạo và huấn luyện mô hình Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_logreg = logreg.predict(X_test)

# Đánh giá mô hình
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("Độ chính xác của Logistic Regression:", accuracy_logreg)
print("Báo cáo phân loại của Logistic Regression:\n", classification_report(y_test, y_pred_logreg))
print("Ma trận nhầm lẫn của Logistic Regression:\n", confusion_matrix(y_test, y_pred_logreg))

# Tạo và huấn luyện mô hình Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_tree = tree.predict(X_test)

# Đánh giá mô hình
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print("Độ chính xác của Decision Tree:", accuracy_tree)
print("Báo cáo phân loại của Decision Tree:\n", classification_report(y_test, y_pred_tree))
print("Ma trận nhầm lẫn của Decision Tree:\n", confusion_matrix(y_test, y_pred_tree))
