import tkinter as tk
from tkinter import ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split  # Thêm thư viện chia dữ liệu
import pandas as pd

# Đọc dữ liệu từ tệp CSV vào một DataFrame
data = pd.read_csv("C:\\Users\\caodu\\OneDrive\\Máy tính\\archive\\spam.csv")

# Xác định các đặc trưng (features) và nhãn (labels)
X = data["Message"]  # Cột 'Message' chứa các tin nhắn
y = data["Category"]  # Cột 'Category' chứa các nhãn spam/ham

# Chia dữ liệu thành tập train và tập test (ví dụ: 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Sử dụng TF-IDF Vectorizer để chuyển đổi văn bản thành đặc trưng
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Khởi tạo mô hình Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_tfidf, y_train)

# Khởi tạo mô hình SVM
svm_model = SVC()
svm_model.fit(X_tfidf, y_train)

# Tính toán kết quả precision, recall, f1-score và accuracy cho tập test
X_test_tfidf = tfidf_vectorizer.transform(X_test)
y_pred_logistic = logistic_model.predict(X_test_tfidf)
y_pred_svm = svm_model.predict(X_test_tfidf)

report_logistic = classification_report(y_test, y_pred_logistic, output_dict=True)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = report_logistic["macro avg"]["precision"]
recall_logistic = report_logistic["macro avg"]["recall"]
f1_score_logistic = report_logistic["macro avg"]["f1-score"]

report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = report_svm["macro avg"]["precision"]
recall_svm = report_svm["macro avg"]["recall"]
f1_score_svm = report_svm["macro avg"]["f1-score"]

# Tạo giao diện đồ họa
window = tk.Tk()
window.title("Email Spam Classification")

frame = ttk.Frame(window)
frame.grid(row=0, column=0, padx=10, pady=10)

# Hiển thị thông số Accuracy, Precision, Recall, và F1-Score cho Logistic Regression
logistic_result_label = tk.Text(frame, height=10, width=40)
logistic_result_label.insert(tk.END, f"Logistic Regression\n")
logistic_result_label.insert(tk.END, f"Accuracy: {accuracy_logistic*100:.2f}%\n")
logistic_result_label.insert(tk.END, f"Precision: {precision_logistic*100:.2f}%\n")
logistic_result_label.insert(tk.END, f"Recall: {recall_logistic*100:.2f}%\n")
logistic_result_label.insert(tk.END, f"F1-Score: {f1_score_logistic*100:.2f}%\n")
logistic_result_label.grid(row=0, column=0, padx=5, pady=5)

# Hiển thị thông số Accuracy, Precision, Recall, và F1-Score cho SVM
svm_result_label = tk.Text(frame, height=10, width=40)
svm_result_label.insert(tk.END, f"SVM\n")
svm_result_label.insert(tk.END, f"Accuracy: {accuracy_svm*100:.2f}%\n")
svm_result_label.insert(tk.END, f"Precision: {precision_svm*100:.2f}%\n")
svm_result_label.insert(tk.END, f"Recall: {recall_svm*100:.2f}%\n")
svm_result_label.insert(tk.END, f"F1-Score: {f1_score_svm*100:.2f}%\n")
svm_result_label.grid(row=0, column=1, padx=5, pady=5)

email_label = ttk.Label(frame, text="Enter Email:")
email_label.grid(row=1, column=0, padx=5, pady=5)

# Sử dụng Text Area thay vì Entry
email_text_area = tk.Text(frame, height=10, width=40)
email_text_area.grid(row=1, column=1, padx=5, pady=5)

# Hàm dự đoán email sử dụng Logistic Regression
def predict_email_logistic():
    email_text = email_text_area.get(
        "1.0", "end-1c"
    )  # Lấy toàn bộ nội dung trong Text Area
    email_tfidf = tfidf_vectorizer.transform([email_text])
    prediction = logistic_model.predict(email_tfidf)

    logistic_result_label.config(state="normal")
    logistic_result_label.delete(1.0, tk.END)
    logistic_result_label.insert(tk.END, f"Logistic Regression\n")
    logistic_result_label.insert(tk.END, f"Accuracy: {accuracy_logistic*100:.2f}%\n")
    logistic_result_label.insert(tk.END, f"Precision: {precision_logistic*100:.2f}%\n")
    logistic_result_label.insert(tk.END, f"Recall: {recall_logistic*100:.2f}%\n")
    logistic_result_label.insert(tk.END, f"F1-Score: {f1_score_logistic*100:.2f}%\n")
    logistic_result_label.insert(tk.END, f"Prediction: {prediction[0]}\n")
    logistic_result_label.config(state="disabled")

# Hàm dự đoán email sử dụng SVM
def predict_email_svm():
    email_text = email_text_area.get(
        "1.0", "end-1c"
    )  # Lấy toàn bộ nội dung trong Text Area
    email_tfidf = tfidf_vectorizer.transform([email_text])
    prediction = svm_model.predict(email_tfidf)

    svm_result_label.config(state="normal")
    svm_result_label.delete(1.0, tk.END)
    svm_result_label.insert(tk.END, f"SVM\n")
    svm_result_label.insert(tk.END, f"Accuracy: {accuracy_svm*100:.2f}%\n")
    svm_result_label.insert(tk.END, f"Precision: {precision_svm*100:.2f}%\n")
    svm_result_label.insert(tk.END, f"Recall: {recall_svm*100:.2f}%\n")
    svm_result_label.insert(tk.END, f"F1-Score: {f1_score_svm*100:.2f}%\n")
    svm_result_label.insert(tk.END, f"Prediction: {prediction[0]}\n")
    svm_result_label.config(state="disabled")

predict_logistic_button = ttk.Button(frame, text="Predict using Logistic Regression", command=predict_email_logistic)
predict_logistic_button.grid(row=2, column=0, padx=5, pady=5)

predict_svm_button = ttk.Button(frame, text="Predict using SVM", command=predict_email_svm)
predict_svm_button.grid(row=2, column=1, padx=5, pady=5)

window.mainloop()