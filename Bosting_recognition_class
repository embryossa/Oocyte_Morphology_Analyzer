import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt

# Загрузка данных
abnormal = pd.read_csv("C:/Users/User/PycharmProjects/pythonProject/Imaje recognition/Imaje recognition/Blastocyst/comparative_results/abnormal_morphometrics.csv")
normal = pd.read_csv("C:/Users/User/PycharmProjects/pythonProject/Imaje recognition/Imaje recognition/Blastocyst/comparative_results/normal_morphometrics.csv")

# Объединение данных
abnormal["label"] = 1
normal["label"] = 0
data = pd.concat([abnormal, normal], axis=0)

# Удаление ненужных колонок
data = data.drop(columns=["filename"])

# Разделение на признаки и целевую переменную
X = data.drop(columns=["class", "label"])
y = data["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Обучение модели LightGBM
model = lgb.LGBMClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Прогнозы
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Вероятности положительного класса

# Оценка
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# PRC (Precision-Recall Curve)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"Avg Precision = {avg_precision:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

# Важность признаков
lgb.plot_importance(model, figsize=(10, 6))
plt.show()

# Сохранение модели
model.booster_.save_model("oocyte_classifier.txt")
