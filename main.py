import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve

df = pd.read_csv("Food_Delivery_Time_Prediction.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df = df.drop(columns=["Order_ID", "Customer_Location", "Restaurant_Location"])

df = pd.get_dummies(df, drop_first=True)

scaler = StandardScaler()

numeric_cols = [
    "Distance",
    "Delivery_Person_Experience",
    "Restaurant_Rating",
    "Customer_Rating",
    "Order_Cost",
    "Tip_Amount"
]

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nDescriptive Statistics:")
print(df.describe())

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.savefig("heatmap.png")
plt.close()

plt.figure()
sns.boxplot(x=df["Delivery_Time"])
plt.savefig("boxplot.png")
plt.close()

print("\nEDA Graphs saved")

X = df.drop("Delivery_Time", axis=1)
y = df["Delivery_Time"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("\nLinear Regression Results")
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

df["Delivery_Status"] = df["Delivery_Time"].apply(lambda x: 1 if x > 60 else 0)

X = df.drop(["Delivery_Time", "Delivery_Status"], axis=1)
y = df["Delivery_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig("confusion_matrix.png")
plt.close()

y_prob = log_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.savefig("roc_curve.png")
plt.close()

print("\nGraphs saved:")
print("heatmap.png")
print("boxplot.png")
print("confusion_matrix.png")
print("roc_curve.png")