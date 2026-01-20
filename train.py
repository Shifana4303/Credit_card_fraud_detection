import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

print("Loading dataset...")
df = pd.read_csv("dataset.csv")
print("Dataset Loaded:", df.shape)

df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["trans_hour"] = df["trans_date_trans_time"].dt.hour
df["trans_day"] = df["trans_date_trans_time"].dt.day
df["trans_month"] = df["trans_date_trans_time"].dt.month


FEATURES = [
    "amt",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "trans_hour",
    "trans_day",
    "trans_month",
    "gender",
]

df = df[FEATURES + ["is_fraud"]]
print("Using Columns:", df.columns.tolist())
df["gender"] = df["gender"].map({"F": 1, "M": 0})


X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


print("Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
print("After SMOTE:", X_train_resampled.shape, y_train_resampled.shape)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)


print("Training Logistic Regression...")
model = LogisticRegression(
    max_iter=500,
    class_weight="balanced",
    solver="lbfgs"
)

model.fit(X_train_scaled, y_train_resampled)


print("\nEvaluating Model...")

train_pred = model.predict(X_train_scaled)
train_acc = accuracy_score(y_train_resampled, train_pred)


test_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_pred)

print("\n==================== ACCURACY ====================")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy : {test_acc:.4f}")

print("\n================ CLASSIFICATION REPORT =================")
print(classification_report(y_test, test_pred))

print("\n================ CONFUSION MATRIX =================")
print(confusion_matrix(y_test, test_pred))


pickle.dump(model, open("logistic_regression_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nTraining Completed Successfully!")
print("Saved: logistic_regression_model.pkl & scaler.pkl")
