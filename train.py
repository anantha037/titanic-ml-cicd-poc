import pandas as pd
import joblib
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# 1. MODEL VERSION
MODEL_VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = f"{MODEL_DIR}/model_{MODEL_VERSION}.joblib"

# 2. LOAD DATA

df = pd.read_csv("data/train_and_test.csv")

# 3. Preprocessing

df.rename(columns = {"2urvived":"Survived","sibsp":"Sibsp"},inplace=True)
# Select useful columns
df = df[[
    "Pclass",
    "Sex",
    "Age",
    "Sibsp",
    "Parch",
    "Fare",
    "Embarked",
    "Survived"
]]

# Handling missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(0)


# 4. SPLIT FEATURES & TARGET
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. TRAIN MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# 6. SAVE MODEL
joblib.dump(model, MODEL_PATH)

print(f"Model saved at {MODEL_PATH}")