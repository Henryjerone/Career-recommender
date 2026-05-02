"""
train_model.py — AI Career Recommendation System
Step 1: ML Model Training (KNN + Decision Tree + Random Forest)

Run this once to generate model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ─────────────────────────────────────────────
# 1. SAMPLE DATASET (replace with your dataset.csv)
# ─────────────────────────────────────────────

data = {
    "cgpa": [9.2, 8.5, 7.8, 6.5, 9.0, 7.2, 8.0, 6.8, 9.5, 7.5,
             8.8, 6.2, 7.0, 9.1, 8.3, 6.9, 7.7, 8.6, 9.3, 7.1],
    "python_skill": [1, 1, 1, 0, 1, 0, 1, 0, 1, 1,
                     1, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    "ml_skill":    [1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
                     1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    "sql_skill":   [0, 1, 1, 1, 0, 1, 0, 1, 0, 1,
                     0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    "web_skill":   [0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
                     0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    "interest":    [0, 0, 1, 1, 0, 1, 0, 2, 0, 0,
                     0, 2, 1, 0, 0, 1, 1, 0, 0, 2],
    # 0 = ML/AI, 1 = Web Dev, 2 = Data Analysis
    "internship":  ["ML Engineer", "ML Engineer", "Web Developer", "Web Developer",
                    "ML Engineer", "Web Developer", "ML Engineer", "Data Analyst",
                    "ML Engineer", "ML Engineer", "ML Engineer", "Data Analyst",
                    "Web Developer", "ML Engineer", "ML Engineer", "Web Developer",
                    "Web Developer", "ML Engineer", "ML Engineer", "Data Analyst"],
    "project":     ["NLP System", "Image Classifier", "E-Commerce Site", "Portfolio Website",
                    "Chatbot", "Blog Platform", "Sentiment Analyzer", "Sales Dashboard",
                    "Object Detection", "Recommender System", "Fraud Detection", "HR Dashboard",
                    "Social Media App", "Text Summarizer", "Spam Filter", "Inventory Tracker",
                    "Task Manager", "Stock Predictor", "Face Recognition", "Financial Report"]
}

# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────

if os.path.exists("dataset.csv"):
    print("✅ Loading from dataset.csv")
    df = pd.read_csv("dataset.csv")
else:
    print("⚠️  dataset.csv not found — using built-in sample data")
    df = pd.DataFrame(data)

print(f"\n📊 Dataset shape: {df.shape}")
print(df.head())

# ─────────────────────────────────────────────
# 3. ENCODE LABELS
# ─────────────────────────────────────────────

le_internship = LabelEncoder()
le_project    = LabelEncoder()

df["internship_encoded"] = le_internship.fit_transform(df["internship"])
df["project_encoded"]    = le_project.fit_transform(df["project"])

# ─────────────────────────────────────────────
# 4. FEATURES & TARGETS
# ─────────────────────────────────────────────

FEATURES = ["cgpa", "python_skill", "ml_skill", "sql_skill", "web_skill", "interest"]

X = df[FEATURES]
y_internship = df["internship_encoded"]
y_project    = df["project_encoded"]

# ─────────────────────────────────────────────
# 5. SCALE FEATURES
# ─────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# 6. TRAIN/TEST SPLIT
# ─────────────────────────────────────────────

X_train, X_test, yi_train, yi_test = train_test_split(
    X_scaled, y_internship, test_size=0.2, random_state=42
)
_, _, yp_train, yp_test = train_test_split(
    X_scaled, y_project, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 7. TRAIN MODELS
# ─────────────────────────────────────────────

models = {
    "KNN":           KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
}

print("\n🚀 Training models...\n")
results = {}

for name, model in models.items():
    # Train on internship target
    model.fit(X_train, yi_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(yi_test, y_pred)
    results[name] = acc
    print(f"  {name:20s} → Accuracy: {acc:.2%}")

# ─────────────────────────────────────────────
# 8. PICK BEST MODEL
# ─────────────────────────────────────────────

best_name = max(results, key=results.get)
best_model = models[best_name]

# Retrain best model on full data for production
best_model.fit(X_scaled, y_internship)

# Also train project model (Random Forest default)
rf_project = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_project.fit(X_scaled, y_project)

print(f"\n🏆 Best model: {best_name} ({results[best_name]:.2%})")
print(f"\n📋 Accuracy comparison:")
for name, acc in results.items():
    bar = "█" * int(acc * 20)
    print(f"  {name:20s} {bar} {acc:.2%}")

# ─────────────────────────────────────────────
# 9. SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────────

bundle = {
    "internship_model": best_model,
    "project_model":    rf_project,
    "scaler":           scaler,
    "le_internship":    le_internship,
    "le_project":       le_project,
    "features":         FEATURES,
    "model_name":       best_name,
    "accuracies":       results,
}

with open("model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\n✅ model.pkl saved successfully!")
print("   → Contains: internship model, project model, scaler, label encoders")
print("\nNext step: Run resume_parser.py or app.py")