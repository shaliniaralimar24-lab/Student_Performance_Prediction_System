"""Training pipeline for the Student Performance Prediction System."""
import json
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "student_data.csv")
VIZ = os.path.join(BASE, "visualizations")
MODELS = os.path.join(BASE, "models")
REPORTS = os.path.join(BASE, "reports")
for folder in (VIZ, MODELS, REPORTS):
    os.makedirs(folder, exist_ok=True)

TARGET = "final_score"
sns.set_theme(style="whitegrid")


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ, name), dpi=120)
    plt.close()


# ---------- 1. Load ----------
raw = pd.read_csv(DATA)
print("Raw shape:", raw.shape)

# ---------- 2. Clean ----------
df = raw.drop_duplicates()
df = df[(df[TARGET] >= 0) & (df[TARGET] <= 100)].copy()

numeric_cols = ["study_hours_per_week", "attendance_percentage", "previous_score",
                "sleep_hours", "assignments_completed", "age"]
categorical_cols = ["gender", "extracurricular", "parental_education",
                    "internet_access", "health_status"]
for c in numeric_cols:
    df[c] = df[c].fillna(df[c].median())
for c in categorical_cols:
    df[c] = df[c].fillna(df[c].mode()[0])
df.to_csv(os.path.join(BASE, "data", "student_data_cleaned.csv"), index=False)

# ---------- 3. EDA charts ----------
# 1. Missing values (raw data)
missing = raw.isna().sum()
missing = missing[missing > 0]
plt.figure(figsize=(8, 5))
missing.sort_values().plot(kind="barh", color="salmon")
plt.title("Missing Values by Column (Raw Data)")
plt.xlabel("Count")
save_fig("01_missing_values.png")

# 2. Target distribution
plt.figure(figsize=(8, 5))
sns.histplot(df[TARGET], kde=True, bins=30, color="steelblue")
plt.title("Distribution of Final Score")
plt.xlabel("Final Score")
save_fig("02_final_score_distribution.png")

# 3. Correlation heatmap
plt.figure(figsize=(10, 8))
num_df = df.select_dtypes(include=[np.number]).drop(columns=["student_id"])
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap (Numeric Features)")
save_fig("03_correlation_heatmap.png")

# 4. Scatter plot: the EDA example
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="study_hours_per_week", y=TARGET,
                hue="pass_fail", palette={0: "tomato", 1: "seagreen"}, alpha=0.6)
plt.title("Study Hours vs Final Score (0 = Fail, 1 = Pass)")
save_fig("04_study_hours_vs_score.png")

# 5. Score by parental education
order = ["High School", "Bachelor", "Master", "PhD"]
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="parental_education", y=TARGET, order=order,
            hue="parental_education", hue_order=order, palette="Set2", legend=False)
plt.title("Final Score by Parental Education")
save_fig("05_score_by_parental_education.png")

# ---------- 4. Features, split, preprocessing ----------
X = df.drop(columns=["student_id", TARGET, "pass_fail"])
y = df[TARGET]
num_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

preprocess = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 5. Train and evaluate ----------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

results, trained = {}, {}
for name, model in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    results[name] = {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "MAE": round(float(mean_absolute_error(y_test, preds)), 4),
        "R2": round(float(r2_score(y_test, preds)), 4),
    }
    trained[name] = pipe
    print(name, results[name])

res_df = pd.DataFrame(results).T

# 6. Model comparison
plt.figure(figsize=(8, 5))
res_df["RMSE"].plot(kind="bar", color=["cornflowerblue", "orange"])
plt.title("Model RMSE Comparison (lower is better)")
plt.ylabel("RMSE")
plt.xticks(rotation=0)
save_fig("06_model_rmse_comparison.png")

best_name = min(results, key=lambda k: results[k]["RMSE"])
best_model = trained[best_name]
print("Best model:", best_name)

# ---------- 6. Best model charts ----------
preds = best_model.predict(X_test)

# 7. Actual vs predicted
plt.figure(figsize=(7, 7))
plt.scatter(y_test, preds, alpha=0.6, color="darkorange")
mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
plt.plot([mn, mx], [mn, mx], "r--", label="Ideal")
plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title(f"Actual vs Predicted ({best_name})")
plt.legend()
save_fig("07_actual_vs_predicted.png")

# 8. Feature importance
model_step = best_model.named_steps["model"]
fnames = [n.split("__", 1)[1] for n in best_model.named_steps["preprocess"].get_feature_names_out()]
if hasattr(model_step, "feature_importances_"):
    imp = pd.Series(model_step.feature_importances_, index=fnames)
    label = "importance"
else:
    imp = pd.Series(np.abs(model_step.coef_), index=fnames)
    label = "absolute coefficient"
top = imp.sort_values(ascending=False).head(10)
plt.figure(figsize=(9, 6))
top.sort_values().plot(kind="barh", color="teal")
plt.title(f"Top 10 Features ({best_name}, {label})")
save_fig("08_feature_importance.png")

# ---------- 7. Pass/Fail result and saving ----------
pass_acc = float(accuracy_score((y_test >= 50).astype(int), (preds >= 50).astype(int)))
print("Pass/Fail accuracy:", round(pass_acc, 4))

joblib.dump(best_model, os.path.join(MODELS, "student_performance_model.pkl"))
with open(os.path.join(REPORTS, "metrics.json"), "w") as f:
    json.dump({"best_model": best_name, "results": results,
               "pass_fail_accuracy": round(pass_acc, 4)}, f, indent=2)

print("\nPipeline complete.")
print(res_df)
