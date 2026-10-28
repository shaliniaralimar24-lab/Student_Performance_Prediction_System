"""Generate a synthetic student performance dataset for the project."""
import os

import numpy as np
import pandas as pd

np.random.seed(42)
n = 600

genders = np.random.choice(["Male", "Female"], n, p=[0.52, 0.48])
ages = np.random.randint(16, 22, n)
study_hours = np.clip(np.random.normal(15, 6, n), 1, 40)
attendance = np.clip(np.random.normal(78, 12, n), 40, 100)
previous_score = np.clip(np.random.normal(65, 15, n), 0, 100)
sleep_hours = np.clip(np.random.normal(7, 1.5, n), 3, 12)
assignments = np.clip(np.random.normal(7, 2, n), 0, 10).round().astype(int)
extracurricular = np.random.choice(["Yes", "No"], n, p=[0.4, 0.6])
parental_edu = np.random.choice(
    ["High School", "Bachelor", "Master", "PhD"], n, p=[0.35, 0.35, 0.2, 0.1]
)
internet = np.random.choice(["Yes", "No"], n, p=[0.82, 0.18])
health = np.random.choice(["Poor", "Fair", "Good", "Excellent"], n, p=[0.1, 0.25, 0.45, 0.2])

edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
health_map = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}

final_score = (
    0.05 * study_hours
    + 0.35 * attendance
    + 0.30 * previous_score
    + 2.0 * sleep_hours
    + 1.5 * assignments
    + 3 * np.array([edu_map[x] for x in parental_edu])
    + 2 * np.array([health_map[x] for x in health])
    + 4 * np.where(internet == "Yes", 1, 0)
    + 2 * np.where(extracurricular == "Yes", 1, 0)
)
# scale to roughly 30-90, then add noise so the data is not perfectly predictable
final_score = (final_score - final_score.min()) / (final_score.max() - final_score.min()) * 60 + 30
final_score = final_score + np.random.normal(0, 5, n)
final_score = np.clip(final_score, 0, 100).round(2)

df = pd.DataFrame({
    "student_id": range(1, n + 1),
    "gender": genders,
    "age": ages,
    "study_hours_per_week": study_hours.round(2),
    "attendance_percentage": attendance.round(2),
    "previous_score": previous_score.round(2),
    "sleep_hours": sleep_hours.round(2),
    "assignments_completed": assignments,
    "extracurricular": extracurricular,
    "parental_education": parental_edu,
    "internet_access": internet,
    "health_status": health,
    "final_score": final_score,
})
df["pass_fail"] = (df["final_score"] >= 50).astype(int)

# Inject missing values (~5%) in 3 columns so there is something to clean
for col in ["study_hours_per_week", "attendance_percentage", "parental_education"]:
    mask = np.random.rand(n) < 0.05
    df.loc[mask, col] = np.nan

# Paths are relative to the project folder, so this runs on any computer
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(BASE, "data", "student_data.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)

print("Saved", out, "shape:", df.shape)
print(df.head())
print(df.isna().sum())
