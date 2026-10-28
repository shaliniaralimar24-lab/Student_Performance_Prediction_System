"""Predict a student's final score and Pass/Fail result."""
import os

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "student_performance_model.pkl")


def predict_one(values: dict):
    model = joblib.load(MODEL_PATH)
    score = float(model.predict(pd.DataFrame([values]))[0])
    score = max(0.0, min(100.0, score))
    return score, "Pass" if score >= 50 else "Fail"


def main():
    print("Student Performance Prediction System")
    values = {
        "gender": input("Gender (Male/Female): ").strip(),
        "age": float(input("Age: ")),
        "study_hours_per_week": float(input("Study hours per week: ")),
        "attendance_percentage": float(input("Attendance percentage: ")),
        "previous_score": float(input("Previous score (0-100): ")),
        "sleep_hours": float(input("Sleep hours per day: ")),
        "assignments_completed": int(input("Assignments completed (0-10): ")),
        "extracurricular": input("Extracurricular (Yes/No): ").strip(),
        "parental_education": input("Parental education (High School/Bachelor/Master/PhD): ").strip(),
        "internet_access": input("Internet access (Yes/No): ").strip(),
        "health_status": input("Health status (Poor/Fair/Good/Excellent): ").strip(),
    }
    score, status = predict_one(values)
    print(f"\nPredicted Final Score: {score:.2f}")
    print(f"Predicted Result: {status}")


if __name__ == "__main__":
    main()
