"""
train_model.py
--------------
Generates a synthetic dataset (600 rows), trains a Pipeline
(StandardScaler + RandomForestRegressor), prints the R2 score,
and saves the model (model.pkl) and the dataset (dataset.csv).
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42
N_ROWS = 600

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
EVENTS = ["Regular", "Festival", "Exam", "Holiday", "Special Event"]
WEATHER = ["Sunny", "Rainy", "Cloudy"]

NUMERIC_FEATURES = ["food_prepared", "expected_attendance", "actual_attendance", "attendance_ratio"]
CATEGORICAL_FEATURES = ["day_of_week", "event_type", "weather"]


def generate_dataset(n_rows: int = N_ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a realistic synthetic food-waste dataset."""
    rng = np.random.default_rng(seed)

    day = rng.choice(DAYS, n_rows)
    event = rng.choice(EVENTS, n_rows, p=[0.55, 0.12, 0.13, 0.10, 0.10])
    weather = rng.choice(WEATHER, n_rows, p=[0.45, 0.25, 0.30])

    expected = rng.integers(40, 1200, n_rows)

    # Actual attendance as a fraction of expected attendance
    ratio = rng.normal(0.88, 0.14, n_rows)
    ratio = np.clip(ratio, 0.40, 1.10)

    # Rain, holidays and exams reduce attendance a little more
    ratio = ratio - np.where(weather == "Rainy", 0.05, 0.0)
    ratio = ratio - np.where(event == "Holiday", 0.08, 0.0)
    ratio = ratio - np.where(event == "Exam", 0.03, 0.0)
    ratio = ratio + np.where(event == "Festival", 0.04, 0.0)
    ratio = np.clip(ratio, 0.35, 1.10)

    actual = np.round(expected * ratio).astype(int)
    actual = np.maximum(actual, 1)

    # Kitchen prepares ~0.4 kg per expected person, with some over/under-preparation
    prepared = expected * 0.4 * rng.uniform(0.90, 1.35, n_rows)
    prepared = np.clip(prepared, 10, 1000)

    attendance_ratio = actual / expected

    # ---- Waste rate (fraction of prepared food wasted) ----
    rate = np.full(n_rows, 0.05)

    event_effect = {"Regular": 0.00, "Festival": 0.03, "Exam": 0.02, "Holiday": 0.04, "Special Event": 0.02}
    rate += np.array([event_effect[e] for e in event])

    weather_effect = {"Sunny": 0.00, "Rainy": 0.03, "Cloudy": 0.01}
    rate += np.array([weather_effect[w] for w in weather])

    rate += np.where(np.isin(day, ["Saturday", "Sunday"]), 0.02, 0.0)

    # NON-LINEAR SPIKE: when attendance drops below 80%, waste jumps sharply
    shortfall = np.maximum(0.0, 0.80 - attendance_ratio)
    rate += shortfall * 1.2 + np.where(attendance_ratio < 0.80, 0.06, 0.0)

    # Over-preparation compared to what was really needed
    needed = actual * 0.4
    overprep = np.maximum(0.0, prepared - needed) / prepared
    rate += overprep * 0.35

    rate += rng.normal(0, 0.012, n_rows)
    rate = np.clip(rate, 0.0, 0.90)

    waste = prepared * rate
    waste = np.clip(waste, 0.0, prepared)

    return pd.DataFrame(
        {
            "food_prepared": np.round(prepared, 2),
            "expected_attendance": expected,
            "actual_attendance": actual,
            "day_of_week": day,
            "event_type": event,
            "weather": weather,
            "food_waste": np.round(waste, 2),
        }
    )


def add_attendance_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Derived feature: actual / expected attendance."""
    df = df.copy()
    df["attendance_ratio"] = df["actual_attendance"] / df["expected_attendance"].clip(lower=1)
    return df


def build_pipeline() -> Pipeline:
    """
    Pipeline = preprocessing (StandardScaler + OneHotEncoder) -> RandomForestRegressor.

    WHY RANDOM FOREST (and NOT Linear Regression)?
    Food waste is non-linear. When attendance drops below 80%, waste suddenly
    spikes. Linear Regression cannot capture this because it fits one straight
    line through the data. Random Forest uses many decision trees that split the
    data on thresholds (e.g. attendance_ratio < 0.8), so it naturally learns
    such sudden jumps and interactions between features (event, weather, day).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=14,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    # 1. Generate and save dataset
    df = generate_dataset()
    dataset_path = os.path.join(BASE_DIR, "dataset.csv")
    df.to_csv(dataset_path, index=False)
    print(f"Dataset generated: {df.shape[0]} rows, saved to dataset.csv")

    # 2. Features / target
    data = add_attendance_ratio(df)
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data["food_waste"]

    # 3. Split FIRST (before scaling) -> prevents data leakage.
    #    The scaler lives inside the Pipeline, so it is fitted on training data only.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 4. Train
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on unseen test data
    preds = pipeline.predict(X_test)
    print(f"R2 Score (test set): {r2_score(y_test, preds):.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, preds):.2f} Kg")

    # 6. Save model
    model_path = os.path.join(BASE_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print("Model saved as model.pkl")


if __name__ == "__main__":
    main()
