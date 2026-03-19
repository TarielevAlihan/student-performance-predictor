import os, pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(__file__)



df = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})


FEATURES = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced"
]
TARGET = "Performance Index"

X = df[FEATURES]
y = df[TARGET]

# 3. РАЗБИВКА TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# 4. ОБУЧЕНИЕ
print("Обучаем LinearRegression...")
model = LinearRegression()
model.fit(X_train, y_train)

# 5. МЕТРИКИ
y_pred = model.predict(X_test)
mse  = mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
rmse = mse ** 0.5

print(f"\nМетрики:  R²={r2:.6f}  RMSE={rmse:.4f}")
print("Коэффициенты:")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"   {feat:<40} {coef:+.4f}")
print(f"   {'intercept':<40} {model.intercept_:+.4f}")

# 6. СОХРАНЕНИЕ МОДЕЛИ
payload = {
    "model": model, "features": FEATURES, "target": TARGET,
    "r2": r2, "rmse": rmse,
    "coef": dict(zip(FEATURES, model.coef_)),
    "intercept": float(model.intercept_),
}
model_path = os.path.join(BASE_DIR, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(payload, f)
print(f"\nМодель сохранена → {model_path}")

# 7. ТЕСТОВОЕ ПРЕДСКАЗАНИЕ
with open(model_path, "rb") as f:
    p = pickle.load(f)

sample = pd.DataFrame([{
    "Hours Studied": 7,
    "Previous Scores": 80,
    "Extracurricular Activities": 1,
    "Sleep Hours": 7,
    "Sample Question Papers Practiced": 3
}])[p["features"]]

pred = p["model"].predict(sample)[0]
print(f"Тест: total_score ≈ {pred:.1f}")
print("\nГотово! Запускай: python app.py")
