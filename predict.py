import pickle
import pandas as pd

# 1. ЗАГРУЗКА МОДЕЛИ
print("Загружаем модель...")
with open("model.pkl", "rb") as f:
    p = pickle.load(f)

model    = p["model"]
features = p["features"]

print(f"Модель загружена!")
print(f"   R²: {p['r2']:.4f}  |  RMSE: {p['rmse']:.2f}")
print()

# 2. ВВОД ДАННЫХ
print("Введите данные студента:")

while True:
    try:
        hours = float(input("   Часы учёбы в день (1–9): "))
        if 1 <= hours <= 9:
            break
        print("Введите число от 1 до 9")
    except ValueError:
        print("Введите число!")

while True:
    try:
        prev = float(input("Предыдущие оценки (40–99): "))
        if 40 <= prev <= 99:
            break
        print("Введите число от 40 до 99")
    except ValueError:
        print("Введите число!")

while True:
    extra = input("Внеклассные занятия? (yes/no): ").strip().lower()
    if extra in ("yes", "no", "да", "нет"):
        extra_val = 1 if extra in ("yes", "да") else 0
        break
    print("Введите yes или no")

while True:
    try:
        sleep = float(input("Часов сна (1–9): "))
        if 1 <= sleep <= 9:
            break
        print("Введите число от 1 до 9")
    except ValueError:
        print("Введите число!")

while True:
    try:
        papers = float(input("Решено пробных тестов (0–9): "))
        if 0 <= papers <= 9:
            break
        print("Введите число от 0 до 9")
    except ValueError:
        print("Введите число!")

# 3. ПРЕДСКАЗАНИЕ
sample = pd.DataFrame([{
    "Hours Studied":                      hours,
    "Previous Scores":                    prev,
    "Extracurricular Activities":         extra_val,
    "Sleep Hours":                        sleep,
    "Sample Question Papers Practiced":   papers
}])[features]

score = float(model.predict(sample)[0])
score = max(0, min(100, score))

# 4. ОЦЕНКА
if   score >= 90: grade = "A"
elif score >= 75: grade = "B"
elif score >= 60: grade = "C"
elif score >= 40: grade = "D"
else:             grade = "F"

# 5. ВЫВОД
print()
print("─" * 40)
print(f"Результат предсказания:")
print(f"Performance Index: {score:.1f} / 100")
print(f"Оценка:            {grade}")
print("─" * 40)
