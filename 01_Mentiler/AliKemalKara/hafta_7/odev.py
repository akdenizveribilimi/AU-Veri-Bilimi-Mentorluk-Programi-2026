# -*- coding: utf-8 -*-
"""7. Hafta Odev: Karar agaci ve rassal orman ile not tahmini."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


DATA_PATH = Path(__file__).resolve().parent.parent / "hafta_4" / "student_performance_dataset.csv"
studentDf = pd.read_csv(DATA_PATH)

featureNames = [
    "study_time_hours",
    "attendance_percent",
    "previous_grade",
    "sleep_hours",
]
features = studentDf[featureNames]
target = studentDf["final_exam_score"]

xTrain, xTest, yTrain, yTest = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# 1. Karar agaci modeli.
decisionTreeModel = DecisionTreeRegressor(max_depth=5, random_state=42)
decisionTreeModel.fit(xTrain, yTrain)
decisionTreePrediction = decisionTreeModel.predict(xTest)

# 2. Rassal orman modeli.
randomForestModel = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)
randomForestModel.fit(xTrain, yTrain)
randomForestPrediction = randomForestModel.predict(xTest)


def printModelResults(modelName, actualValues, predictedValues):
    mae = mean_absolute_error(actualValues, predictedValues)
    mse = mean_squared_error(actualValues, predictedValues)
    r2 = r2_score(actualValues, predictedValues)

    print(f"\n--- {modelName} ---")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2: {r2:.3f}")

    return {"MAE": mae, "MSE": mse, "R2": r2}


print("--- Veri Seti ---")
print(f"Toplam ogrenci sayisi: {len(studentDf)}")
print(f"Tahmin hedefi: final_exam_score")
print(f"Kullanilan ozellikler: {', '.join(featureNames)}")

decisionTreeResults = printModelResults(
    "Karar Agaci Regresyonu", yTest, decisionTreePrediction
)
randomForestResults = printModelResults(
    "Rassal Orman Regresyonu", yTest, randomForestPrediction
)

print("\n--- Ozellik Onemleri ---")
for featureName, importance in zip(featureNames, randomForestModel.feature_importances_):
    print(f"{featureName}: {importance:.3f}")

print("\n--- Model Karsilastirmasi ---")
print("                 MAE      MSE       R2")
print(
    f"Karar Agaci      {decisionTreeResults['MAE']:.2f}    "
    f"{decisionTreeResults['MSE']:.2f}    {decisionTreeResults['R2']:.3f}"
)
print(
    f"Rassal Orman     {randomForestResults['MAE']:.2f}    "
    f"{randomForestResults['MSE']:.2f}    {randomForestResults['R2']:.3f}"
)

if randomForestResults["R2"] > decisionTreeResults["R2"]:
    print("Rassal orman modeli, test verisinde daha yuksek R2 basarisi gostermistir.")
elif decisionTreeResults["R2"] > randomForestResults["R2"]:
    print("Karar agaci modeli, test verisinde daha yuksek R2 basarisi gostermistir.")
else:
    print("Iki model test verisinde ayni R2 basarisini gostermistir.")
print("R2 degeri 1'e yaklastikca modelin aciklama gucu artar.")
