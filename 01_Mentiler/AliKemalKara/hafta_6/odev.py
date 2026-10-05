# -*- coding: utf-8 -*-
"""6. Hafta Odev: Dogrusal regresyon ile final notu tahmini."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parent.parent / "hafta_4" / "student_performance_dataset.csv"
studentDf = pd.read_csv(DATA_PATH)

# Tahmin edecegimiz surekli degisken.
target = studentDf["final_exam_score"]

# 1. Basit dogrusal regresyon: yalnizca calisma suresi kullaniliyor.
simpleFeatures = studentDf[["study_time_hours"]]
simpleXTrain, simpleXTest, simpleYTrain, simpleYTest = train_test_split(
    simpleFeatures, target, test_size=0.2, random_state=42
)

simpleModel = LinearRegression()
simpleModel.fit(simpleXTrain, simpleYTrain)
simplePrediction = simpleModel.predict(simpleXTest)

# 2. Coklu dogrusal regresyon: birden fazla ozellik birlikte kullaniliyor.
multipleFeatureNames = [
    "study_time_hours",
    "attendance_percent",
    "previous_grade",
    "sleep_hours",
]
multipleFeatures = studentDf[multipleFeatureNames]
multipleXTrain, multipleXTest, multipleYTrain, multipleYTest = train_test_split(
    multipleFeatures, target, test_size=0.2, random_state=42
)

multipleModel = LinearRegression()
multipleModel.fit(multipleXTrain, multipleYTrain)
multiplePrediction = multipleModel.predict(multipleXTest)


def printModelResults(modelName, actualValues, predictedValues):
    mae = mean_absolute_error(actualValues, predictedValues)
    mse = mean_squared_error(actualValues, predictedValues)
    r2 = r2_score(actualValues, predictedValues)

    print(f"\n--- {modelName} ---")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2: {r2:.3f}")

    return r2


print("--- Veri Seti ---")
print(f"Toplam ogrenci sayisi: {len(studentDf)}")
print(f"Tahmin hedefi: final_exam_score")

simpleR2 = printModelResults(
    "Basit Dogrusal Regresyon", simpleYTest, simplePrediction
)
multipleR2 = printModelResults(
    "Coklu Dogrusal Regresyon", multipleYTest, multiplePrediction
)

print("\n--- Model Parametreleri ---")
print(f"Basit model katsayisi: {simpleModel.coef_[0]:.3f}")
print(f"Basit model sabiti: {simpleModel.intercept_:.3f}")
print("Coklu model katsayilari:")
for featureName, coefficient in zip(multipleFeatureNames, multipleModel.coef_):
    print(f"  {featureName}: {coefficient:.3f}")
print(f"Coklu model sabiti: {multipleModel.intercept_:.3f}")

print("\n--- Sonuc ---")
if multipleR2 > simpleR2:
    print("Coklu regresyon modeli, test verisinde daha iyi tahmin basarisi gostermistir.")
else:
    print("Basit regresyon modeli, test verisinde daha iyi tahmin basarisi gostermistir.")
print("R2 degeri 1'e yaklastikca modelin aciklama gucu artar.")
