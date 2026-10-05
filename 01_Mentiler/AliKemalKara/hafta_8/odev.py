# -*- coding: utf-8 -*-
"""8. Hafta Odev: Lojistik regresyon ve K-NN ile basari siniflandirmasi."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parent.parent / "hafta_4" / "student_performance_dataset.csv"
studentDf = pd.read_csv(DATA_PATH)

featureNames = [
    "study_time_hours",
    "attendance_percent",
    "previous_grade",
    "sleep_hours",
]
features = studentDf[featureNames]

# 85 ve uzeri final notunu basarili, altini basarisiz kabul ediyoruz.
studentDf["passed"] = (studentDf["final_exam_score"] >= 85).astype(int)
target = studentDf["passed"]

xTrain, xTest, yTrain, yTest = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target,
)

# Olcekleme, iki modelin de farkli birimlerdeki ozelliklerle daha adil calismasini saglar.
logisticModel = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
    ]
)
logisticModel.fit(xTrain, yTrain)
logisticPrediction = logisticModel.predict(xTest)

knnModel = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=7)),
    ]
)
knnModel.fit(xTrain, yTrain)
knnPrediction = knnModel.predict(xTest)


def printModelResults(modelName, actualValues, predictedValues):
    accuracy = accuracy_score(actualValues, predictedValues)
    precision = precision_score(actualValues, predictedValues, zero_division=0)
    recall = recall_score(actualValues, predictedValues, zero_division=0)
    matrix = confusion_matrix(actualValues, predictedValues, labels=[0, 1])

    print(f"\n--- {modelName} ---")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print("Confusion Matrix (satir: gercek, sutun: tahmin):")
    print(matrix)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
    }


print("--- Veri Seti ---")
print(f"Toplam ogrenci sayisi: {len(studentDf)}")
print(f"Kullanilan ozellikler: {', '.join(featureNames)}")
print(f"Basarili ogrenci sayisi: {target.sum()}")
print(f"Basarisiz ogrenci sayisi: {(target == 0).sum()}")

logisticResults = printModelResults(
    "Lojistik Regresyon", yTest, logisticPrediction
)
knnResults = printModelResults("K-NN", yTest, knnPrediction)

print("\n--- Model Karsilastirmasi ---")
print("                 Accuracy  Precision  Recall")
print(
    f"Lojistik Reg.    {logisticResults['Accuracy']:.3f}      "
    f"{logisticResults['Precision']:.3f}      {logisticResults['Recall']:.3f}"
)
print(
    f"K-NN             {knnResults['Accuracy']:.3f}      "
    f"{knnResults['Precision']:.3f}      {knnResults['Recall']:.3f}"
)

if logisticResults["Accuracy"] > knnResults["Accuracy"]:
    print("Accuracy degerine gore lojistik regresyon daha basarilidir.")
elif knnResults["Accuracy"] > logisticResults["Accuracy"]:
    print("Accuracy degerine gore K-NN daha basarilidir.")
else:
    print("Iki model ayni Accuracy degerine sahiptir.")
