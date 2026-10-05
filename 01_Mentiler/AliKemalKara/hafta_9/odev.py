# -*- coding: utf-8 -*-
"""9. Hafta Odev: Phishing tespiti icin agac tabanli siniflandirma."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).resolve().parent / "phishing_websites_model.csv"
phishingDf = pd.read_csv(DATA_PATH)

# Result, label sutununun sayisal karsiligi oldugu icin modele alinmiyor.
featureNames = [column for column in phishingDf.columns if column not in ["Result", "label"]]
features = phishingDf[featureNames]
target = (phishingDf["label"] == "Phishing").astype(int)

xTrain, xTest, yTrain, yTest = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target,
)

# 1. Karar agaci siniflandiricisi.
decisionTreeModel = DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
)
decisionTreeModel.fit(xTrain, yTrain)
decisionTreePrediction = decisionTreeModel.predict(xTest)

# 2. Rassal orman siniflandiricisi.
randomForestModel = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
)
randomForestModel.fit(xTrain, yTrain)
randomForestPrediction = randomForestModel.predict(xTest)


def printModelResults(modelName, actualValues, predictedValues):
    accuracy = accuracy_score(actualValues, predictedValues)
    precision = precision_score(actualValues, predictedValues, zero_division=0)
    recall = recall_score(actualValues, predictedValues, zero_division=0)
    matrix = confusion_matrix(actualValues, predictedValues, labels=[0, 1])

    print(f"\n--- {modelName} ---")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision (phishing): {precision:.3f}")
    print(f"Recall (phishing): {recall:.3f}")
    print("Confusion Matrix (satir: gercek, sutun: tahmin):")
    print(matrix)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
    }


print("--- Phishing Veri Seti ---")
print(f"Toplam URL sayisi: {len(phishingDf)}")
print(f"Ozellik sayisi: {len(featureNames)}")
print(f"Phishing sayisi: {target.sum()}")
print(f"Legitimate sayisi: {(target == 0).sum()}")

# Recall, phishing sitelerini kacirmama basarisini gosterir.
decisionTreeResults = printModelResults(
    "Karar Agaci Siniflandiricisi", yTest, decisionTreePrediction
)
randomForestResults = printModelResults(
    "Rassal Orman Siniflandiricisi", yTest, randomForestPrediction
)

print("\n--- Model Karsilastirmasi ---")
print("                 Accuracy  Precision  Recall")
print(
    f"Karar Agaci      {decisionTreeResults['Accuracy']:.3f}      "
    f"{decisionTreeResults['Precision']:.3f}      {decisionTreeResults['Recall']:.3f}"
)
print(
    f"Rassal Orman     {randomForestResults['Accuracy']:.3f}      "
    f"{randomForestResults['Precision']:.3f}      {randomForestResults['Recall']:.3f}"
)

if randomForestResults["Recall"] > decisionTreeResults["Recall"]:
    print("Phishing Recall degerine gore rassal orman daha basarilidir.")
elif decisionTreeResults["Recall"] > randomForestResults["Recall"]:
    print("Phishing Recall degerine gore karar agaci daha basarilidir.")
else:
    print("Iki model ayni phishing Recall degerine sahiptir.")
