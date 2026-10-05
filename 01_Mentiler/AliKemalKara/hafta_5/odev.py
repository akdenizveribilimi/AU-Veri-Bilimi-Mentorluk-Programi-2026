# -*- coding: utf-8 -*-
"""5. Hafta Odev: Student Performance veri setinin gorsellestirilmesi."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent.parent / "hafta_4" / "student_performance_dataset.csv"
studentDf = pd.read_csv(DATA_PATH)

educationMode = studentDf["parental_education"].mode()[0]
studentDf["parental_education"] = studentDf["parental_education"].fillna(educationMode)

print("--- Veri Seti ---")
print(studentDf.head())
print("\nEksik veri sayisi:")
print(studentDf.isnull().sum())

# 1. Scatter grafigi: calisma suresi ile final sinav notu arasindaki iliski.
plt.figure(figsize=(8, 5))
plt.scatter(
    studentDf["study_time_hours"],
    studentDf["final_exam_score"],
    alpha=0.7,
    color="steelblue",
    edgecolors="white",
)
plt.title("Calisma Suresi ve Final Sinav Notu")
plt.xlabel("Calisma suresi (saat)")
plt.ylabel("Final sinav notu")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# 2. Bar grafigi: ebeveyn egitim durumuna gore ortalama final notu.
educationMeans = (
    studentDf.groupby("parental_education")["final_exam_score"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))
educationMeans.plot(kind="bar", color="darkorange")
plt.title("Ebeveyn Egitim Durumuna Gore Ortalama Final Notu")
plt.xlabel("Ebeveyn egitim durumu")
plt.ylabel("Ortalama final notu")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

# 3. Line grafigi: ogrencilerin final notlarini sirali index boyunca izlemek.
orderedScores = studentDf["final_exam_score"].sort_values().reset_index(drop=True)

plt.figure(figsize=(9, 5))
plt.plot(orderedScores.index + 1, orderedScores, color="seagreen", linewidth=2)
plt.title("Ogrencilerin Sirali Final Sinav Notlari")
plt.xlabel("Ogrenci sirasi")
plt.ylabel("Final sinav notu")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

print("\n--- Grafiklerden Cikarilan Kisa Yorumlar ---")
print("1. Scatter grafigi, calisma suresi ile final notu arasindaki dagilimi gosterir.")
print("2. Bar grafigi, ebeveyn egitim gruplarinin ortalama notlarini karsilastirir.")
print("3. Line grafigi, final notlarinin dusukten yuksege dagilimini gosterir.")
print(f"En yuksek grup ortalamasi: {educationMeans.index[0]} ({educationMeans.iloc[0]:.2f})")
print(f"En dusuk grup ortalamasi: {educationMeans.index[-1]} ({educationMeans.iloc[-1]:.2f})")
