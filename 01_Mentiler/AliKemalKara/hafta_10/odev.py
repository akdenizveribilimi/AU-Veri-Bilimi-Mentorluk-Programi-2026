# -*- coding: utf-8 -*-
"""10. Hafta Odev: K-Means ile musteri segmentasyonu."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "Mall_Customers.csv"
customerDf = pd.read_csv(DATA_PATH)

featureNames = ["Annual Income (k$)", "Spending Score (1-100)"]
features = customerDf[featureNames]

# Ozellikleri olceklendirerek gelir ve harcama puaninin etkisini dengeliyoruz.
scaler = StandardScaler()
scaledFeatures = scaler.fit_transform(features)

# Farkli k degerleri icin inertia hesaplayarak dirsek yontemini uyguluyoruz.
inertiaValues = []
clusterRange = range(2, 9)
for clusterCount in clusterRange:
    model = KMeans(n_clusters=clusterCount, random_state=42, n_init=10)
    model.fit(scaledFeatures)
    inertiaValues.append(model.inertia_)

print("--- K-Means Dirsek Analizi ---")
for clusterCount, inertia in zip(clusterRange, inertiaValues):
    print(f"K={clusterCount}: Inertia={inertia:.2f}")

# Dirsek grafiginde 5 kume, bu veri seti icin kullanisli bir segment sayisidir.
selectedClusterCount = 5
kmeansModel = KMeans(
    n_clusters=selectedClusterCount,
    random_state=42,
    n_init=10,
)
customerDf["cluster"] = kmeansModel.fit_predict(scaledFeatures)

clusterCenters = scaler.inverse_transform(kmeansModel.cluster_centers_)
clusterSummary = (
    customerDf.groupby("cluster")[featureNames]
    .mean()
    .sort_values("Spending Score (1-100)", ascending=False)
)

print("\n--- Musteri Segmentleri ---")
print(f"Toplam musteri sayisi: {len(customerDf)}")
print(f"Secilen kume sayisi: {selectedClusterCount}")
print("\nKume basina musteri sayisi:")
print(customerDf["cluster"].value_counts().sort_index())
print("\nKume ortalamalari:")
print(clusterSummary.round(2))

print("\n--- Segment Yorumlari ---")
for clusterId, row in clusterSummary.iterrows():
    income = row["Annual Income (k$)"]
    spending = row["Spending Score (1-100)"]

    if income >= 70 and spending >= 60:
        segmentName = "Yuksek gelirli ve aktif harcayanlar"
    elif income >= 70 and spending < 60:
        segmentName = "Yuksek gelirli ancak temkinli harcayanlar"
    elif income < 70 and spending >= 60:
        segmentName = "Orta gelirli ve aktif harcayanlar"
    else:
        segmentName = "Dusuk/orta gelirli ve dusuk harcayanlar"

    print(f"Kume {clusterId}: {segmentName}")

# Dirsek grafigi.
plt.figure(figsize=(8, 5))
plt.plot(list(clusterRange), inertiaValues, marker="o", color="darkorange")
plt.title("K-Means Dirsek Yontemi")
plt.xlabel("Kume sayisi")
plt.ylabel("Inertia")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# Musterileri secilen iki ozellik uzerinden renkli olarak gosteriyoruz.
plt.figure(figsize=(8, 5))
plt.scatter(
    customerDf[featureNames[0]],
    customerDf[featureNames[1]],
    c=customerDf["cluster"],
    cmap="viridis",
    alpha=0.8,
)
plt.scatter(
    clusterCenters[:, 0],
    clusterCenters[:, 1],
    marker="X",
    s=180,
    color="red",
    edgecolors="black",
    label="Kume merkezi",
)
plt.title("Musteri Segmentleri")
plt.xlabel("Yillik gelir (k$)")
plt.ylabel("Harcama puani")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
