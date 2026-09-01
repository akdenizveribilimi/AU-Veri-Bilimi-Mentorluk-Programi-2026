#müşteri segmentasyonu gruplayalım.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#verimizi hazırlayalım.
data = pd.read_csv("Mall_Customers.csv")
x = data.iloc[:, [3, 4]].values     #Annual Income, Spending Score benim için önemli olucak burada.

#model eğit.
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)

#hem modeli eğitiyoruz hem de hangi gruba ait olunduğunu belirliyoruz.
y_kmeans = kmeans.fit_predict(x)

#çizdirelim.
plt.figure(figsize=(10, 6))       #10'a 6 sadece tuvalin ölçüleri gibi düşün.

#kümeleri grafiğe ekliyorum.
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')

#kümelerin merkez noktalarını devasa sarı noktalar olarak ekliyoruz.
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

plt.title('Customer Segmentation (K-Means)')
plt.xlabel('Annula Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()