#tümör iyi huylu mu kötü huylu mu?

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv")

#veri setini temizleyelim.
clean_data = data.drop(['id', 'Unnamed: 32'], axis=1)
print("Data Set:\n----------------------------------------------------")
print(clean_data)

#encoding yapıyoruz.
le = LabelEncoder()
clean_data["diagnosis"] = le.fit_transform(clean_data["diagnosis"])

#bağımlı ve bağımsız değişkenlerin belirlenmesi.
y = clean_data["diagnosis"]  #diagnosis sütunu(hedef sütun)
x = clean_data.drop("diagnosis", axis=1)

#örnek olarak ilk 5 satırı gösteriyorum.
print("X Table First 5\n----------------------------------------------------------")
print(x.head())

print("Y Table First 5\n----------------------------------------------------------")
print(y.head())

#train ve test olarak veri setimizi ayıralım.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nx train size: ", x_train.shape)
print("x test size: ", x_test.shape, "\n")

print("y train size: ", y_train.shape)
print("y test size: ", y_test.shape)

#çok büyük ve çok küçük sayılarla çalışma durumundan dolayı ölçeklenidirici kullandım.
#bu şekilde aynı orantıya çekiyoruz.
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#model oluşturma.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_scaled, y_train)

knn_prediction = knn.predict(x_test_scaled)

print("Scaled KNN R² Score:", accuracy_score(y_test, knn_prediction))


