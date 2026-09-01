#sağlık sigortası masrafını tahmin ediyoruz.

#kütüphaneleri hazırlayalım.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

#veri setimizi okuyalım ve boş verileri temizleyelim.
data = pd.read_csv("insurance.csv")
print("Data Set:\n----------------------------------------------------------")
print(data)

clean_data = data.dropna()
print("Clean Data Set:\n----------------------------------------------------------")
print(clean_data)

#veri ön işleme(encoding, x ve y'ye ayırma işlemi) yapıyoruz.
le = LabelEncoder()
clean_data["sex"] = le.fit_transform(clean_data["sex"])
clean_data["smoker"] = le.fit_transform(clean_data["smoker"])
clean_data = pd.get_dummies(clean_data, columns=["region"], dtype=int)

#charges dışındaki değişkenler(x)
x = pd.concat([clean_data.iloc[:, :5], clean_data.iloc[:, 6:]], axis=1)
y = clean_data.iloc[:, 5]  #charges sütunu(hedef sütun)

print("X Table First 5\n----------------------------------------------------------")
print(x.head())

print("Y Table First 5\n----------------------------------------------------------")
print(y.head())

#test ve train verilerimizi ayıralım.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nx train size: ", x_train.shape)
print("x test size: ", x_test.shape, "\n")

print("y train size: ", y_train.shape)
print("y test size: ", y_test.shape)

#model eğitme zamanı.
lr = LinearRegression()
lr.fit(x_train, y_train)

#tahmin yapalım.
y_prediction = lr.predict(x_test)
result_table = pd.DataFrame({
    "true value: (y_test)" : y_test.values,
    "model's prediction: (y_prediction)" : y_prediction
})

print("comparison:")
print(result_table.head(10))
print("---------------------------------------------------------")

#R-Kare ve MSE hesaplama
mse = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)

print("--- Model Performance Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-Square (R^2) Score: {r2:.4f}")