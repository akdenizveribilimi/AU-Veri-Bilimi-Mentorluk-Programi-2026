#phishing website mı değil mi? tahmin ediyoruz!

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

#veri setimizi hazırlayalım.
data = pd.read_csv("phishing_websites_model.csv")
clean_data = data.drop("Result", axis=1)       #bunu yapmamdaki sebep, zaten 'label' sütununun da aynı şeyi ifade etmesi.

print("Clean Data Set:\n----------------------------------------------------")
print(clean_data)

#Encoding zamanı.
le = LabelEncoder()
clean_data["label"] = le.fit_transform(clean_data["label"])

#x ve y'leri ayır.
y = clean_data["label"]
x = clean_data.drop("label", axis=1)

print("X Table First 5\n----------------------------------------------------------")
print(x.head())

print("Y Table First 5\n----------------------------------------------------------")
print(y.head())

#train ve test verilerini ayıralım.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nx train size: ", x_train.shape)
print("x test size: ", x_test.shape, "\n")

print("y train size: ", y_train.shape)
print("y test size: ", y_test.shape)

#model eğitelim.
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(x_train, y_train)
dtc_prediction = dtc.predict(x_test)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(x_train, y_train)
rfc_prediction = rfc.predict(x_test)

print("\n=== DECISION TREE ===")
print(f"Accuracy : {accuracy_score(y_test, dtc_prediction):.4f}")
print(f"Recall : {recall_score(y_test, dtc_prediction):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, dtc_prediction))

print("\n=== RANDOM FOREST ===")
print(f"Accuracy : {accuracy_score(y_test, rfc_prediction):.4f}")
print(f"Recall : {recall_score(y_test, rfc_prediction):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, rfc_prediction))


