import pandas as pd

dataset = pd.read_excel('dataset/Dataset Prediksi Diabetes.xlsx', sheet_name='Sheet1')

#menentukan variabel independen dalam x
x = dataset.drop(["Outcome"],axis=1) 
#menentukan variabel dependen dalam y
y = dataset["Outcome"]

#membagi data yang ada menjadi data test dan data train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, recall_score, precision_score
# Build a Gaussian Classifier
nb = GaussianNB()
# Model training
nb.fit(x_train, y_train)
# Predict Output
y_pred = nb.predict(x_test)

# Menghitung nilai akurasi dari klasifikasi naive bayes 
from sklearn.metrics import classification_report
print(f"Hasil Confusion Matrix : \n {confusion_matrix(y_test,y_pred)}")
print(f"Nilai Akurasi : {accuracy_score(y_test, y_pred)}")
print(f"Nilai Presisi : {precision_score(y_test, y_pred)}")
print(f"Nilai Recall : {recall_score(y_test, y_pred)}")
print(f"Hasil Klasifikasi dalam bentuk Report : \n {classification_report(y_test,y_pred)}")