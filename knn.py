import pandas as pd
import datascalling as ds

dataset = pd.read_excel('dataset/Dataset Prediksi Diabetes.xlsx', sheet_name='Sheet1')

#menentukan variabel independen dalam x
x = dataset.drop(["Outcome"],axis=1)                                                             #axis bernilai 1 jika memilih drop by nama column, bernilai 0 jika memilih by index column
#menentukan variabel dependen dalam y
y = dataset["Outcome"]

#membagi data yang ada menjadi data test dan data train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    

find_neighbour = [3,5,7,9,11,13,15,17,19]
akurasi_tanpa_scalling = ds.akurasiTanpaScalling(x_train, x_test, y_train, y_test,find_neighbour)
akurasi_standarisasi = ds.akurasiStandarisasi(x_train, x_test, y_train, y_test, find_neighbour)
akurasi_normalisasi = ds.akurasiNormalisasi(x_train, x_test, y_train, y_test, find_neighbour)

semua_akurasi = {}
for n in find_neighbour:
    semua_akurasi[n] = [akurasi_tanpa_scalling[n],akurasi_standarisasi[n],akurasi_normalisasi[n]]

show_df = pd.DataFrame.from_records(index=['Akurasi Tanpa Scalling', 'Akurasi Standarisasi','Akurasi Normalisasi'], data = semua_akurasi)
show_df.columns = ['3','5','7','9','11','13','15','17','19']
print(show_df)

# mencari best n dari ketiga data scalling
dict_best_neighbour = {}
dict_best_neighbour['Tanpa Scalling'] = max(akurasi_tanpa_scalling,key=akurasi_tanpa_scalling.get)
dict_best_neighbour['Scalling Standarisasi'] = max(akurasi_standarisasi,key=akurasi_standarisasi.get)
dict_best_neighbour['Scalling Normalisasi'] = max(akurasi_normalisasi,key=akurasi_normalisasi.get)

best_neighbour = max(dict_best_neighbour, key=dict_best_neighbour.get)
print(f"\nNilai akurasi yang paling baik adalah {best_neighbour} dengan jumlah neighbour sebesar {dict_best_neighbour[best_neighbour]}.")


# Karena didapatkan hasil paling baik adalah tanpa scalling, maka dilakukan knn dengan neighbour yang didapatkan
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, recall_score, precision_score
knn = KNeighborsClassifier(n_neighbors= dict_best_neighbour[best_neighbour], metric="euclidean", weights="distance")
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print(f"Hasil Confusion Matrix : \n {confusion_matrix(y_test,y_pred)}")
print(f"Nilai Akurasi : {accuracy_score(y_test, y_pred)}")
print(f"Nilai Presisi : {precision_score(y_test, y_pred)}")
print(f"Nilai Recall : {recall_score(y_test, y_pred)}")
print(f"Hasil Klasifikasi dalam bentuk Report : \n {classification_report(y_test,y_pred)}")

