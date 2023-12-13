import pandas as pd
import datascalling_crossval as ds
dataset = pd.read_excel('dataset/Dataset Prediksi Diabetes.xlsx', sheet_name='Sheet2')

#menentukan variabel independen dalam x
x = dataset.drop(["Outcome"],axis=1)
#menentukan variabel dependen dalam y
y = dataset["Outcome"]

#melakukan train and test data dengan Stratified Cross Validation
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, random_state=42,shuffle=True)

find_neighbour = [3,5,7,9,11,13,15,17,19]
akurasi_tanpa_scalling = ds.akurasiTanpaScalling(x,y,skf,find_neighbour)
akurasi_standarisasi = ds.akurasiStandarisasi(x,y,skf,find_neighbour)
akurasi_normalisasi = ds.akurasiNormalisasi(x,y,skf,find_neighbour)

semua_akurasi = {}
for n in find_neighbour:
    semua_akurasi[n] = [akurasi_tanpa_scalling[n],akurasi_standarisasi[n],akurasi_normalisasi[n]]

show_df = pd.DataFrame.from_records(index=['Akurasi Tanpa Scalling', 'Akurasi Standarisasi','Akurasi Normalisasi'], data = semua_akurasi)
show_df.columns = ['3','5','7','9','11','13','15','17','19']
print(show_df)

# mencari best n dari ketiga data scalling
max_neighbour_tanpa_scalling = max(akurasi_tanpa_scalling,key=akurasi_tanpa_scalling.get)
max_neighbour_standarisasi = max(akurasi_standarisasi,key=akurasi_standarisasi.get)
max_neighbour_normalisasi = max(akurasi_normalisasi,key=akurasi_normalisasi.get)

dict_best_neighbour = {}
dict_best_neighbour['Tanpa Scalling'] = akurasi_tanpa_scalling[max_neighbour_tanpa_scalling]
dict_best_neighbour['Scalling Standarisasi'] = akurasi_standarisasi[max_neighbour_standarisasi]
dict_best_neighbour['Scalling Normalisasi'] = akurasi_normalisasi[max_neighbour_normalisasi]


best_neighbour = max(dict_best_neighbour, key=dict_best_neighbour.get)
max_result_neighbour = 0
if(best_neighbour=='Tanpa Scalling'):
    max_result_neighbour = max_neighbour_tanpa_scalling
elif(best_neighbour=='Scalling Standarisasi'):
    max_result_neighbour = max_neighbour_standarisasi
else:
    max_result_neighbour = max_neighbour_normalisasi

print(f"\nNilai akurasi yang paling baik adalah {best_neighbour} dengan nilai akurasi sebesar {dict_best_neighbour[best_neighbour]} pada nilai K sebesar {max_result_neighbour}.")

