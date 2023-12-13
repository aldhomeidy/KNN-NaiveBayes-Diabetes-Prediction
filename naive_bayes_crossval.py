import pandas as pd

dataset = pd.read_excel('dataset/Dataset Prediksi Diabetes.xlsx', sheet_name='Sheet2')

#menentukan variabel independen dalam x
x = dataset.drop(["Outcome"],axis=1) 
#menentukan variabel dependen dalam y
y = dataset["Outcome"]

#melakukan train and test data dengan Stratified Cross Validation
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, random_state=42,shuffle=True)

# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
# Build a Gaussian Classifier
nb = GaussianNB()
from sklearn.model_selection import cross_val_score

# Menghitung nilai akurasi dari klasifikasi naive bayes 
accuracy = (cross_val_score(nb,x,y,cv=skf,scoring="accuracy")).mean()

print(f"Nilai Akurasi : {accuracy}")