from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# ========================Start Tanpa Scalling========================
def akurasiTanpaScalling(x,y,skf, find_neighbour):
    result_accuracy = {}
    # Pencarian jumlah Neighbour ganjil terbaik mulai dari 3 hingga 19
    for n in find_neighbour:
        knn = KNeighborsClassifier(n_neighbors= n, metric="euclidean", weights="distance")
        results = cross_val_score(knn,x,y,cv=skf,scoring="accuracy")
        result_accuracy[n] = results.mean()
        
    return result_accuracy
# ========================End Tanpa Scalling========================





# ========================Start Standarisasi (Zero-Mean)========================
def akurasiStandarisasi(x,y,skf, find_neighbour):
    result_accuracy = {}
    
    from sklearn.preprocessing import StandardScaler
    scaler_standard = StandardScaler()
    scaler_standard.fit(x)

    x_transform = scaler_standard.transform(x)

    # Pencarian jumlah Neighbour ganjil terbaik mulai dari 3 hingga 19
    for n in find_neighbour:
        knn = KNeighborsClassifier(n_neighbors= n, metric="euclidean", weights="distance")
        results = cross_val_score(knn,x_transform,y,cv=skf,scoring="accuracy")
        result_accuracy[n] = results.mean()

    return result_accuracy
# ========================End Standarisasi (Zero-Mean)========================





# ========================Start Normalisasi (Min-Max)========================
def akurasiNormalisasi(x,y,skf, find_neighbour):
    result_accuracy = {}
    
    from sklearn.preprocessing import MinMaxScaler
    scaler_minmax = MinMaxScaler()
    scaler_minmax.fit(x)

    x_transform = scaler_minmax.transform(x)
    
    # Pencarian jumlah Neighbour ganjil terbaik mulai dari 3 hingga 19
    for n in find_neighbour:
        knn = KNeighborsClassifier(n_neighbors= n, metric="euclidean", weights="distance")
        results = cross_val_score(knn,x_transform,y,cv=skf,scoring="accuracy")
        result_accuracy[n] = results.mean()

    return result_accuracy
# ========================End Normalisasi (Min-Max)========================