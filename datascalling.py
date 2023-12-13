from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ========================Start Tanpa Scalling========================
def akurasiTanpaScalling(x_train, x_test, y_train, y_test, find_neighbour):
    result_accuracy = {}
    # Pencarian jumlah Neighbour ganjil terbaik mulai dari 3 hingga 19
    for n in find_neighbour:
        knn = KNeighborsClassifier(n_neighbors= n, metric="euclidean", weights="distance")
        knn.fit(x_train,y_train)
        y_pred = knn.predict(x_test)
        result_accuracy[n] = accuracy_score(y_test, y_pred)

    return result_accuracy
# ========================End Tanpa Scalling========================





# ========================Start Standarisasi (Zero-Mean)========================
def akurasiStandarisasi(x_train, x_test, y_train, y_test, find_neighbour):
    result_accuracy = {}
    
    from sklearn.preprocessing import StandardScaler
    scaler_standard = StandardScaler()
    scaler_standard.fit(x_train)

    x_train = scaler_standard.transform(x_train)
    x_test = scaler_standard.transform(x_test)

    # Pencarian jumlah Neighbour ganjil terbaik mulai dari 3 hingga 19
    for n in find_neighbour:
        knn = KNeighborsClassifier(n_neighbors= n, metric="euclidean", weights="distance")
        knn.fit(x_train,y_train)
        y_pred = knn.predict(x_test)
        result_accuracy[n] = accuracy_score(y_test, y_pred)

    return result_accuracy
# ========================End Standarisasi (Zero-Mean)========================





# ========================Start Normalisasi (Min-Max)========================
def akurasiNormalisasi(x_train, x_test, y_train, y_test, find_neighbour):
    result_accuracy = {}
    
    from sklearn.preprocessing import MinMaxScaler
    scaler_minmax = MinMaxScaler()
    scaler_minmax.fit(x_train)

    x_train = scaler_minmax.transform(x_train)
    x_test = scaler_minmax.transform(x_test)
    
    # Pencarian jumlah Neighbour ganjil terbaik mulai dari 3 hingga 19
    for n in find_neighbour:
        knn = KNeighborsClassifier(n_neighbors= n, metric="euclidean", weights="distance")
        knn.fit(x_train,y_train)
        y_pred = knn.predict(x_test)
        result_accuracy[n] = accuracy_score(y_test, y_pred)

    return result_accuracy
# ========================End Normalisasi (Min-Max)========================