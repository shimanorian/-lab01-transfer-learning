import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score

input_file = 'income_data.txt'

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

# Преобразование строковых данных в числовые
label_encoder = []
X_encoded = np.empty(X.shape, dtype=object)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X_final = X_encoded[:, :-1].astype(int)
y_final = X_encoded[:, -1].astype(int)

# Создание SVM-классификатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=5000))
classifier.fit(X_final, y_final)

# Перекрестная проверка
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=5)
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=5000))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

f1 = cross_val_score(classifier, X_final, y_final, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

# --- Предсказание для нескольких тестовых точек (не менее 3) ---
test_points = [
    ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners',
     'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States'],
    ['45', 'Local-gov', '150000', 'Masters', '14', 'Married-civ-spouse', 'Exec-managerial',
     'Husband', 'White', 'Male', '5000', '0', '50', 'United-States'],
    ['22', 'Private', '90000', '11th', '7', 'Never-married', 'Sales',
     'Own-child', 'Black', 'Female', '0', '0', '25', 'United-States'],
]

for point in test_points:
    input_data_encoded = [-1] * len(point)
    count = 0
    for i, item in enumerate(point):
        if item.isdigit():
            input_data_encoded[i] = int(point[i])
        else:
            input_data_encoded[i] = int(label_encoder[count].transform([point[i]])[0])
            count += 1
    input_data_encoded = np.array(input_data_encoded).reshape(1, -1)
    predicted_class = classifier.predict(input_data_encoded)
    print("Точка:", point, "-> Прогноз:", label_encoder[-1].inverse_transform(predicted_class)[0])