import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

data = load_diabetes()

X, y = shuffle(data.data, data.target, random_state=7)

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Разные комбинации параметров (не менее 3, как просит методичка)
param_sets = [
    {'C': 1.0, 'epsilon': 0.1},
    {'C': 10.0, 'epsilon': 0.1},
    {'C': 1.0, 'epsilon': 1.0},
]
test_points = [X_test[0], X_test[1], X_test[2]]

for params in param_sets:
    sv_regressor = SVR(kernel='linear', **params)
    sv_regressor.fit(X_train, y_train)

    y_test_pred = sv_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    evs = explained_variance_score(y_test, y_test_pred)

    print(f"\n#### C={params['C']}, epsilon={params['epsilon']} ####")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))

    for i, point in enumerate(test_points):
        pred_value = sv_regressor.predict([point])[0]
        true_value = y_test[i]
        print(f"  Точка {i+1} предсказано: {round(pred_value, 2)}, истинное значение: {round(true_value, 2)}")