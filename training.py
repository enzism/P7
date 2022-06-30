import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'])

model = RandomForestClassifier(n_estimators=100, max_feature=0.8)
model.fit(X_train, y_train)

X_test = pd.DataFrame(X_test, columns=iris['feature_names'])
pickle.dump(model, open('trained_model', 'wb'))
pickle.dump((X_test, y_test), open('test_data', 'wb'))
