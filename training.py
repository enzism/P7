import pickle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


X_train = pd.read_csv("C:/projets/Openclassrooms/P7/App/export/train_X.csv", index_col=0)
y_train = pd.read_csv("C:/projets/Openclassrooms/P7/App/export/train_y.csv", index_col=0)
X_test = pd.read_csv("C:/projets/Openclassrooms/P7/Data/application_test.csv", index_col=0)

model = LGBMClassifier(boosting='gbdt', learning_rate=0.1,
                       max_bin=510, num_leaves=16, objective='binary',
                       random_state=510, reg_alpha=1.2, reg_lambda=1.4,
                       subsample=0.7)
y_train = y_train.values.flatten()
model.fit(X_train, y_train)

pickle.dump(model, open('trained_model', 'wb'))
pickle.dump((X_test), open('test_data', 'wb'))
