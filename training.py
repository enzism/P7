import pickle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


X = pd.read_csv("C:/projets/Openclassrooms/P7/App/export/train_X.csv", index_col=0)
y = pd.read_csv("C:/projets/Openclassrooms/P7/App/export/train_y.csv", index_col=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

model = LGBMClassifier(boosting='gbdt', learning_rate=0.1,
                       max_bin=510, num_leaves=16, objective='binary',
                       random_state=510, reg_alpha=1.2, reg_lambda=1.4,
                       subsample=0.7)
y_train = y_train.squeeze()
model.fit(X_train, y_train)

pickle.dump(model, open('trained_model', 'wb'))
pickle.dump((X_test), open('test_data', 'wb'))
