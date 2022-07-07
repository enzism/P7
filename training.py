import pickle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


X = pd.read_csv("C:/projets/Openclassrooms/P7/App/export/train_X.csv", index_col=0)
y = pd.read_csv("C:/projets/Openclassrooms/P7/App/export/train_y.csv", index_col=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

y_train = y_train.squeeze()
y_test = y_test.squeeze()

model = LGBMClassifier(boosting='gbdt', learning_rate=0.1,
                       max_bin=510, num_leaves=16, objective='binary',
                       random_state=510, reg_alpha=1.2, reg_lambda=1.4,
                       subsample=0.7)
clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

model.fit(X_train, y_train)

pickle.dump(model, open('trained_model', 'wb'))
pickle.dump((X_test, y_test), open('test_data', 'wb'))
