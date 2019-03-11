#!/bin/env python

import sys
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer

if len(sys.argv) < 7 or len(sys.argv) > 8:
    print "usage: {0} TRAIN [TEST] N_EST MAX_FEAT MIN_SAMPLES_SPLIT MIN_SAMPLES_LEAF MAX_DEPTH".format(sys.argv[0])
    sys.exit(1)

train = np.loadtxt(sys.argv[1])
y_train, X_train = train[:, 0], train[:, 1:]

#scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
#X_train = scaler.fit_transform(X_train)

if len(sys.argv) == 7:
    params = sys.argv[2:]
else:
    test = np.loadtxt(sys.argv[2])
    y_test, X_test = test[:, 0], test[:, 1:]
    params = sys.argv[3:]
    #X_test = scaler.transform(X_test)

n_estimators = int(params[0])
max_features = float(params[1])
min_samples_split = int(params[2])
min_samples_leaf = int(params[3])
try:
    max_depth = int(params[4])
except:
    max_depth = None

model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features,
        max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

if len(sys.argv) == 7:
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=mae_scorer)
    mae = -np.mean(scores)
else:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    #np.savetxt('rfr.pred', np.hstack((y_test[:, None], y_pred[:, None])))
    #np.savetxt('rfr.feat', model.feature_importances_[:, None])

print n_estimators, max_features, min_samples_split, min_samples_leaf, max_depth, mae_train, mae
