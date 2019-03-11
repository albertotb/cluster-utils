#!/bin/env python

import sys
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer

if len(sys.argv) < 11 or len(sys.argv) > 13:
    print ('usage: {0} TRAIN [TEST] N_EST MAX_FEAT COLSAMPLE ROWSAMPLE '
           'MIN_CHILD GAMMA LAMBDA MAX_DEPTH L_RATE').format(sys.argv[0])
    sys.exit(1)

train = np.loadtxt(sys.argv[1])
y_train, X_train = train[:, 0], train[:, 1:]

#scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
#X_train = scaler.fit_transform(X_train)

if len(sys.argv) == 9:
    params = sys.argv[2:]
else:
    test = np.loadtxt(sys.argv[2])
    y_test, X_test = test[:, 0], test[:, 1:]
    params = sys.argv[3:]
    #X_test = scaler.transform(X_test)

n_estimators = int(params[0])
colsample_bylevel = float(params[1])
colsample_bytree = float(params[2])
subsample = float(params[3])
min_child_weight = int(params[4])
gamma = float(params[5])
lam = float(params[6])
max_depth = int(params[7])
eta = float(params[8])

model = xgb.XGBRegressor(n_estimators=n_estimators, reg_lambda=lam,
        gamma=gamma, subsample=subsample, colsample_bylevel=colsample_bylevel,
        colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
        max_depth=max_depth, learning_rate=eta, objective='reg:linear')

if len(sys.argv) == 9:
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=mae_scorer)
    mae = -np.mean(scores)
else:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

print n_estimators, colsample_bylevel, colsample_bytree, subsample, \
        min_child_weight, gamma, lam, max_depth, eta, mae_train, mae
