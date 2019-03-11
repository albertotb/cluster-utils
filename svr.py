#!/bin/env python

import os
import sys
import pickle
import numpy as np

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer

if __name__ == '__main__': 

    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print "usage: {0} TRAIN [TEST] C GAMMA EPS".format(sys.argv[0])
        sys.exit(1)

    train = np.loadtxt(sys.argv[1])
    y_train, X_train = train[:, 0], train[:, 1:]

    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #X_train = scaler.fit_transform(X_train)

    if len(sys.argv) == 5:
        params = sys.argv[2:]
    else:
        test = np.loadtxt(sys.argv[2])
        y_test, X_test = test[:, 0], test[:, 1:]
        params = sys.argv[3:]
        #X_test = scaler.transform(X_test)

    C = float(params[0])
    gamma = float(params[1])
    epsilon = float(params[2])

    model = SVR(C=C, gamma=gamma, epsilon=epsilon, cache_size=2000)

    if len(sys.argv) == 5:
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=mae_scorer)
        mae = -np.mean(scores)
        print C, gamma, epsilon, mae
    else:
        model.fit(X_train, y_train)
        mae_train = mean_absolute_error(y_train, model.predict(X_train))
        y_pred = model.predict(X_test)
        mae_test = mean_absolute_error(y_test, y_pred)
        print C, gamma, epsilon, mae_train, mae_test
        #np.savetxt("svr.pred", np.hstack((y_test[:, None], y_pred[:, None])))


