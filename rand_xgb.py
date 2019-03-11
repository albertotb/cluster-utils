#!/bin/env python

import sys
import numpy as np
from predictor import train
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold

class XGBR(XGBRegressor):

    folding = KFold

    @staticmethod
    def space(X, y):
       return {
            'n_estimators': [500, 3000],
            'subsample': [0.5, 1],
            'colsample_bytree': [0.5, 1],
            'colsample_bylevel': [0.4, 1],
            'min_child_weight': [1, 10],
            'gamma': [0, 1],
            'reg_lambda': [0, 10],
            'max_depth': [1, 10],
            'log_learning_rate': [-4, np.log(0.5)]
        }

    @staticmethod
    def loss(y_true, y_pred):
        return mean_absolute_error(y_true=y_true, y_pred=y_pred)

    metrics = [mean_absolute_error]

    def __init__(
        self,
        n_estimators=100,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        min_child_weight=1,
        gamma=0,
        reg_lambda=1,
        max_depth=4,
        log_learning_rate=-1,
        missing=None
    ):
        XGBRegressor.__init__(
            self,
            objective='reg:linear',
            n_estimators=int(max(1, n_estimators)),
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            min_child_weight=int(max(1, min_child_weight)),
            gamma=gamma,
            reg_lambda=reg_lambda,
            max_depth=int(max(1, max_depth)),
            learning_rate=10 ** log_learning_rate,
            missing=missing
        )

    def fit(self, X, y):
        return XGBRegressor.fit(self, X, y)

    def predict(self, X):
        return XGBRegressor.predict(self, X)


if __name__ == "__main__":

    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print "usage: {0} EVALS JOBS TRAIN [VAL]".format(sys.argv[0])
        sys.exit(1)

    nevals = int(sys.argv[1])
    njobs = int(sys.argv[2])

    train_data = np.loadtxt(sys.argv[3])
    y_train, X_train = train_data[:, 0], train_data[:, 1:]

    if len(sys.argv) == 4:
        y_val, X_val = None
    else:
        val_data = np.loadtxt(sys.argv[4])
        y_val, X_val = val_data[:, 0], val_data[:, 1:]

    print "Evals: {}, jobs: {}".format(nevals, njobs)
    print XGBR.space(X_train, y_train)

    p, metaparams, metrics = train(
        estimator=XGBR,
        num_evals=nevals,
        #init_args ={'log_learning_rate': -1},
        num_jobs=njobs,
        X=X_train,
        y=y_train,
        X_test=X_val,
        y_test=y_val
    )

    print metaparams
    print metrics

