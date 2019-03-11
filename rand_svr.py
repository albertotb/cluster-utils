#!/bin/env python

import sys
import numpy as np
from predictor import train
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold

class SVMR(SVR):

    folding = KFold

    @staticmethod
    def space(X, y):
       return {
            'log_C': [0, 10],
            'log_gamma': [-15, -5],
            'log_epsilon': [-6, 0],
        }

    @staticmethod
    def loss(y_true, y_pred):
        return mean_absolute_error(y_true=y_true, y_pred=y_pred)

    metrics = [mean_absolute_error]

    def __init__(
        self,
        log_C=0,
        log_gamma=-5,
        log_epsilon=0,
    ):
        SVR.__init__(
            self,
            cache_size=2000,
            C = 2**log_C,
            gamma = 2**log_gamma,
            epsilon = 2**log_epsilon,
        )

    def fit(self, X, y):
        return SVR.fit(self, X, y)

    def predict(self, X):
        return SVR.predict(self, X)


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
    print SVMR.space(X_train, y_train)

    p, metaparams, metrics = train(
        estimator=SVMR,
        num_evals=nevals,
        num_jobs=njobs,
        X=X_train,
        y=y_train,
        X_test=X_val,
        y_test=y_val
    )

    print metaparams
    print 2**np.array(metaparams.values())
    print metrics

