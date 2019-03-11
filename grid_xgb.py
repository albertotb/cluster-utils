#!/bin/env python

import sys
import os
import grp
import subprocess
from itertools import product

ada2="/home/proyectos/ada2/atorres/"
sbatch="/usr/local/slurm/slurm-2.6.4/bin/sbatch"

if len(sys.argv) < 3 or len(sys.argv) > 4:
    print "usage: {0} OUTDIR TRAIN [VAL]".format(sys.argv[0])
    sys.exit(1)

if grp.getgrgid(os.getgid()).gr_name != 'ada2':
    print "group is not ada2"
    sys.exit(2)

outdir = sys.argv[1]

n_estimators_range = [200, 400, 600]
colsample_bylevel_range = [0.5, 0.75, 1]
min_child_weight_range = [1, 2, 4]
gamma_range = [1, 10, 100]
lambda_range = [0.5, 5, 50]
max_depth_range = [2, 4, 8]
eta_range = [0.1]

grid = product(n_estimators_range, colsample_bylevel_range, min_child_weight_range, 
        gamma_range, lambda_range, max_depth_range, eta_range)

for idx, params in enumerate(grid):
    # run job in SLURM and return jobid
    fout = '{0}/xgb-{1:03d}.out'.format(outdir, idx)
    args = [sbatch, '-p', 'ccc', '-A', 'ada2_serv', '-o', fout,
            ada2 + 'xgb.py'] + sys.argv[2:] + [ str(param) for param in params ]
    output = subprocess.Popen(args, stdout=subprocess.PIPE).stdout.read()
    print output.split()[-1], params
