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

n_estimators_range = [100, 250, 500, 750, 1000]
max_features_range = [0.2, 0.3, 0.4, 0.5, 0.6]
min_samples_split_range = [5, 10, 20, 30, 40]
min_samples_leaf_range = [1, 2, 5, 10]
max_depth_range = [None]

grid = product(n_estimators_range, max_features_range, min_samples_split_range,
               min_samples_leaf_range, max_depth_range)

for idx, params in enumerate(grid):
    # run job in SLURM and return jobid
    fout = '{0}/rfr-{1:03d}.out'.format(outdir, idx)
    args = [sbatch, '-p', 'ccc', '-A', 'ada2_serv', '-o', fout,
            ada2 + 'rfr.py'] + sys.argv[2:] + [ str(param) for param in params ]
    output = subprocess.Popen(args, stdout=subprocess.PIPE).stdout.read()
    print output.split()[-1], params
