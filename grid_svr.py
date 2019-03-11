#!/bin/env python

import sys
import os
import grp
import subprocess
import numpy as np
from itertools import product

ada2="/home/proyectos/ada2/atorres/"
sbatch="/usr/local/slurm/slurm-2.6.4/bin/sbatch"

if len(sys.argv) < 3 or len(sys.argv) > 4:
    print "usage: {0} OUTDIR TRAIN [VAL]".format(sys.argv[0])
    sys.exit(1)

if grp.getgrgid(os.getgid()).gr_name != 'ada2':
    print "group is not ada2"
    sys.exit(2)

# check first arg is a dir, check other args are files

outdir = sys.argv[1]

C_base = g_base = e_base = 2.0
C_inc = g_inc = e_inc = 1

C_from, C_to =  1, 10
g_from, g_to = -15,-5
e_from, e_to =  -6, 0

C_range = C_base ** np.arange(C_from, C_to, C_inc)
gamma_range = g_base ** np.arange(g_from, g_to, g_inc)
epsilon_range = e_base ** np.arange(e_from, e_to, e_inc)

print "C_range = {:.2g} ** [{:.2g}:{:.2g}:{:.2g}]".format(C_base, C_from, C_to, C_inc)
print "gamma_range = {:.2g} ** [{:.2g}:{:.2g}:{:.2g}]".format(g_base, g_from, g_to, g_inc)
print "eps_range = {:.2g} ** [{:.2g}:{:.2g}:{:.2g}]".format(e_base, e_from, e_to, e_inc)

grid = product(C_range, gamma_range, epsilon_range)

for idx, params in enumerate(grid):
    # run job in SLURM and return jobid
    fout = '{0}/svr-{1:03d}.out'.format(outdir, idx)
    args = [sbatch, '-p', 'ccc', '-A', 'ada2_serv', '-o', fout,
            ada2 + 'svr.py'] + sys.argv[2:] + [ str(param) for param in params ]
    output = subprocess.Popen(args, stdout=subprocess.PIPE).stdout.read()
    print output.split()[-1], params
