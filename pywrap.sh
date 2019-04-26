#!/bin/bash
#$ -q all.q
#$ -cwd
#$ -j y
python $@
