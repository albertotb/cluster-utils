#!/bin/bash

USER="atorres"
EMAIL="albertotb@gmail.com"
POLL_TIME=100

if (( $# < 0 )) || (( $# > 1 )); then
   echo "usage: $(basename $0) [JOB_NAME]"
   exit 1
fi

while [[ $(squeue -o "%80j" -u $USER | grep "$1" | wc -l) != 0 ]]
do
   sleep $POLL_TIME
done

if (( $# == 1 )); then
   msg="No jobs left with name $1\nTime(s) = $SECONDS"
else
   msg="No jobs left\nTime(s) = $SECONDS"
fi

echo -e $msg | mail -s "SLURM" $EMAIL
