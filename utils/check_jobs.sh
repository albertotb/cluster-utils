#!/bin/bash

EMAIL="albertotb@gmail.com"
POLL_TIME=100

if (( $# < 0 )) || (( $# > 1 )); then
   echo "usage: $(basename $0) [JOB_NAME]"
   exit 1
fi

if (( $# == 1 )); then
   name="-n $1"
fi

while [[ $(squeue -u atorres $name | wc -l) != 1 ]]
do
   sleep $POLL_TIME
done

if (( $# == 1 )); then
   msg="No jobs left with name $name"
else
   msg="No jobs left\nTime(s) = $SECONDS"
fi

echo -e $msg | mail -s "SLURM" $EMAIL
