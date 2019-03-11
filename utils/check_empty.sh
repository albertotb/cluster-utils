#!/bin/bash

if (( $# != 1 )); then
   echo "usage: $(basename $0) DIR"
   exit 1
fi

while true; do
   has_empty=0
   for file in $1/*; do
      if [ ! -s "$file" ]; then
         echo $file
         has_empty=1
         #break
      fi
   done

   if [ $has_empty != 1 ]; then
      break
   fi

   sleep 3600
done

echo "Dir has no empty files" #| mail -s 'SLURM' albertotb@gmail.com
