#!/bin/bash

# check if an input filename was passed as a command
# line argument:
if [ $# -lt 2 ]; then
echo “Please specify two csv files at least to concat!”
exit
fi

# create a directory to store the output:
mkdir -p concat_output
filename=${1%.*}
cat $1 >> ./concat_output/$filename.csv
shift
for f in $@; 
do tail -n +2 $f >> ./concat_output/$filename.csv; 
done;

exit