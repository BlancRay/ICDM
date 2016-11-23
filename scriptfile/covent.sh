#!/bin/bash
file=./output
out=./out.csv
cat ./output | while read line
do
if  [[ $line == process* ]] ;
then
echo "">>./out.csv
    echo -n $line'	'>>./out.csv
fi
if  [[ $line == PT* ]] ;
then
    echo -n $line'	'>>./out.csv
fi
if  [[ $line == TestError* ]] ;
then
    echo -n $line'	'>>./out.csv
fi
done 