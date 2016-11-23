#!/bin/bash
file=./oup.txt
out=./tmp
cat ./oup.txt | while read line
do
if  [[ $line == process* ]] ;
then
echo "">>$out
    echo -n $line'	'>>$out
fi
if  [[ $line == PT* ]] ;
then
    echo -n $line'	'>>$out
fi
if  [[ $line == TestError* ]] ;
then
    echo -n $line'	'>>$out
fi
done 