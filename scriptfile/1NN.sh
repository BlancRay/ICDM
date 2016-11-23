#!/bin/bash
dir=./1NN
file=./output
out=./out.csv
echo dataset,error,fmeasure > ./out
for dataset in `ls $dir`;do
	cat ./$dir/$dataset/$dataset.out | while read line
	do
	if  [[ $line == TestError* ]] ;
	then
		flg=1
		echo -n $dataset','>>./out
		echo -n `echo ${line#T*:}`','>>./out
		continue
	fi
	if	[[ $flg == 1 ]] ;
	then
		echo $line>>./out
		flg=0
	fi
	done 
done 