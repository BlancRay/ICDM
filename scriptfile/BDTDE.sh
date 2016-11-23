#!/bin/bash
dir=./BDTDE
file=./output
out=./out.csv
echo dataset,error,fmeasure,Average Error,Average FMeasure > ./out
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
		echo -n $line','>>./out
		flg=0
	fi
	if  [[ $line == 'Average Error'* ]] ;
	then
		error=`echo ${line#A*:}`
		echo -n `echo ${error%' '*}`','>>./out
		echo `echo ${error##*F*e}`','>>./out
	fi
	done 
done 