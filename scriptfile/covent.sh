#!/bin/bash
out=./out.csv
dir=./J48
echo "dataset	Training time	Testing time	TestError	FMeasures" >./out.csv
for dataset in `ls $dir`;do
	cat ./$dir/$dataset/$dataset.out | while read line
    do
    if  [[ $line == "processing: "* ]] ;
    then
        echo -n `echo ${line#p*: }`'	'>>./out.csv
    fi
    if  [[ $line == "Training time:"* ]] ;
    then
        echo -n `echo ${line#T*:}`'	'>>./out.csv
    fi
    if  [[ $line == "Testing time:"* ]] ;
    then
        echo -n `echo ${line#T*:}`'	'>>./out.csv
    fi
    if  [[ $line == "TestError:"* ]] ;
    then
        echo -n `echo ${line#T*:}`'	'>>./out.csv
    fi
    if  [[ $line == "FMeasures:"* ]] ;
    then
        echo `echo ${line#F*:}`'	'>>./out.csv
    fi
    done 
done