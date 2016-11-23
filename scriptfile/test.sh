#!/bin/sh
dir=/mnt/c/Users/xulei/workspace/ICDM
datadir=/mnt/c/Users/xulei/workspace/ICDM/UCR_TS_Archive_2015
java -Xmx2g -Dfile.encoding=UTF-8 -classpath ./bin:./lib/weka.jar:./lib/commons-math3-3.2.jar classif.ExperimentsLauncher