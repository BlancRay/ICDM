#!/bin/env bash
dir=/home/lxu/ICDM/UCR_TS_Archive_2015
rm -rf ./runfk/
rm -rf ./outfk/
mkdir ./runfk/
mkdir ./outfk/
module load java/1.8.0_77
for dataset in `ls $dir`;do
	echo "creating $dataset.sh"
	echo "#!/bin/sh" >> ./run/run-$dataset.sh
	echo "#SBATCH --job-name=$dataset" >> ./run/run-$dataset.sh
	echo "#SBATCH --mem=4000" >> ./run/run-$dataset.sh
	echo "#SBATCH --time=150:00:00" >> ./run/run-$dataset.sh
	echo "#SBATCH --mail-type=FAIL" >> ./run/run-$dataset.sh
	echo "#SBATCH --output=./outfk/$dataset.out" >> ./run/run-$dataset.sh
	echo "module load java/1.8.0_77" >> ./run/run-$dataset.sh
	echo "java -Xmx2g -Dfile.encoding=UTF-8 -classpath /home/lxu/ICDM/fk/bin:/home/lxu/ICDM/lib/weka.jar:/home/lxu/ICDM/lib/commons-math3-3.2.jar classif.ExperimentsLauncher $dataset">> ./run/run-$dataset.sh
chmod +x ./runfk/run-$dataset.sh
done

for dataset in `ls $dir`;do
	echo "launching run-$dataset.sh"
	sbatch /home/lxu/ICDM/runfk/run-$dataset.sh
done

