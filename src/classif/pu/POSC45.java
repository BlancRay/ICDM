package classif.pu;

import weka.core.*;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

import org.apache.commons.math3.random.RandomDataGenerator;

import weka.classifiers.*;

public class POSC45 extends Classifier {
	private static final long serialVersionUID = -8055689900166489949L;

	// the classifier
	private ClassifyPOSC45 claC45posunl;
	private ClassifyPOSC45 c45posunl[];

	// training
	public void buildClassifier(Instances data) throws Exception{
		//split data to P and U sets
		Instances posData = new Instances(data, 0);
		Instances unlData = new Instances(data, 0);
		Enumeration enu = data.enumerateInstances();
		int flg=0;
		while (enu.hasMoreElements()) {
			Instance instance = (Instance) enu.nextElement();
			if (instance.classValue() == 0 && flg < 30) {
				posData.add(instance);
				flg++;
			} else {
				instance.setClassValue(1.0);
				unlData.add(instance);
			}
		}
		// split the POS dataset
		Instances two[] = splitdata(posData);
		Instances posTrainData = two[0];
		Instances posTestData = two[1];
		// split the UN dataset
		two = splitdata(unlData);
		Instances unlTrainData = two[0];
		Instances unlTestData = two[1];
		
		c45posunl = new ClassifyPOSC45[9];
		for (int i = 0; i < 9; i++) {
			c45posunl[i] = new ClassifyPOSC45((i + 1) / 10.0);
			c45posunl[i].setDataset(posTrainData, unlTrainData);
			c45posunl[i].buildClassifier(null);
		}

		// select best DF
		double dEstimate[] = new double[9];
		for (int i = 0; i < 9; i++) {
			dEstimate[i] = evaluateBaseEstimate(c45posunl[i], posTestData, unlTestData);
			System.out.println(dEstimate[i]);
		}

		int nBestIndex = Utils.minIndex(dEstimate);

		// train the final classifier
		claC45posunl = new ClassifyPOSC45((nBestIndex + 1) / 10.0);
		claC45posunl.setDataset(posData, unlData);
		claC45posunl.buildClassifier(null);
	}

	// estimate the performance of base classifier
	double evaluateBaseEstimate(ClassifyPOSC45 c45posunl, Instances posTestData, Instances unTestData) throws Exception {
		int nPosError = 0;
		int nUnlError = 0;
		double error=0.0;

		// evaluate on POS dataset
		for (int i = 0; i < posTestData.numInstances(); i++) {
			double classlabel=-1;
			Instance sample = posTestData.instance(i);
			classlabel = c45posunl.classifyInstance(sample);
			if (!Utils.eq(classlabel, sample.classValue()))
				nPosError++;
		}

		// evaluate on UNL dataset
		for (int i = 0; i < unTestData.numInstances(); i++) {
			double classlabel=-1;
			Instance sample = unTestData.instance(i);
			classlabel = c45posunl.classifyInstance(sample);
			if (!Utils.eq(classlabel, sample.classValue()))
				nUnlError++;
		}
		error = 2.0 * (double) nPosError / posTestData.numInstances() + (double) nUnlError / unTestData.numInstances();
		return error;
	}

	// classify
	public double classifyInstance(Instance instance) throws Exception {
		return claC45posunl.classifyInstance(instance);
	}

	private Instances[] splitdata(Instances data) {
		Instances[] subsets = new Instances[2];
		subsets[0] = new Instances(data, 0);
		subsets[1] = new Instances(data, 0);
		RandomDataGenerator randGen = new RandomDataGenerator();
		int[] classselected = randGen.nextPermutation(data.numInstances(), data.numInstances() * 2 / 3);
		Arrays.sort(classselected);
		for (int i = 0; i < data.numInstances(); i++) {
			if (Arrays.binarySearch(classselected, i) >= 0)
				subsets[0].add(data.instance(i));
			else
				subsets[1].add(data.instance(i));
		}
		return subsets;
	}
}
