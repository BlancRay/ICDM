package classif.pu;

import weka.core.*;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

import org.apache.commons.math3.random.RandomDataGenerator;

import items.MonoDoubleItemSet;
import items.Sequence;
import weka.classifiers.*;

public class DTWD extends Classifier {
	private static final long serialVersionUID = -8055689900166489949L;

	// the classifier
	private ClassifyPOSC45 claC45posunl;
	private ClassifyPOSC45 c45posunl[];
	private RandomDataGenerator randGen;
	protected double ratio=0.5;

	// training
	public void buildClassifier(Instances data) throws Exception{
		Instances traindata=new Instances(data);
		// split data to P and U sets
		Instances posData = new Instances(traindata, 0);
		Instances unlData = new Instances(traindata, 0);

		while (posData.numInstances() ==0.0) {
			int rd = new Random().nextInt(traindata.numInstances());
			if (traindata.instance(rd).classValue() == 0) {
				posData.add(traindata.instance(rd));
				traindata.delete(rd);
			}
		}
		for (int i = 0; i < traindata.numInstances(); i++) {
			traindata.instance(i).setClassValue(1);
		}
		for (int i = 0; i < 208 *ratio; i++) {
			int unlpos = findUnlabeledNN(posData, traindata);
			traindata.instance(unlpos).setClassValue(0);
			posData.add(traindata.instance(unlpos));
			traindata.delete(unlpos);
		}
		unlData=new Instances(traindata);
		
		
		// split the POS dataset
		Instances two[] = splitdata(posData);
		Instances posTrainData = two[0];
		Instances posTestData = two[1];
		// split the UN dataset
		two = splitdata(unlData);
		Instances unlTrainData = two[0];
		Instances unlTestData = two[1];
		
		System.out.println("Train:\tPOS:"+posTrainData.numInstances()+"\tUNL:"+unlTrainData.numInstances());
		
		c45posunl = new ClassifyPOSC45[9];
		for (int i = 0; i < 9; i++) {
			c45posunl[i] = new ClassifyPOSC45((i + 1) / 10.0);
			c45posunl[i].setDataset(posTrainData, unlTrainData);
			c45posunl[i].buildClassifier(null);
		}

		System.out.println("Test:\tPOS:"+posTestData.numInstances()+"\tUNL:"+unlTestData.numInstances());
		// select best DF
		double dEstimate[] = new double[9];
		for (int i = 0; i < 9; i++) {
			dEstimate[i] = evaluateBaseEstimate(c45posunl[i], posTestData, unlTestData);
			System.out.println("error rate of "+i+" classifier is: "+dEstimate[i]);
		}

		int nBestIndex = Utils.minIndex(dEstimate);
		System.out.println(nBestIndex);
		System.out.println("Final Classifier:\tPOS:"+posData.numInstances()+"\tUNL:"+unlData.numInstances());
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
//		System.out.println(claC45posunl.classifyInstance(instance));
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
	private int findUnlabeledNN(Instances p,Instances u){
		int unlpos=-1;
		double dist=Double.POSITIVE_INFINITY;
		for (int i = 0; i < u.numInstances(); i++) {
			for (int j = 0; j < p.numInstances(); j++) {
				Sequence[] splitsequences = new Sequence[2];
				Instance splitInstance1 = u.instance(i);
				Instance splitInstance2 = p.instance(j);
				splitsequences[0] = InsToSeq(splitInstance1);
				splitsequences[1] = InsToSeq(splitInstance2);
				double d = splitsequences[1].distance(splitsequences[0]);
				if (d < dist) {
					dist = d;
					unlpos = i;
				}
			}
		}
		return unlpos;
	}

	public void setRatio(double ratio) {
		this.ratio = ratio;
	}
	
	/**
	 * Convert Instance to Sequence
	 * @param sample Instance to be convert
	 * @return Converted Sequence
	 */
	private Sequence InsToSeq(Instance sample) {
		MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
		int shift = (sample.classIndex() == 0) ? 1 : 0;
		for (int t = 0; t < sequence.length; t++) {
			sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
		}
		return new Sequence(sequence);
	}
}
