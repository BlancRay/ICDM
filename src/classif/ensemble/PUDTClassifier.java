package classif.ensemble;

import java.util.Arrays;
import java.util.Random;
import org.apache.commons.math3.random.RandomDataGenerator;
import classif.BIGDT.ClassifyBigDT;
import items.MonoDoubleItemSet;
import items.Sequence;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class PUDTClassifier extends Classifier{
	private static final long serialVersionUID = -7921845058957451623L;
	public PUDTClassifier() {
		super();
	}
	private int nbclassifiers;
	private int K=5;
	private int nbbestclassifiers;
	private ClassifyBigDT[] bigDTs;
	private Instances Traindata;
	private RandomDataGenerator randGen;
	public void buildClassifier(Instances data) throws Exception {
		Traindata=new Instances(data);
		Instances posData = new Instances(Traindata, 0);
		// split data to P and U sets

		while (posData.numInstances() ==0.0) {
			int rd = new Random().nextInt(Traindata.numInstances());
			if (Traindata.instance(rd).classValue() == 0) {
				posData.add(Traindata.instance(rd));
				Traindata.delete(rd);
			}
		}
		for (int i = 0; i < Traindata.numInstances(); i++) {
			Traindata.instance(i).setClassValue(1);
		}
		double pre_dist = 0;
		for (int i = 1; i < Traindata.numInstances(); i++) {
			double[] mindist = findUnlabeledNN(posData, Traindata);
			int unlpos = Utils.minIndex(mindist);
			double now_dist = mindist[unlpos];
			if (i == 1)
				pre_dist = now_dist;
			if (Math.abs(now_dist - pre_dist) < 1) {
				Traindata.instance(unlpos).setClassValue(0.0);
				posData.add(Traindata.instance(unlpos));
				data.delete(unlpos);
				Traindata.delete(unlpos);
				pre_dist = now_dist;
			} else {
				Instance tmp = posData.lastInstance();
				tmp.setClassValue(1.0);
				Traindata.add(tmp);
				posData.delete(posData.numInstances() - 1);
				break;
			}
		}
		for (int i = 0; i < posData.numInstances(); i++) {
			Traindata.add(posData.instance(i));
		}
//		K=Traindata.numInstances()/2;
		nbclassifiers=10;
//		nbclassifiers=(Traindata.numInstances() / Traindata.numClasses())/2;
//		nbbestclassifiers=nbclassifiers/10;
		bigDTs=new ClassifyBigDT[nbclassifiers];
		for (int i = 0; i < nbclassifiers; i++) {
			Instances resample = new Instances(Traindata, Traindata.numInstances()/2);
			randGen = new RandomDataGenerator();
			int[] selected = randGen.nextPermutation(Traindata.numInstances(), Traindata.numInstances()/2);
			for (int j = 0; j < selected.length; j++) {
				resample.add(Traindata.instance(selected[j]));
			}
			bigDTs[i] = new ClassifyBigDT();
			bigDTs[i].buildClassifier(resample);
			System.out.println(i+" Decision Tree build finished");
		}
	}
	
	public double classifyInstance(Instance sample) throws Exception {
		/*
		 * Find KNN
		 * Test each kMeansCached classifier
		 * select best 10 classify with weight
		 * classify query
		 */
		FindKNN findKNN=new FindKNN(sample, Traindata, K);
		Instances KNNInstances = findKNN.KNN();
		double[] errorRate = new double[nbclassifiers];
		
		for (int i = 0; i < bigDTs.length; i++) {
			ClassifyBigDT bigDT = bigDTs[i];
			Evaluation evalKNNtest = new Evaluation(Traindata);
			evalKNNtest.evaluateModel(bigDT, KNNInstances);
			errorRate[i]=evalKNNtest.errorRate();
//			if(errorRate[i]>0.0)
//			System.out.println(errorRate[i]);
		}
		nbbestclassifiers=0;
		double avgerror=Utils.sum(errorRate)/errorRate.length;
		for (int i = 0; i < errorRate.length; i++) {
			if(Utils.smOrEq(errorRate[i], avgerror))
				nbbestclassifiers++;
		}
		
		int[] classifiers=Utils.sort(errorRate);
		double[] besterrorRate=new double[nbbestclassifiers];
		double[] errorRateCopy=errorRate.clone();
		Arrays.sort(errorRateCopy);
		besterrorRate=Arrays.copyOf(errorRateCopy, nbbestclassifiers);
		for (int i = 0; i < besterrorRate.length; i++) {
			besterrorRate[i]=(1-besterrorRate[i]);
		}
		Utils.normalize(besterrorRate);
//		System.out.println(Arrays.toString(besterrorRate));
		int[] classlabel = new int[sample.numClasses()];
		for (int i = 0; i < nbbestclassifiers; i++) {
			classlabel[(int) bigDTs[classifiers[i]].classifyInstance(sample)] +=(int) (besterrorRate[i]*100);
		}
		
//		System.out.println(Arrays.toString(classlabel));
		return Utils.maxIndex(classlabel);
	}
	
	/**
	 * Find a reliable P from U with a positive example
	 * @param p
	 * @param u
	 * @return a reliable P
	 */
	private double[] findUnlabeledNN(Instances p, Instances u) {
		double[][] updist = new double[u.numInstances()][p.numInstances()];
		double[] mindist = new double[u.numInstances()];
		for (int i = 0; i < u.numInstances(); i++) {
			for (int j = 0; j < p.numInstances(); j++) {
				Sequence[] splitsequences = new Sequence[2];
				splitsequences[0] = InsToSeq(u.instance(i));
				splitsequences[1] = InsToSeq(p.instance(j));
				updist[i][j] = splitsequences[0].distance(splitsequences[1]);
			}
			// min distance from u to p
			mindist[i] = updist[i][Utils.minIndex(updist[i])];
		}
		return mindist;
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
