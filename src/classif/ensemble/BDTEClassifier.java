package classif.ensemble;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.random.RandomDataGenerator;

import classif.BIGDT.ClassifyBigDT;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class BDTEClassifier extends Classifier{
	private static final long serialVersionUID = -7921845058957451623L;
	public BDTEClassifier() {
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
		Traindata=data;
//		K=Traindata.numInstances()/2;
		nbclassifiers=(Traindata.numInstances() / Traindata.numClasses())/2;
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
			bigDTs[i].buildClassifier(data);
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
	
}