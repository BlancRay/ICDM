package classif.ensemble;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.random.RandomDataGenerator;

import classif.fastkmeans.DTWKNNClassifierKMeansCached;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class FKMDEClassifier extends AbstractClassifier{
	private static final long serialVersionUID = -9122604448343033527L;
	public FKMDEClassifier() {
		super();
	}
	private RandomDataGenerator randGen;
	private int nbclassifiers;
	private int K;
	private int nbbestclassifiers;
	private DTWKNNClassifierKMeansCached[] FKM;
	Instances Traindata;
	public void buildClassifier(Instances data) throws Exception {
		Traindata=new Instances(data);
		nbclassifiers=(Traindata.numInstances() / Traindata.numClasses())/2;
		K=Traindata.numInstances()/2;
//		nbbestclassifiers=nbclassifiers/10;
		FKM=new DTWKNNClassifierKMeansCached[nbclassifiers];
		randGen= new RandomDataGenerator();
		int[] nbprototypes;
		do {
			nbprototypes = randGen.nextPermutation(Traindata.numInstances() / Traindata.numClasses(), nbclassifiers);
			// System.out.println(Arrays.toString(nbprototypes));
		} while (Utils.kthSmallestValue(nbprototypes, 1) == 0);
		for (int i = 0; i < nbclassifiers; i++) {
			Instances resample = new Instances(Traindata, Traindata.numInstances());
			Random random = new Random();
			for (int j = 0; j < Traindata.numInstances(); j++) {
				int index = random.nextInt(Traindata.numInstances());
				resample.add(Traindata.instance(index));
			}

			FKM[i] = new DTWKNNClassifierKMeansCached();
			// System.out.println(nbprototypes[i]);
			FKM[i].setNbPrototypesPerClass(nbprototypes[i]);
			FKM[i].setFillPrototypes(true);
			FKM[i].buildClassifier(resample);
			System.out.println("Fast K-Means with "+nbprototypes[i]+" Prototypes build finished");
			
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
		for (int i = 0; i < FKM.length; i++) {
			DTWKNNClassifierKMeansCached kMeansCached = FKM[i];
			Evaluation evalKNNtest = new Evaluation(Traindata);
			evalKNNtest.evaluateModel(kMeansCached, KNNInstances);
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
		System.out.println(Arrays.toString(besterrorRate));
		int[] classlabel = new int[sample.numClasses()];
		for (int i = 0; i < nbbestclassifiers; i++) {
			classlabel[(int) FKM[classifiers[i]].classifyInstance(sample)] +=(int) (besterrorRate[i]*100);
		}
		
		System.out.println(Arrays.toString(classlabel));
		return Utils.maxIndex(classlabel);
	}
	
}
