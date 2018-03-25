package nwafu.dm.tsc.classif.ensemble;

import java.util.Arrays;

import org.apache.commons.math3.random.RandomDataGenerator;

import nwafu.dm.tsc.classif.BIGDT.ClassifyBigDT;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class BDTEClassifier extends AbstractClassifier{
	private static final long serialVersionUID = -2642757275050804364L;
	public BDTEClassifier() {
		super();
	}
	private int nbclassifiers = 30;
	private int K=5;
	private int nbbestclassifiers;
	private ClassifyBigDT[] bigDTs;
	private Instances Traindata;
	private RandomDataGenerator randGen;
	private long startTime,endTime,duration;
	public void buildClassifier(Instances data) throws Exception {
		Traindata=new Instances(data);
//		K=Traindata.numInstances()/2;
//		nbclassifiers=Math.min(Math.max((Traindata.numInstances() / Traindata.numClasses())/2,30),100);
//		nbclassifiers=30;
		System.out.println("nbclassifiers:"+nbclassifiers);
//		nbbestclassifiers=nbclassifiers/10;
		bigDTs=new ClassifyBigDT[nbclassifiers];
		for (int i = 0; i < nbclassifiers; i++) {
			startTime = System.currentTimeMillis();
			Instances resample = new Instances(Traindata, Traindata.numInstances()/2);
			randGen = new RandomDataGenerator();
			int[] selected = randGen.nextPermutation(Traindata.numInstances(), Traindata.numInstances()/2);
			for (int j = 0; j < selected.length; j++) {
				resample.add(Traindata.instance(selected[j]));
			}
			bigDTs[i] = new ClassifyBigDT();
			bigDTs[i].buildClassifier(resample);
			endTime = System.currentTimeMillis();
			duration = endTime - startTime;
			System.out.println(i+" Decision Tree build finished,time used:"+duration);
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
		double[] classlabel = new double[sample.numClasses()];
		for (int i = 0; i < nbbestclassifiers; i++) {
			classlabel[(int) bigDTs[classifiers[i]].classifyInstance(sample)] +=besterrorRate[i];
		}
		
//		System.out.println(Arrays.toString(classlabel));
		return Utils.maxIndex(classlabel);
	}

	public ClassifyBigDT[] getBigDTs() {
		return bigDTs;
	}
	
}
