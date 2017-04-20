package classif.pu;

import weka.core.*;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

import org.apache.commons.math3.random.RandomDataGenerator;

import classif.BIGDT.ClassifyBigDT;
import classif.ensemble.FindKNN;
import weka.classifiers.*;

public class TSPOSC45 extends Classifier {
	private static final long serialVersionUID = -8055689900166489949L;

	// the classifier
	private ClassifyPOSC45 claC45posunl;
	private ClassifyPOSC45[] claC45posunls;
	private ClassifyPOSC45[] c45posunl;
	public int nbpos;
	private Instances Traindata;
	private int nbclassifiers=15;
	public double r=0.3;
	private long startTime,endTime,dfTime,buildTime;

	// training
	public void buildClassifier(Instances data) throws Exception{
		startTime = System.currentTimeMillis();
		// split data to P and U sets
		Traindata=new Instances(data);
		nbpos=(int) (nbpos*r);
		Instances posData = new Instances(Traindata, nbpos);
		Instances unlData = new Instances(Traindata, Traindata.numInstances() - nbpos);
		Enumeration enu = Traindata.enumerateInstances();
		int flg = 0;
		while (enu.hasMoreElements()) {
			Instance instance = (Instance) enu.nextElement();
			if (instance.classValue() == 0 && flg < nbpos) {
				posData.add(instance);
				flg++;
			} else {
				instance.setClassValue("-1.0");
				unlData.add(instance);
			}
		}
		// split the POS dataset
/*		double[] dfj=new double[nbclassifiers];
		for (int i = 0; i < nbclassifiers; i++) {
			Instances two[] = splitdata(posData);
			Instances posTrainData = two[0];
			Instances posTestData = two[1];
			// split the UN dataset
			two = splitdata(unlData);
			Instances unlTrainData = two[0];
			Instances unlTestData = two[1];

			c45posunl = new ClassifyPOSC45[9];
			for (int j = 0; j < 9; j++) {
				c45posunl[j] = new ClassifyPOSC45((j + 1) / 10.0);
				c45posunl[j].setDataset(posTrainData, unlTrainData);
				c45posunl[j].buildClassifier(null);
			}

			// select best DF
			double dEstimate[] = new double[9];
			for (int j = 0; j < 9; j++) {
				dEstimate[j] = evaluateBaseEstimate(c45posunl[j], posTestData, unlTestData);
			}
			dfj[i] = Utils.minIndex(dEstimate);
			System.out.println("df_" + (i+1) + ":" + (dfj[i]+1)/10.0);
			System.out.println("");
		}
		double df = (Utils.mean(dfj) + 1) / 10.0;
		System.out.println("df="+df);
		endTime = System.currentTimeMillis();
		dfTime=endTime-startTime;
		System.out.println("dfTime:"+dfTime);*/

		// train the final classifier
//		claC45posunl = new ClassifyPOSC45((nBestIndex + 1) / 10.0);
//		claC45posunl.setDataset(posData, unlData);
//		claC45posunl.buildClassifier(null);
//		nbclassifiers=Math.min(Math.max((Traindata.numInstances() / Traindata.numClasses())/2,30),100);
		double df = 0.35333333333333333;
		claC45posunls = new ClassifyPOSC45[nbclassifiers];
		for (int i = 0; i < claC45posunls.length; i++) {
			startTime = System.currentTimeMillis();
			Instances resample_pos = new Instances(posData, posData.numInstances()/2);
			Instances resample_unl = new Instances(unlData, unlData.numInstances()/2);
			int[] selected_pos = new RandomDataGenerator().nextPermutation(posData.numInstances(), posData.numInstances()/2);
			int[] selected_unl = new RandomDataGenerator().nextPermutation(unlData.numInstances(), unlData.numInstances()/2);
			for (int j = 0; j < selected_pos.length; j++) {
				resample_pos.add(posData.instance(selected_pos[j]));
			}
			for (int j = 0; j < selected_unl.length; j++) {
				resample_unl.add(unlData.instance(selected_unl[j]));
			}
			
			claC45posunls[i] = new ClassifyPOSC45(df);
			claC45posunls[i].setDataset(resample_pos, resample_unl);
			claC45posunls[i].buildClassifier(null);
			endTime = System.currentTimeMillis();
			buildTime=endTime-startTime;
			System.out.println("buildTime:"+buildTime);
		}
		
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
		System.out.println("nPosError:"+nPosError+"\tnUnlError:"+nUnlError+"\tError:"+error);
		return error;
	}

	// classify
/*	public double classifyInstance(Instance sample) throws Exception {
		
		 * Find KNN
		 * Test each kMeansCached classifier
		 * select best 10 classify with weight
		 * classify query
		 
		FindKNN findKNN=new FindKNN(sample, Traindata, 5);
		Instances KNNInstances = findKNN.KNN();
		double[] errorRate = new double[nbclassifiers];
		int nbbestclassifiers;
		for (int i = 0; i < claC45posunls.length; i++) {
			ClassifyPOSC45 posc45 = claC45posunls[i];
			Evaluation evalKNNtest = new Evaluation(Traindata);
			evalKNNtest.evaluateModel(posc45, KNNInstances);
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
			classlabel[(int) claC45posunls[classifiers[i]].classifyInstance(sample)] +=besterrorRate[i];
		}
		
//		System.out.println(Arrays.toString(classlabel));
		return Utils.maxIndex(classlabel);
	}*/
	
	public double classifyInstance(Instance instance) throws Exception {
		double[] classlabel = new double[instance.numClasses()];
		for (int i = 0; i < claC45posunls.length; i++) {
			classlabel[(int) claC45posunls[i].classifyInstance(instance)] ++;
		}
		return Utils.maxIndex(classlabel);
		
//		return claC45posunl.classifyInstance(instance);
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

	public ClassifyPOSC45[] getClaC45posunls() {
		return claC45posunls;
	}
	
}
