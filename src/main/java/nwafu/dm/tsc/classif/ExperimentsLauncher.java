/*******************************************************************************
 * Copyright (C) 2014 Anonymized
 * Contributors:
 * 	Anonymized
 * 
 * This file is part of ICDM2014SUBMISSION. 
 * This is a program related to the paper "Dynamic Time Warping Averaging of 
 * Time Series allows more Accurate and Faster Classification" submitted to the
 * 2014 Int. Conf. on Data Mining.
 * 
 * ICDM2014SUBMISSION is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * ICDM2014SUBMISSION is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with ICDM2014SUBMISSION.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package nwafu.dm.tsc.classif;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
//import java.time.Duration;
import java.util.Arrays;
import java.util.Random;

import nwafu.dm.tsc.classif.BIGDT.ClassifyBigDT;
import nwafu.dm.tsc.classif.DT.ClassifyDT;
import nwafu.dm.tsc.classif.Majority.KMeans;
import nwafu.dm.tsc.classif.NNClassifier.DTWKNNClassifier;
import nwafu.dm.tsc.classif.POSC45.POSC45;
import nwafu.dm.tsc.classif.ahc.DTWKNNClassifierAHC;
import nwafu.dm.tsc.classif.dropx.DTWKNNClassifierDropOne;
import nwafu.dm.tsc.classif.dropx.DTWKNNClassifierDropThree;
import nwafu.dm.tsc.classif.dropx.DTWKNNClassifierDropTwo;
import nwafu.dm.tsc.classif.dropx.DTWKNNClassifierSimpleRank;
import nwafu.dm.tsc.classif.dropx.PrototyperSorted;
import nwafu.dm.tsc.classif.ensemble.BDTEClassifier;
import nwafu.dm.tsc.classif.ensemble.FKMDEClassifier;
import nwafu.dm.tsc.classif.ensemble.PUDTClassifier;
import nwafu.dm.tsc.classif.ensemble.StaticEnsembleClassify;
import nwafu.dm.tsc.classif.ensemble.TestThread;
import nwafu.dm.tsc.classif.fastkmeans.DTWKNNClassifierKMeansCached;
import nwafu.dm.tsc.classif.fuzzycmeans.DTWKNNClassifierFCM;
import nwafu.dm.tsc.classif.gmm.DTWKNNClassifierGmm;
import nwafu.dm.tsc.classif.gmm.EUCKNNClassifierGmm;
import nwafu.dm.tsc.classif.kmeans.DTWKNNClassifierKMeans;
import nwafu.dm.tsc.classif.kmeans.DTWProbabilisticClassifierKMeans;
import nwafu.dm.tsc.classif.kmeans.EUCKNNClassifierKMeans;
import nwafu.dm.tsc.classif.kmeans.EUCProbabilisticClassifierKMeans;
import nwafu.dm.tsc.classif.kmedoid.DTWKNNClassifierKMedoids;
import nwafu.dm.tsc.classif.newkmeans.DTWKNNClassifierNK;
import nwafu.dm.tsc.classif.pu.TSPOSC45;
import nwafu.dm.tsc.classif.pukmeans.DTWPUKMeans;
import nwafu.dm.tsc.classif.pukmeans.PUtoPN;
import nwafu.dm.tsc.classif.random.DTWKNNClassifierRandom;
import nwafu.dm.tsc.classif.sep.Unbalancecluster;
import nwafu.dm.tsc.items.ClassedSequence;
import nwafu.dm.tsc.tools.UCR2CSV;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;

public class ExperimentsLauncher {
	public static String username = "xulei";
	private File rep;
	private int nbExp;
	private int nbPrototypesMax;
	private Instances train, test;
	private PrintStream out;
	private String dataName;
	long startTime;
	long endTime;
	long duration;
	private static boolean append = false;
//	private static String datasetsDir = "./PUDATA/";
	private static String datasetsDir = "./UCR_TS_Archive_2015/";
	private static String saveoutputDir = "./save/";

	public ExperimentsLauncher(File rep, Instances train, Instances test, String dataName, int nbExp,
			int nbPrototypesMax) {
		this.rep = rep;
		this.train = train;
		this.test = test;
		this.dataName = dataName;
		this.nbExp = nbExp;
		this.nbPrototypesMax = nbPrototypesMax;
	}

	private void runDropsSteped(String algo, Prototyper prototype) {
		try {
			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();

			for (int i = 1; i <= nbPrototypesMax; i++) {
				prototype.setNbPrototypesPerClass(i);
				prototype.setFillPrototypes(false);

				startTime = System.currentTimeMillis();
				prototype.buildClassifier(train);
				endTime = System.currentTimeMillis();
				duration = endTime - startTime;

				int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(prototype.prototypes, train);

				Evaluation eval = new Evaluation(train);
				eval.evaluateModel(prototype, test);

				double testError = eval.errorRate();
				double trainError = Double.NaN;

				out.format("%s;%s;%d;%d;%.4f;%.4f;%s\n", dataName, algo, (i * train.numClasses()), duration, trainError,
						testError, Arrays.toString(classDistrib));
				out.flush();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void runDrops(String algo, Prototyper prototype) {
		try {
			for (int i = 1; i <= this.train.numInstances(); i++) {
				prototype.setNbPrototypesPerClass(i);
				prototype.setFillPrototypes(false);

				startTime = System.currentTimeMillis();
				prototype.buildClassifier(train);
				endTime = System.currentTimeMillis();
				duration = endTime - startTime;

				int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(prototype.prototypes, train);

				Evaluation eval = new Evaluation(train);
				eval.evaluateModel(prototype, test);

				double testError = eval.errorRate();
				// double trainError = prototype.predictAccuracyXVal(10);
				double trainError = Double.NaN;

				out.format("%s;%s;%d;%d;%.4f;%.4f;%s\n", dataName, algo, i, duration, trainError, testError,
						Arrays.toString(classDistrib));
				out.flush();
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void launchDrops() {
		File f = new File(rep + "/" + dataName + "_results.csv");
		// if somebody is processing it
		if (f.exists()) {
			return;
		}

		try {
			out = new PrintStream(new FileOutputStream(rep + "/" + dataName + "_results.csv", append));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		try {
			out.println(
					"dataset;algorithm;nbPrototypes;execTime;trainErrorRate;testErrorRate;prototypesPerClassDistribution");

			DTWKNNClassifierDropOne drop1 = new DTWKNNClassifierDropOne();
			drop1.setIsNbPrototypesPerClass(true);
			runDropsSteped("DROP1", drop1);
			// PrototyperUtil.savePrototypes(DTWKNNClassifierDropOne.sortedSequences,
			// rep + "/" + dataName + "_DROP1.proto");
			PrototyperSorted.reset();

			DTWKNNClassifierDropTwo drop2 = new DTWKNNClassifierDropTwo();
			drop2.setIsNbPrototypesPerClass(true);
			runDropsSteped("DROP2", drop2);
			// PrototyperUtil.savePrototypes(DTWKNNClassifierDropTwo.sortedSequences,
			// rep + "/" + dataName + "_DROP2.proto");
			PrototyperSorted.reset();

			DTWKNNClassifierDropThree drop3 = new DTWKNNClassifierDropThree();
			drop3.setIsNbPrototypesPerClass(true);
			runDropsSteped("DROP3", drop3);
			// PrototyperUtil.savePrototypes(DTWKNNClassifierDropThree.sortedSequences,
			// rep + "/" + dataName + "_DROP3.proto");
			PrototyperSorted.reset();

			DTWKNNClassifierSimpleRank sr = new DTWKNNClassifierSimpleRank();
			sr.setIsNbPrototypesPerClass(true);
			runDropsSteped("SR", sr);
			// PrototyperUtil.savePrototypes(DTWKNNClassifierSimpleRank.sortedSequences,
			// rep + "/" + dataName + "_SR.proto");
			PrototyperSorted.reset();

			out.close();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void launchKMeans() {
		try {
//			File f = new File(rep + "/" + dataName + "_results.csv");
//			// if somebody is processing it
//			if (f.exists()) {
//				return;
//			}
//
//			out = new PrintStream(new FileOutputStream(rep + "/KMeansDTW_" + "all" + "_results.csv", true));
//			out.println("dataset,algorithm,nbPrototypes,testErrorRate,trainErrorRate");
			String algo = "KMEANS";
			System.out.println(algo);
//			PrintStream outProto = new PrintStream(new FileOutputStream(rep + "/" + dataName + "_KMEANS.proto", append));

			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();
			if (nbPrototypesMax>10)
			nbPrototypesMax = 10;
			int tmp;
			tmp = nbExp;

			for (int j = 1; j <= nbPrototypesMax; j++) {
				if (j == 1)
					nbExp = 1;
				else
					nbExp = tmp;
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
					System.out.println("This is the "+n+" time.");
					DTWKNNClassifierKMeans classifierKMeans = new DTWKNNClassifierKMeans();
					classifierKMeans.setNbPrototypesPerClass(j);
					classifierKMeans.setFillPrototypes(true);

					startTime = System.currentTimeMillis();
					classifierKMeans.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;
//					Duration traintime = Duration.ofMillis(duration);
//					System.out.println(traintime);

					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifierKMeans.prototypes, train);

					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(classifierKMeans, test);
					
					double testError = eval.errorRate();
					System.out.println("TestError:"+testError+"\n");
					
//					PrototyperUtil.savePrototypes(classifierKMeans.prototypes, rep + "/" + dataName + "_KMEANS[" + j + "]_XP" + n + ".proto");

//					out.format("%s,%s,%d,%.4f\n", dataName, algo, (j * train.numClasses()), testError);
//					out.flush();
				}

			}
//			outProto.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * KMeans with Euclidean distance measure
	 */
	public void launchKMeansEUC() {
		try {
			out = new PrintStream(new FileOutputStream(rep + "/KEUC_All_results.csv", true));
//			out.println("dataset;algorithm;nbPrototypes;execTime;trainErrorRate;testErrorRate;prototypesPerClassDistribution");
			String algo = "KMEANSEUC";
			System.out.println(algo);

//			PrintStream outProto = new PrintStream(new FileOutputStream(rep + "/" + dataName + "_KMEANS.proto", append));

			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();
//			if(nbPrototypesMax>20)
			nbPrototypesMax=10;
//			if (nbPrototypesMax > 100)
//				nbPrototypesMax = 100;
			int tmp;
			tmp = nbExp;

			for (int j = 1; j <= nbPrototypesMax; j++) {
				if (j == 1)
					nbExp = 1;
				else
					nbExp = tmp;
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
					EUCKNNClassifierKMeans classifierKMeansEUC = new EUCKNNClassifierKMeans();
					classifierKMeansEUC.setNbPrototypesPerClass(j);
					classifierKMeansEUC.setFillPrototypes(true);

					startTime = System.currentTimeMillis();
					classifierKMeansEUC.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;

//					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifierKMeansEUC.prototypes, train);

					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(classifierKMeansEUC, test);
					System.out.println(eval.toSummaryString());
//					Evaluation evaltrain = new Evaluation(train);
//					evaltrain.evaluateModel(classifierKMeansEUC, train);

					double testError = eval.errorRate();
//					double trainError = evaltrain.errorRate();
//					System.out.println("TrainError:"+trainError+"\n");
					System.out.println("TestError:"+testError+"\n");
					
//					PrototyperUtil.savePrototypes(classifierKMeansEUC.prototypes, rep + "/" + dataName + "_KMEANSEUC[" + j + "]_XP" + n + ".proto");

					out.format("%s,%s,%d,%.4f\n", dataName, algo, (j * train.numClasses()), testError);
					out.flush();
				}

			}
//			outProto.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * KMeans with Bayesian theory
	 */
	public void launchKMeansProbabilistic() {
		try {
			out = new PrintStream(new FileOutputStream(rep + "/All_results.csv", true));
//			out.println("dataset;algorithm;nbPrototypes;trainErrorRate_Now;testErrorRate_Now;trainErrorRate_Before;testErrorRate_Before");
			String algo = "KMEANS";


			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();
//			if(nbPrototypesMax>2)
//			nbPrototypesMax=2;
			int tmp;
			tmp = nbExp;

			for (int j = 1; j <= nbPrototypesMax; j++) {
				if (j == 1)
					nbExp = 1;
				else
					nbExp = tmp;
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
					System.out.println("This is the "+n+" time.");
					DTWProbabilisticClassifierKMeans classifierKMeansPro = new DTWProbabilisticClassifierKMeans();
					classifierKMeansPro.setNClustersPerClass(j);

					startTime = System.currentTimeMillis();
					classifierKMeansPro.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;
					
					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifierKMeansPro.getPrototypes(), train);
					
					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(classifierKMeansPro, test);
					double testError = eval.errorRate();
					double trainError = Double.NaN;
					
					System.out.println(testError+"\n");
					PrototyperUtil.savePrototypes(classifierKMeansPro.getPrototypes(), rep + "/" + dataName + "_KMEANSPro[" + j + "]_XP" + n + ".proto");

					out.format("%s;%s;%d;%.4f\n", dataName, algo, (j * train.numClasses()), testError);
					out.flush();
				}
			}
		}catch(	Exception e){
		e.printStackTrace();
		}
	}
	/**
	 * KMeans with Bayesian theory and Euclidean distance measure
	 */
	public void launchKMeansProbabilisticEUC() {
		try {
//			out = new PrintStream(new FileOutputStream(rep + "/KPEUC_All_results.csv", true));
//			out.println("dataset;algorithm;nbPrototypes;trainErrorRate_Now;testErrorRate_Now;trainErrorRate_Before;testErrorRate_Before");
			String algo = "KMEANSProbabilisticEUC";
			System.out.println(algo);
			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();
//			if(nbPrototypesMax>20)
			nbPrototypesMax=1;
//			if (nbPrototypesMax > 100)
//				nbPrototypesMax = 100;
			int tmp;
			tmp = nbExp;

			for (int j = 1; j <= nbPrototypesMax; j++) {
				if (j == 1)
					nbExp = 1;
				else
					nbExp = tmp;
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
					System.out.println("This is the "+n+" time.");
					EUCProbabilisticClassifierKMeans classifierKMeansProEUC = new EUCProbabilisticClassifierKMeans();
					classifierKMeansProEUC.setNClustersPerClass(j);

					startTime = System.currentTimeMillis();
					classifierKMeansProEUC.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;
					
//					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifierKMeansProEUC.getPrototypes(), train);
					
					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(classifierKMeansProEUC, test);
					Evaluation evaltrain = new Evaluation(train);
					evaltrain.evaluateModel(classifierKMeansProEUC, train);
					
					double testError = eval.errorRate();
					double trainError = evaltrain.errorRate();
					
					System.out.println("TrainError:"+trainError+"\n");
					System.out.println("TestError:"+testError+"\n");
//					PrototyperUtil.savePrototypes(classifierKMeansProEUC.getPrototypes(), rep + "/" + dataName + "_KMEANSProEUC[" + j + "]_XP" + n + ".proto");

//					out.format("%s;%s;%d;%.4f;%.4f\n", dataName, algo, (j * train.numClasses()), trainError,testError);
//					out.flush();
				}
			}
		}catch(	Exception e){
		e.printStackTrace();
		}
	}
	
	public void launchNewKMeans() {
		try {
//			out = new PrintStream(new FileOutputStream(rep + "/GMMDTW_"+dataName+"_results.csv", true));
//			out.println("dataset;algorithm;nbPrototypes;testErrorRate");
			String algo = "GMM";
			System.out.println(algo);

//			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();
//			if(nbPrototypesMax>10)
			nbPrototypesMax=10;
			int tmp;
			tmp = nbExp;

			for (int j = 1; j <= nbPrototypesMax; j++) {
				if (j == 1)
					nbExp = 1;
				else
					nbExp = tmp;
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
					System.out.println("This is the "+n+" time.");
					DTWKNNClassifierNK classifierGmm = new DTWKNNClassifierNK();
					classifierGmm.setNClustersPerClass(j);

					startTime = System.currentTimeMillis();
					classifierGmm.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;
					
//					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifierGmm.getPrototypes(), train);
					
					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(classifierGmm, test);
					double testError = eval.errorRate();
					double trainError = Double.NaN;
					
					System.out.println("TestError:"+testError+"\n");
//					PrototyperUtil.savePrototypes(classifierGmm.getPrototypes(), rep + "/" + dataName + "_GMM[" + j + "]_XP" + n + ".proto");

//					out.format("%s,%s,%d,%.4f\n", dataName, algo, (j * train.numClasses()), testError);
//					out.flush();
				}
			}
		}catch(	Exception e){
		e.printStackTrace();
		}
	}
	/**
	 * Gaussian mixture model instead of KMeans
	 */
	public void launchGmm() {
		try {
//			out = new PrintStream(new FileOutputStream(rep + "/GMMDTW_"+dataName+"_results.csv", true));
//			out.println("dataset;algorithm;nbPrototypes;testErrorRate");
			String algo = "GMM";
			System.out.println(algo);

//			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();
//			if(nbPrototypesMax>10)
			nbPrototypesMax=10;
			int tmp;
			tmp = nbExp;

			for (int j = 1; j <= nbPrototypesMax; j++) {
				if (j == 1)
					nbExp = 1;
				else
					nbExp = tmp;
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
					System.out.println("This is the "+n+" time.");
					DTWKNNClassifierGmm classifierGmm = new DTWKNNClassifierGmm();
					classifierGmm.setNClustersPerClass(j);

					startTime = System.currentTimeMillis();
					classifierGmm.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;
					
//					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifierGmm.getPrototypes(), train);
					
					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(classifierGmm, test);
					double testError = eval.errorRate();
					double trainError = Double.NaN;
					
					System.out.println("TestError:"+testError+"\n");
//					PrototyperUtil.savePrototypes(classifierGmm.getPrototypes(), rep + "/" + dataName + "_GMM[" + j + "]_XP" + n + ".proto");

//					out.format("%s,%s,%d,%.4f\n", dataName, algo, (j * train.numClasses()), testError);
//					out.flush();
				}
			}
		}catch(	Exception e){
		e.printStackTrace();
		}
	}
	/**
	 * GMM in Euclidean distance measure
	 */
	public void launchGmmEUC() {
		try {
//			out = new PrintStream(new FileOutputStream(rep + "/GMMEUC_"+dataName+"_results.csv", true));
//			out.println("dataset;algorithm;nbPrototypes;trainErrorRate_Now;testErrorRate_Now;trainErrorRate_Before;testErrorRate_Before");
			String algo = "GMMEUC";
			System.out.println(algo);
//			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();
//			if (nbPrototypesMax > 10)
				nbPrototypesMax = 3;
//			if (nbPrototypesMax > 100)
//				nbPrototypesMax = 100;
			int tmp;
			tmp = nbExp;

			for (int j = 1; j <= nbPrototypesMax; j++) {
				if (j == 1)
					nbExp = 1;
				else
					nbExp = tmp;
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
					System.out.println("This is the "+n+" time.");
					EUCKNNClassifierGmm classifierGmmEUC = new EUCKNNClassifierGmm();
					classifierGmmEUC.setNClustersPerClass(j);

					startTime = System.currentTimeMillis();
					classifierGmmEUC.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;
					
					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifierGmmEUC.getPrototypes(), train);
					
					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(classifierGmmEUC, test);
//					Evaluation evaltrain = new Evaluation(train);
//					evaltrain.evaluateModel(classifierGmmEUC, train);
					
					double testError = eval.errorRate();
//					double trainError = evaltrain.errorRate();
//					System.out.println("TrainError:"+trainError+"\n");
					System.out.println("TestError:"+testError+"\n");
//					PrototyperUtil.savePrototypes(classifierGmmEUC.getPrototypes(), rep + "/" + dataName + "_GMMEUC[" + j + "]_XP" + n + ".proto");

//					out.format("%s,%s,%d,%.4f\n", dataName, algo, (j * train.numClasses()), testError);
//					out.flush();
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * use fuzzy-C-Means instead of KMeans 
	 */
	public void launchFCM() {
		try {
			out = new PrintStream(new FileOutputStream(rep + "/FCMDTW_"+dataName+"_results.csv", true));
//			out.println("dataset,algorithm,nbPrototypes,testErrorRate");
			String algo = "FCM";
			System.out.println(algo);

//			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();
//			if(nbPrototypesMax>10)
			nbPrototypesMax=10;
			int tmp;
			tmp = nbExp;

			for (int j = 1; j <= nbPrototypesMax; j++) {
				if (j == 1)
					nbExp = 1;
				else
					nbExp = tmp;
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
					System.out.println("This is the "+n+" time.");
					DTWKNNClassifierFCM classifierFCM = new DTWKNNClassifierFCM();
					classifierFCM.setNClustersPerClass(j);

					startTime = System.currentTimeMillis();
					classifierFCM.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;
					
//					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifierGmm.getPrototypes(), train);
					
					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(classifierFCM, test);
					double testError = eval.errorRate();
					double trainError = Double.NaN;
					
					System.out.println("TestError:"+testError+"\n");
//					PrototyperUtil.savePrototypes(classifierGmm.getPrototypes(), rep + "/" + dataName + "_GMM[" + j + "]_XP" + n + ".proto");

					out.format("%s,%s,%d,%.4f\n", dataName, algo, (j * train.numClasses()), testError);
					out.flush();
				}
			}
		}catch(	Exception e){
		e.printStackTrace();
		}
	}
	
	public void launchRandom() {
		try {
			File f = new File(rep + "/" + dataName + "_results.csv");
			// if somebody is processing it
			if (f.exists()) {
				return;
			}

			out = new PrintStream(new FileOutputStream(rep + "/" + dataName + "_results.csv", append));
			out.println(
					"dataset;algorithm;nbPrototypes;execTime;trainErrorRate;testErrorRate;prototypesPerClassDistribution");
			String algo = "RANDOM";

			PrintStream outProto = new PrintStream(
					new FileOutputStream(rep + "/" + dataName + "_RANDOM.proto", append));

			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();

			int indexLastChange = 0;
			double errorRateLastChange = 1.0;
			ClassedSequence[] testSequences = Prototyper.convertWekaSetToClassedSequence(test);
			DTWKNNClassifierRandom classifier = new DTWKNNClassifierRandom();
			for (int n = 0; n < nbExp; n++) {
				for (int j = 1; j <= nbPrototypesMax && (j - indexLastChange) < 10; j++) {
					classifier.setNbPrototypesPerClass(j);
					classifier.setFillPrototypes(false);

					startTime = System.currentTimeMillis();
					classifier.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;

					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifier.prototypes, train);

					double testError = classifier.evalErrorRate(testSequences);
					double trainError = Double.NaN;

					// PrototyperUtil.savePrototypes(classifier.prototypes,rep+"/"+dataName+"_KMEANS["+j+"]_XP"+n+".proto");

					out.format("%s;%s;%d;%d;%.4f;%.4f;%s\n", dataName, algo, (j * train.numClasses()), duration,
							trainError, testError, Arrays.toString(classDistrib));
					out.flush();

					outProto.format("%s\n", classifier.indexPrototypeInTrainData);
					outProto.flush();

					if (errorRateLastChange != testError) {
						errorRateLastChange = testError;
						indexLastChange = j;
					}
				}

			}
			outProto.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void launchKMedoids() {
		try {
			File f = new File(rep + "/" + dataName + "_results.csv");
			// if somebody is processing it
			if (f.exists()) {
				return;
			}

			out = new PrintStream(new FileOutputStream(rep + "/" + dataName + "_results.csv", append));
			out.println(
					"dataset;algorithm;nbPrototypes;execTime;trainErrorRate;testErrorRate;prototypesPerClassDistribution");
			String algo = "KMEDOIDS";

			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();

			for (int j = 1; j <= nbPrototypesMax; j++) {
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
					DTWKNNClassifierKMedoids classifier = new DTWKNNClassifierKMedoids();
					classifier.setNbPrototypesPerClass(j);
					classifier.setFillPrototypes(true);

					startTime = System.currentTimeMillis();
					classifier.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;

					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifier.prototypes, train);

					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(classifier, test);

					double testError = eval.errorRate();
					double trainError = classifier.predictAccuracyXVal(10);

					PrototyperUtil.savePrototypes(classifier.prototypes,
							rep + "/" + dataName + "_KMEDOIDS[" + j + "]_XP" + n + ".proto");

					out.format("%s;%s;%d;%d;%.4f;%.4f;%s\n", dataName, algo, (j * train.numClasses()), duration,
							trainError, testError, Arrays.toString(classDistrib));
					out.flush();
					// deterministic
					if (j == 1) {
						break;
					}
				}

			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void launchAHC() {
		try {
			File f = new File(rep + "/" + dataName + "_results.csv");
			// if somebody is processing it
			if (f.exists()) {
				return;
			}

			out = new PrintStream(new FileOutputStream(rep + "/" + dataName + "_results.csv", append));
			out.println(
					"dataset;algorithm;nbPrototypes;execTime;trainErrorRate;testErrorRate;prototypesPerClassDistribution");
			String algo = "AHC";

			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();

			ClassedSequence[] testSequences = Prototyper.convertWekaSetToClassedSequence(test);

			int indexLastChange = 0;
			double errorRateLastChange = 1.0;

			DTWKNNClassifierAHC classifier = new DTWKNNClassifierAHC();
			for (int j = 1; j <= nbPrototypesMax && (j - indexLastChange) < 10; j++) {
				System.out.println("nbPrototypes=" + j);
				classifier.setNbPrototypesPerClass(j);
				classifier.setFillPrototypes(false);

				startTime = System.currentTimeMillis();
				classifier.buildClassifier(train);
				endTime = System.currentTimeMillis();
				duration = endTime - startTime;

				int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifier.prototypes, train);

				double testError = classifier.evalErrorRate(testSequences);
				double trainError = Double.NaN;

				PrototyperUtil.savePrototypes(classifier.prototypes,
						rep + "/" + dataName + "_AHC[" + j + "]_XP" + 0 + ".proto");

				out.format("%s;%s;%d;%d;%.4f;%.4f;%s\n", dataName, algo, (j * train.numClasses()), duration, trainError,
						testError, Arrays.toString(classDistrib));
				out.flush();

				if (errorRateLastChange != testError) {
					errorRateLastChange = testError;
					indexLastChange = j;
				}

			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * each cluster has different number of prototypes
	 * use fast KMeans to cluster
	 */
	public void launchUnbalanceCluster() {
		try {
			nbPrototypesMax = 10;
			int[] bestprototypes= new int[train.numClasses()];
			double lowerror=1.0;
			for (int j = 1; j <= nbPrototypesMax; j++) {
				int[] nbPrototypesPerClass = new int[train.numClasses()];
				for (int i = 0; i < train.numClasses(); i++) {
					nbPrototypesPerClass[i] = j;
				}
				double errorBefore = 1;
				double errorNow = 1;
				int flag = 0;
				do {
					Unbalancecluster classifierseq = new Unbalancecluster();
					classifierseq.setNbPrototypesPerClass(nbPrototypesPerClass);
					System.out.println(Arrays.toString(nbPrototypesPerClass));
//					classifierseq.buildClassifier(train);
					Evaluation evalcv = new Evaluation(train);
					Random rand = new Random(1);
					evalcv.crossValidateModel(classifierseq, train, 10, rand);
//					errorNow = classifierseq.predictAccuracyXVal(10);
					errorNow = evalcv.errorRate();
					System.out.println("errorBefore " + errorBefore);
					System.out.println("errorNow " + errorNow);
					if (errorNow < errorBefore) {
						nbPrototypesPerClass[flag]++;
						errorBefore = errorNow;
					} else {
						nbPrototypesPerClass[flag]--;
						flag++;
						if (flag >= nbPrototypesPerClass.length)
							break;
						nbPrototypesPerClass[flag]++;
					}
				} while (flag < nbPrototypesPerClass.length);
//				System.out.println("\nbest nbPrototypesPerClass " + Arrays.toString(nbPrototypesPerClass));
				double testError = 0;
				for (int n = 0; n < nbExp; n++) {
					Unbalancecluster classifier = new Unbalancecluster();
					classifier.setNbPrototypesPerClass(nbPrototypesPerClass);
					classifier.buildClassifier(train);
					Evaluation evaltest = new Evaluation(train);
					evaltest.evaluateModel(classifier, test);
					testError += evaltest.errorRate();
				}
				double avgTestError=testError / nbExp;
				System.out.println(avgTestError);
				if(avgTestError<lowerror){
					bestprototypes=nbPrototypesPerClass;
					lowerror=avgTestError;
				}
			}
			System.out.println("Best prototypes:" + Arrays.toString(bestprototypes) + "\n");
			System.out.println("Best errorRate:" + lowerror + "\n");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void launchFSKMeans() {
		try {
//			File f = new File(rep + "/" + dataName + "_results.csv");
//			// if somebody is processing it
//			if (f.exists()) {
//				return;
//			}
//
//			out = new PrintStream(new FileOutputStream(rep + "/FastKMeansDTW_" + dataName + "_results.csv", true));
//			out.println("dataset,algorithm,nbPrototypes,testErrorRate,trainErrorRate");
			String algo = "FastKMEANS";
			System.out.println(algo);
//			PrintStream outProto = new PrintStream(new FileOutputStream(rep + "/" + dataName + "_KMEANS.proto", append));

			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();
			if (nbPrototypesMax>100)
			nbPrototypesMax = 200;
			int tmp;
			tmp = nbExp;
			double[] avgerror=new double[5];
			double[] avgf1=new double[5];
//			double[] trainrctmp = new double[5];
//			double[] testrctmp = new double[5];
//			double[] cvrctmp = new double[5];
//			boolean stopflag=false;
			for (int j = 5; j <= nbPrototypesMax; j+=5) {
//				double[] trainrc = new double[5];
//				double[] testrc = new double[5];
//				double[] cvrc = new double[5];
				if (j == 1)
					nbExp = 1;
				else
					nbExp = tmp;
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
//					System.out.println("This is the "+n+" time.");
					DTWKNNClassifierKMeansCached classifierKMeans = new DTWKNNClassifierKMeansCached();
					classifierKMeans.setNbPrototypesPerClass(j);
					classifierKMeans.setFillPrototypes(true);

					startTime = System.currentTimeMillis();
					classifierKMeans.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;

					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifierKMeans.prototypes, train);

					Evaluation evaltest = new Evaluation(train);
					evaltest.evaluateModel(classifierKMeans, test);
					avgerror[n]=evaltest.errorRate();
					avgf1[n]=evaltest.fMeasure(0);
					System.out.println(evaltest.toMatrixString());
//					Evaluation evaltrain = new Evaluation(train);
//					evaltrain.evaluateModel(classifierKMeans, train);
					
					/*DTWKNNClassifierKMeansCached KMeans = new DTWKNNClassifierKMeansCached();
					KMeans.setNbPrototypesPerClass(j);
					KMeans.setFillPrototypes(true);
					Evaluation evalcv = new Evaluation(train);
					Random rand = new Random(1);
					evalcv.crossValidateModel(KMeans, train, 10, rand);
					double CVError = evalcv.errorRate();
					System.out.println("CVError:"+CVError+"\n");*/

//					PrototyperUtil.savePrototypes(classifierKMeans.prototypes, rep + "/" + dataName + "_KMEANS[" + j + "]_XP" + n + ".proto");

//					out.format("%s,%s,%d,%.4f,%.4f,%.4f\n", dataName, algo, (j * train.numClasses()), testError,CVError,trainError);
//					out.flush();
//					trainrc[n]=trainError;
//					testrc[n]=testError;
//					cvrc[n]=CVError;
//					if (n == 4) {
//						if (j == 1) {
//							trainrctmp = trainrc;
//							testrctmp = testrc;
//							cvrctmp = cvrc;
//						} else {
//							if (Arrays.equals(trainrc, trainrctmp) && Arrays.equals(testrc, testrctmp)
//									&& Arrays.equals(cvrc, cvrctmp)) {
//								System.out.println("Stable at " + j);
//								stopflag=true;
//							} else {
//								trainrctmp = trainrc;
//								testrctmp = testrc;
//								cvrctmp = cvrc;
//							}
//						}
//					}
				}
				System.out.println("TestError:"+Utils.mean(avgerror)+"\tF-Measures:"+Utils.mean(avgf1)+"\n");
//				if(stopflag==true)
//					break;
			}
//			outProto.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	public void launchAllKMeans() {
		try {
			String algo = "AllKMEANS";
			System.out.println(algo);

			nbPrototypesMax = this.train.numInstances();
			int prototypestart = this.train.numClasses();
			for (int j = prototypestart; j <= nbPrototypesMax; j++) {
				double testError=0.0;
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
					System.out.println("This is the "+n+" time.");
					KMeans classifier = new KMeans();
					classifier.setNbPrototypes(j);
					classifier.buildClassifier(train);
					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(classifier, test);
					testError += eval.errorRate();
					System.out.println("TestError:"+eval.errorRate()+"\n");
				}
				System.out.println("TestError of average:"+(testError/nbExp)+"\n");

			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Decision Tree
	 */
	public void launchDT() {
		if (train.numClasses() == 2) {

			try {
				String algo = "DecisionTree";
				System.out.println(algo);

				double testError = 0.0;
				ClassifyDT dt = new ClassifyDT();
				dt.buildClassifier(train);
				System.out.println("\nClassify test sets:\n");
				Evaluation eval = new Evaluation(train);
				eval.evaluateModel(dt, test);
				testError = eval.errorRate();
				System.out.println("TestError:" + testError + "\n");
				System.out.println(eval.toSummaryString());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	public void launchJ48() {
		try {
			String algo = "J48";
			System.out.println(algo);

			double testError = 0.0;
			J48 dt = new J48();
			startTime = System.currentTimeMillis();
			dt.buildClassifier(train);
			endTime = System.currentTimeMillis();
			duration = endTime - startTime;
			System.out.println("Training time:" + duration);
			Evaluation eval = new Evaluation(train);
			startTime = System.currentTimeMillis();
			eval.evaluateModel(dt, test);
			endTime = System.currentTimeMillis();
			duration = endTime - startTime;
			System.out.println("Testing time:" + duration);
			testError = eval.errorRate();
			System.out.println("TestError:" + testError + "\nFMeasures:" + eval.fMeasure(0));
			System.out.println(dt.toSummaryString());
			System.out.println(dt.graph());
			System.out.println(eval.toSummaryString());

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Decision Forest
	 */
	public void launchBigDT() {
		try {
//			out = new PrintStream(new FileOutputStream(rep + "/DT_" + dataName + "_results.csv", true));
			String algo = "BigDT_Forest";
			System.out.println(algo);
			double testError = 0.0;
			startTime = System.currentTimeMillis();
			ClassifyBigDT dt = new ClassifyBigDT();
			dt.buildClassifier(train);
			endTime = System.currentTimeMillis();
			duration = endTime - startTime;
//			Duration traintime = Duration.ofMillis(duration);
//			System.out.println(traintime);
			startTime = System.currentTimeMillis();
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(dt, test);
			testError = eval.errorRate();
			endTime = System.currentTimeMillis();
			duration = endTime - startTime;
//			Duration testtime = Duration.ofMillis(duration);
			System.out.println(duration);
			System.out.println("TestError:" + testError + "\n");
//			System.out.println(eval.toSummaryString());
//			out.format("%s,%.4f\n", dataName,  testError);
//			out.flush();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void launchStaticEnsemble() {
		try {
			String algo = "StaticEnsemble";
			System.out.println(algo);

			double testError = 0.0;
			double testError_DT = 0.0;
			double testError_FKM_4 = 0.0;
			double testError_FKM_10 = 0.0;
//			double testError_KMeans = 0.0;
			startTime = System.currentTimeMillis();
			StaticEnsembleClassify staticensembleClassify = new StaticEnsembleClassify();
			staticensembleClassify.buildClassifier(train);
			endTime = System.currentTimeMillis();
			duration = endTime - startTime;
//			Duration traintime = Duration.ofMillis(duration);
//			System.out.println(traintime);
			
			Evaluation eval_FKM_4 = new Evaluation(train);
			eval_FKM_4.evaluateModel(staticensembleClassify.getFkm_4(), test);
			testError_FKM_4 = eval_FKM_4.errorRate();
			staticensembleClassify.setWeight_fkm_4(testError_FKM_4);
			System.out.println("TestError of FKM_4:" + testError_FKM_4 + "\n");
			
//			Evaluation eval_KMeans = new Evaluation(train);
//			eval_KMeans.evaluateModel(ensembleClassify.getKMeans(), test);
//			testError_KMeans = eval_KMeans.errorRate();
//			ensembleClassify.setWeight_kmeans(testError_KMeans);
//			System.out.println("TestError of KMeans:" + testError_KMeans + "\n");
			
			Evaluation eval_FKM_10 = new Evaluation(train);
			eval_FKM_10.evaluateModel(staticensembleClassify.getFkm_10(), test);
			testError_FKM_10 = eval_FKM_10.errorRate();
			staticensembleClassify.setWeight_fkm_10(testError_FKM_10);
			System.out.println("TestError of FKM_10:" + testError_FKM_10 + "\n");

			Evaluation eval_DT = new Evaluation(train);
			eval_DT.evaluateModel(staticensembleClassify.getDt(), test);
			testError_DT = eval_DT.errorRate();
			staticensembleClassify.setWeight_dt(testError_DT);
			System.out.println("TestError of DT:" + testError_DT + "\n");

			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(staticensembleClassify, test);
			testError = eval.errorRate();
			System.out.println("TestError of Ensemble:" + testError + "\n");
			System.out.println(eval.toSummaryString());

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void launchDTWKNNClassifier() {
		try {
			String algo = "1NNDTWClassifier";
			System.out.println(algo);
			startTime = System.currentTimeMillis();
			DTWKNNClassifier KNNClassifier = new DTWKNNClassifier();
			KNNClassifier.buildClassifier(train);
			endTime = System.currentTimeMillis();
			duration = endTime - startTime;
			System.out.println("Training time:"+duration);
			startTime = System.currentTimeMillis();
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(KNNClassifier, test);
			endTime = System.currentTimeMillis();
			duration = endTime - startTime;
			System.out.println("Testing time:" + duration);
			double error = eval.errorRate();
			System.out.println("TestError:" + error + "\n" + eval.fMeasure(0));
			System.out.println(eval.toMatrixString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Dynamic Ensemble Decision Forest
	 * Dynamic Ensemble Fast KMeans with different prototypes
	 */
	public void launchDynamicEnsemble() {
		int method = 0;
		switch (method) {
		case 0:
			try {
				String algo = "BigDTDynamicEnsemble";
				System.out.println(algo);

				double testError = 0.0;
				startTime = System.currentTimeMillis();
				BDTEClassifier dynamicEnsembleClassify = new BDTEClassifier();
				dynamicEnsembleClassify.buildClassifier(train);
				endTime = System.currentTimeMillis();
				duration = endTime - startTime;
				System.out.println("Training time:"+duration);
				startTime = System.currentTimeMillis();
				Evaluation eval = new Evaluation(train);
				eval.evaluateModel(dynamicEnsembleClassify, test);
				endTime = System.currentTimeMillis();
				duration = endTime - startTime;
				System.out.println("Testing time:"+duration);
				testError = eval.errorRate();
				System.out.println("TestError:" + testError + "\nFMeasures:" + eval.fMeasure(0));
				System.out.println(eval.toMatrixString());
				int nbclassifiers=dynamicEnsembleClassify.getBigDTs().length;
				double[] error = new double[nbclassifiers];
				double[] fmeasures = new double[nbclassifiers];
				double[] testtime = new double[nbclassifiers];
				for (int i = 0; i < nbclassifiers; i++) {
					startTime = System.currentTimeMillis();
					Evaluation evaleach = new Evaluation(train);
					evaleach.evaluateModel(dynamicEnsembleClassify.getBigDTs()[i], test);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;
					error[i]=evaleach.errorRate();
					fmeasures[i]=evaleach.fMeasure(0);
					testtime[i]=duration;
					System.out.println(i+"\tDT TestError:" + error[i] + "\tFMeasures:" + fmeasures[i]);
				}
				System.out.println("Average Error:"+Utils.mean(error)+"\tFMeasure:"+Utils.mean(fmeasures)+"\tTesttime:"+Utils.mean(testtime));
			} catch (Exception e) {
				e.printStackTrace();
			}
			break;

		case 1:
			try {
				String algo = "FKMDynamicEnsemble";
				System.out.println(algo);

				double testError = 0.0;
				FKMDEClassifier dynamicEnsembleClassify = new FKMDEClassifier();
				dynamicEnsembleClassify.buildClassifier(train);
				Evaluation eval = new Evaluation(train);
				eval.evaluateModel(dynamicEnsembleClassify, test);
				testError = eval.errorRate();
				System.out.println("TestError:" + testError + "\n");
				System.out.println(eval.toSummaryString());
			} catch (Exception e) {
				e.printStackTrace();
			}
			break;
		}
	}
	/**
	 * Decision Tree for PU Learning
	 * @param ratio
	 */
	public void launchPU() {
		try {
//			out = new PrintStream(new FileOutputStream(rep + "/DT_" + dataName + "_results.csv", true));
			String algo = "PU";
			System.out.println(algo);
				double testError = 0.0;
				TSPOSC45 tsposc45=new TSPOSC45();
				tsposc45.nbpos=nbPrototypesMax;
				tsposc45.r=0.3;
				startTime = System.currentTimeMillis();
				tsposc45.buildClassifier(train);
				endTime = System.currentTimeMillis();
				duration = endTime - startTime;
//				Duration traintime = Duration.ofMillis(duration);
				System.out.println("Training time:"+duration);
				Evaluation eval = new Evaluation(train);
				startTime = System.currentTimeMillis();
				eval.evaluateModel(tsposc45, test);
				endTime = System.currentTimeMillis();
				duration = endTime - startTime;
//				Duration testtime = Duration.ofMillis(duration);
				System.out.println("Test time:"+duration);
				System.out.println(eval.toMatrixString());
				System.out.println("FMeasure:" + eval.fMeasure(0));
				testError = eval.errorRate();
				System.out.println("TestError:" + testError + "\n");
				
				int nbclassifiers=tsposc45.getClaC45posunls().length;
				double[] error = new double[nbclassifiers];
				double[] fmeasures = new double[nbclassifiers];
				double[] testtime = new double[nbclassifiers];
				for (int i = 0; i < nbclassifiers; i++) {
					Evaluation evaleach = new Evaluation(train);
					startTime = System.currentTimeMillis();
					evaleach.evaluateModel(tsposc45.getClaC45posunls()[i], test);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;
					error[i]=evaleach.errorRate();
					fmeasures[i]=evaleach.fMeasure(0);
					testtime[i]=duration;
					System.out.println(i+"\tDT TestError:" + error[i] + "\tFMeasures:" + fmeasures[i] + "\tTest Time:" + testtime[i]);
				}
				System.out.println("Average Error:"+Utils.mean(error)+"\tFMeasure:"+Utils.mean(fmeasures)+"\tTesttime:"+Utils.mean(testtime));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	public void launchPOSC45() {
		try {
//			out = new PrintStream(new FileOutputStream(rep + "/DT_" + dataName + "_results.csv", true));
			String algo = "PU";
			System.out.println(algo);
				double testError = 0.0;
				POSC45 posc45=new POSC45();
				posc45.nbpos=nbPrototypesMax;
				posc45.r=0.3;
				startTime = System.currentTimeMillis();
				posc45.buildClassifier(train);
				endTime = System.currentTimeMillis();
				duration = endTime - startTime;
//				Duration traintime = Duration.ofMillis(duration);
				System.out.println("Training time:"+duration);
				Evaluation eval = new Evaluation(train);
				startTime = System.currentTimeMillis();
				eval.evaluateModel(posc45, test);
				endTime = System.currentTimeMillis();
				duration = endTime - startTime;
//				Duration testtime = Duration.ofMillis(duration);
				System.out.println("Test time:"+duration);
				System.out.println(eval.toMatrixString());
				System.out.println("FMeasure:" + eval.fMeasure(0));
				testError = eval.errorRate();
				System.out.println("TestError:" + testError + "\n");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * KMeans for PU Learning
	 */
	public void launchPUKMeans() {
		try {
//			File f = new File(rep + "/" + dataName + "_results.csv");
//			// if somebody is processing it
//			if (f.exists()) {
//				return;
//			}
//
//			out = new PrintStream(new FileOutputStream(rep + "/KMeansDTW_" + "all" + "_results.csv", true));
//			out.println("dataset,algorithm,nbPrototypes,testErrorRate,trainErrorRate");
			String algo = "PUKMEANS";
			System.out.println(algo);
//			PrintStream outProto = new PrintStream(new FileOutputStream(rep + "/" + dataName + "_KMEANS.proto", append));

			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();
			if (nbPrototypesMax>100)
			nbPrototypesMax = 200;
			int tmp;
			tmp = nbExp;
			double[] avgerror=new double[5];
			double[] avgf1=new double[5];

			for (int j = 5; j <= nbPrototypesMax; j+=5) {
				if (j <= 1){
					nbExp = 1;
					j=1;
				}
				else
					nbExp = tmp;
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
//					System.out.println("This is the "+n+" time.");
					DTWPUKMeans classifierKMeans = new DTWPUKMeans();
					classifierKMeans.setNbClustersinUNL(j);
					startTime = System.currentTimeMillis();
					classifierKMeans.setDf((n+1)*0.05);
					classifierKMeans.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;
//					Duration traintime = Duration.ofMillis(duration);
//					System.out.println(traintime);
					Evaluation eval = new Evaluation(train);
//					StringBuffer forPredictionsPrinting = new StringBuffer();
//					eval.evaluateModel(classifierKMeans, train, forPredictionsPrinting, null, false);
					eval.evaluateModel(classifierKMeans, test);
					System.out.println(eval.toMatrixString());
					avgerror[n]=eval.errorRate();
					avgf1[n]=eval.fMeasure(0);
					System.out.println(avgerror[n]+"\t"+avgf1[n]);
					
//					PrototyperUtil.savePrototypes(classifierKMeans.prototypes, rep + "/" + dataName + "_KMEANS[" + j + "]_XP" + n + ".proto");

//					out.format("%s,%s,%d,%.4f\n", dataName, algo, (j * train.numClasses()), testError);
//					out.flush();
				}
				System.out.println("TestError:"+Utils.mean(avgerror)+"\tF-Measures:"+Utils.mean(avgf1)+"\n");
			}
//			outProto.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	/**
	 * Decision Tree Dynamic Ensemble for PU Learning
	 */
	public void launchPUDTEnsemble() {
		try {
			String algo = "PUDTDynamicEnsemble";
			System.out.println(algo);
			double testError = 0.0;
			PUDTClassifier dynamicEnsembleClassify = new PUDTClassifier();
			dynamicEnsembleClassify.buildClassifier(train);
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(dynamicEnsembleClassify, test);
			testError = eval.errorRate();
			System.out.println("TestError:" + testError + "\n");
			System.out.println(eval.fMeasure(0));
			System.out.println(eval.toMatrixString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Decision Tree Dynamic Ensemble for PU Learning
	 */
	public void launchPUGMM() {
		try {
			String algo = "PUGMM";
			System.out.println(algo);
			double testError = 0.0;
			PUtoPN pugmm = new PUtoPN();
			pugmm.buildClassifier(train);
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(pugmm, test);
			testError = eval.errorRate();
			System.out.println("TestError:" + testError + "\n");
			System.out.println(eval.fMeasure(0));
			System.out.println(eval.toMatrixString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void launchTestThread() {
		try {
			String algo = "TestThread";
			System.out.println(algo);

			double testError = 0.0;
			startTime = System.currentTimeMillis();
			TestThread testThread = new TestThread();
			testThread.buildClassifier(train);
			endTime = System.currentTimeMillis();
			duration = endTime - startTime;
			System.out.println("Training time:" + duration);
			startTime = System.currentTimeMillis();
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(testThread, test);
			endTime = System.currentTimeMillis();
			duration = endTime - startTime;
			System.out.println("Testing time:" + duration);
			testError = eval.errorRate();
			System.out.println("TestError:" + testError + "\n" + eval.fMeasure(0));
			System.out.println(eval.toMatrixString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
//		String dataname="50words";
//		datasetsDir = "./PUDATA/";
		String dataname=args[0];
		datasetsDir=args[1];
		File repSave = new File(saveoutputDir);
//		File[] repSavelist;
//		if (!repSave.exists()) {
//			repSave.mkdirs();
//		} else {
//			repSavelist = repSave.listFiles();
//			for (int i = 0; i < repSavelist.length; i++) {
//				repSavelist[i].delete();
//			}
//		}

		// datasets folder
		File rep = new File(datasetsDir);
		File[] listData = rep.listFiles(new FileFilter() {
			public boolean accept(File pathname) {
				return pathname.isDirectory();
			}
		});
		Arrays.sort(listData);

		for (File dataRep : listData) {
			// only process GunPoint dataset to illustrates
//			if (dataRep.getName().equals("50words")||dataRep.getName().equals("Phoneme")||dataRep.getName().equals("DiatomSizeReduction"))
//				continue;
			if(!dataRep.getName().equals(dataname)||dataRep.getName().equals("ElectricDevices"))
				continue;
			System.out.println("processing: " + dataRep.getName());
			Instances[] data = readTrainAndTest(dataRep.getName());
//			new ExperimentsLauncher(repSave, data[0], data[1],dataRep.getName(), 10, data[0].numInstances()).launchKMedoids();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 1, data[0].numInstances()).launchKMeans();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchKMeansEUC();
//			new ExperimentsLauncher(repSave, data[0], data[1],dataRep.getName(), 100, data[0].numInstances()).launchRandom();
//			new ExperimentsLauncher(repSave, data[0], data[1],dataRep.getName(), 1, data[0].numInstances()).launchAHC();
//			new ExperimentsLauncher(repSave, data[0], data[1],dataRep.getName(), 1, data[0].numInstances()).launchDrops();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchGmm();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchGmmEUC();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 10, data[0].numInstances()).launchKMeansProbabilistic();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchKMeansProbabilisticEUC();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchNewKMeans();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchFCM();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchseq();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchFSKMeans();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchAllKMeans();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchDT();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchJ48();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchBigDT();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchStaticEnsemble();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchDTWKNNClassifier();
			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchDynamicEnsemble();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 10, data[0].numInstances()).launchPU();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchPOSC45();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchPUKMeans();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchPUDTEnsemble();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 10, data[0].numInstances()).launchPUGMM();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 10, data[0].numInstances()).launchTestThread();
		}
	}

	public static Instances[] readTrainAndTest(String name) {
		File trainFile = new File(datasetsDir + name + "/" + name + "_TRAIN");
		if (!new File(trainFile.getAbsolutePath() + ".csv").exists()) {
			UCR2CSV.run(trainFile, new File(trainFile.getAbsolutePath() + ".csv"));
		}
		trainFile = new File(trainFile.getAbsolutePath() + ".csv");
		File testFile = new File(datasetsDir + name + "/" + name + "_TEST");
		if (!new File(testFile.getAbsolutePath() + ".csv").exists()) {
			UCR2CSV.run(testFile, new File(testFile.getAbsolutePath() + ".csv"));
		}
		testFile = new File(testFile.getAbsolutePath() + ".csv");

		CSVLoader loader = new CSVLoader();
		Instances trainDataset = null;
		Instances testDataset = null;

		try {
			loader.setFile(trainFile);
			loader.setNominalAttributes("first");
			trainDataset = loader.getDataSet();
			trainDataset.setClassIndex(0);

			loader.setFile(testFile);
			loader.setNominalAttributes("first");
			testDataset = loader.getDataSet();
			testDataset.setClassIndex(0);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return new Instances[] { trainDataset, testDataset };
	}
}
