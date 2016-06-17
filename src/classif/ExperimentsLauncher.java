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
package classif;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;

import classif.DT.ClassifyDT;
import classif.Majority.KMeans;
import classif.ahc.DTWKNNClassifierAHC;
import classif.dropx.DTWKNNClassifierDropOne;
import classif.dropx.DTWKNNClassifierDropThree;
import classif.dropx.DTWKNNClassifierDropTwo;
import classif.dropx.DTWKNNClassifierSimpleRank;
import classif.dropx.PrototyperSorted;
import classif.fastkmeans.DTWKNNClassifierKMeansCached;
import classif.fuzzycmeans.DTWKNNClassifierFCM;
import classif.gmm.DTWKNNClassifierGmm;
import classif.gmm.EUCKNNClassifierGmm;
import classif.kmeans.DTWKNNClassifierKMeans;
import classif.kmeans.DTWProbabilisticClassifierKMeans;
import classif.kmeans.EUCKNNClassifierKMeans;
import classif.kmeans.EUCProbabilisticClassifierKMeans;
import classif.kmedoid.DTWKNNClassifierKMedoids;
import classif.newkmeans.DTWKNNClassifierNK;
import classif.random.DTWKNNClassifierRandom;
import classif.sep.Unbalancecluster;
import items.ClassedSequence;
import tools.UCR2CSV;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
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

	/**
	 * 
	 */
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

					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifierKMeans.prototypes, train);

//					Evaluation eval = new Evaluation(train);
//					eval.evaluateModel(classifierKMeans, test);
					Evaluation evaltrain = new Evaluation(train);
					evaltrain.evaluateModel(classifierKMeans, train);
					
//					double testError = eval.errorRate();
					double trainError = evaltrain.errorRate();
//					System.out.println("TestError:"+testError+"\n");
					System.out.println("trainError:"+trainError+"\n");
					
					
/*					DTWKNNClassifierKMeans KMeans = new DTWKNNClassifierKMeans();
					KMeans.setNbPrototypesPerClass(j);
					KMeans.setFillPrototypes(true);
					Evaluation evalcv = new Evaluation(train);
					Random rand = new Random(1);
					evalcv.crossValidateModel(KMeans, train, 10, rand);
					double testError = evalcv.errorRate();
					System.out.println("CVError:"+testError+"\n");*/

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
	
	/**
	 * 
	 */
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

	/**
	 * 
	 */
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

	public void launchseq() {
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
			out = new PrintStream(new FileOutputStream(rep + "/FastKMeansDTW_" + dataName + "_results.csv", true));
//			out.println("dataset,algorithm,nbPrototypes,testErrorRate,trainErrorRate");
			String algo = "FastKMEANS";
			System.out.println(algo);
//			PrintStream outProto = new PrintStream(new FileOutputStream(rep + "/" + dataName + "_KMEANS.proto", append));

//			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();
//			if (nbPrototypesMax>10)
			nbPrototypesMax = 2000;
			int tmp;
			tmp = nbExp;
			double[] trainrctmp = new double[5];
			double[] testrctmp = new double[5];
			double[] cvrctmp = new double[5];
			boolean stopflag=false;
			for (int j = 1; j <= nbPrototypesMax; j++) {
				double[] trainrc = new double[5];
				double[] testrc = new double[5];
				double[] cvrc = new double[5];
				if (j == 1)
					nbExp = 1;
				else
					nbExp = tmp;
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
					System.out.println("This is the "+n+" time.");
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
					Evaluation evaltrain = new Evaluation(train);
					evaltrain.evaluateModel(classifierKMeans, train);
					
					double testError = evaltest.errorRate();
					double trainError = evaltrain.errorRate();
					System.out.println("TestError:"+testError+"\n");
					System.out.println("trainError:"+trainError+"\n");
					
					
					DTWKNNClassifierKMeansCached KMeans = new DTWKNNClassifierKMeansCached();
					KMeans.setNbPrototypesPerClass(j);
					KMeans.setFillPrototypes(true);
					Evaluation evalcv = new Evaluation(train);
					Random rand = new Random(1);
					evalcv.crossValidateModel(KMeans, train, 10, rand);
					double CVError = evalcv.errorRate();
					System.out.println("CVError:"+CVError+"\n");

//					PrototyperUtil.savePrototypes(classifierKMeans.prototypes, rep + "/" + dataName + "_KMEANS[" + j + "]_XP" + n + ".proto");

					out.format("%s,%s,%d,%.4f,%.4f,%.4f\n", dataName, algo, (j * train.numClasses()), testError,CVError,trainError);
					out.flush();
					trainrc[n]=trainError;
					testrc[n]=testError;
					cvrc[n]=CVError;
					if (n == 4) {
						if (j == 1) {
							trainrctmp = trainrc;
							testrctmp = testrc;
							cvrctmp = cvrc;
						} else {
							if (Arrays.equals(trainrc, trainrctmp) && Arrays.equals(testrc, testrctmp)
									&& Arrays.equals(cvrc, cvrctmp)) {
								System.out.println("Stable at " + j);
								stopflag=true;
							} else {
								trainrctmp = trainrc;
								testrctmp = testrc;
								cvrctmp = cvrc;
							}
						}
					}
				}
				if(stopflag==true)
					break;
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
	
	public void launchDT() {
		if(train.numClasses()==2){
			
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
	}}
	
	public void launchJ48() {
		try {
			String algo = "J48";
			System.out.println(algo);

			double testError = 0.0;
			J48 dt = new J48();
			dt.buildClassifier(train);
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(dt, test);
			testError = eval.errorRate();
			System.out.println("TestError:" + testError + "\n");
			System.out.println(dt.toSummaryString());
			System.out.println(dt.graph());
			System.out.println(eval.toSummaryString());

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	public static void main(String[] args) {
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
			@Override
			public boolean accept(File pathname) {
				return pathname.isDirectory();
			}
		});
		Arrays.sort(listData);

		for (File dataRep : listData) {
			// only process GunPoint dataset to illustrates
			if (dataRep.getName().equals("50words")||dataRep.getName().equals("Phoneme")||dataRep.getName().equals("DiatomSizeReduction"))
				continue;
			if(!dataRep.getName().equals(args[0]))
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
			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchDT();
//			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 5, data[0].numInstances()).launchJ48();
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
