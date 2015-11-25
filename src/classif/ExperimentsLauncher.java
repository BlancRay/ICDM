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

import items.ClassedSequence;
import items.Sequence;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;

import tools.UCR2CSV;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import classif.ahc.DTWKNNClassifierAHC;
import classif.dropx.DTWKNNClassifierDropOne;
import classif.dropx.DTWKNNClassifierDropThree;
import classif.dropx.DTWKNNClassifierDropTwo;
import classif.dropx.DTWKNNClassifierSimpleRank;
import classif.dropx.PrototyperSorted;
import classif.kmeans.DTWKNNClassifierKMeans;
import classif.kmedoid.DTWKNNClassifierKMedoids;
import classif.random.DTWKNNClassifierRandom;

public class ExperimentsLauncher {
	public static String username = "forestier";
	private File rep;
	private int nbExp;
	private int nbPrototypesMax;
	private Instances train, test;
	private PrintStream out;
	private String dataName;
	long startTime;
	long endTime;
	long duration;
	private static boolean append = true;

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
			File f = new File(rep + "/" + dataName + "_results.csv");
			// if somebody is processing it
			if (f.exists()) {
				return;
			}

//			out = new PrintStream(new FileOutputStream(rep + "/" + dataName + "_results.csv", append));
//			out.println("dataset;algorithm;nbPrototypes;execTime;trainErrorRate;testErrorRate;prototypesPerClassDistribution");
			out = new PrintStream(new FileOutputStream(rep + "/All_results.csv", append));
//			out.println("dataset;algorithm;nbPrototypes;trainErrorRate_Now;testErrorRate_Now;trainErrorRate_Before;testErrorRate_Before");
			String algo = "KMEANS";


			nbPrototypesMax = this.train.numInstances() / this.train.numClasses();
			if(nbPrototypesMax>2)
			nbPrototypesMax=2;

			for (int j = 2; j <= nbPrototypesMax; j++) {
				System.out.println("nbPrototypes=" + j);
				for (int n = 0; n < nbExp; n++) {
					System.out.println("This is the "+n+" time.");
					DTWKNNClassifierKMeans classifier = new DTWKNNClassifierKMeans();
					DTWProbabilisticClassifierKMeans classifierKMeans = new DTWProbabilisticClassifierKMeans();
					classifierKMeans.setNClustersPerClass(j);
					classifier.setNbPrototypesPerClass(j);
					classifier.setFillPrototypes(true);

					startTime = System.currentTimeMillis();
					classifierKMeans.buildClassifier(train);
					endTime = System.currentTimeMillis();
					duration = endTime - startTime;

					classifier.buildClassifier(train);
					
					int[] classDistrib = PrototyperUtil.getPrototypesPerClassDistribution(classifierKMeans.prototypes, train);
					
					Evaluation eval_a = new Evaluation(train);
					eval_a.evaluateModel(classifierKMeans, test);
					double testError = eval_a.errorRate();
					double trainError = Double.NaN;
					Evaluation eval_b = new Evaluation(train);
					eval_b.evaluateModel(classifier, test);
					double testError_b = eval_b.errorRate();
					double trainError_b = Double.NaN;
					
					System.out.println(testError+"\n");

					PrototyperUtil.savePrototypes(classifierKMeans.prototypes, rep + "/" + dataName + "_KMEANS[" + j + "]_XP" + n + ".proto");

//					out.format("%s;%s;%d;%d;%.4f;%.4f;%s\n", dataName, algo, (j * train.numClasses()), duration, trainError, testError, Arrays.toString(classDistrib));
					out.format("%s;%s;%d;%.4f;%.4f;%.4f;%.4f\n", dataName, algo, (j * train.numClasses()),  trainError, testError,trainError_b,testError_b);
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

	public static void main(String[] args) {
		File repSave = new File("C:\\Users\\leix\\workspace\\ICDM2014\\save\\");
		File[] repSavelist;
		if (!repSave.exists()) {
			repSave.mkdirs();
		} else {
			repSavelist = repSave.listFiles();
			for (int i = 0; i < repSavelist.length; i++) {
				repSavelist[i].delete();
			}
		}

		// datasets folder
		File rep = new File("C:\\Users\\leix\\Downloads\\UCR_TS_Archive_2015\\");
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
			if(!dataRep.getName().equals("Trace"))
				continue;
			System.out.println("processing: " + dataRep.getName());
			Instances[] data = readTrainAndTest(dataRep.getName());
			// new ExperimentsLauncher(repSave, data[0], data[1],dataRep.getName(), 10, data[0].numInstances()).launchKMedoids();
			new ExperimentsLauncher(repSave, data[0], data[1], dataRep.getName(), 1, data[0].numInstances()).launchKMeans();
			// new ExperimentsLauncher(repSave, data[0], data[1],dataRep.getName(), 100, data[0].numInstances()).launchRandom();
			// new ExperimentsLauncher(repSave, data[0], data[1],dataRep.getName(), 1, data[0].numInstances()).launchAHC();
			// new ExperimentsLauncher(repSave, data[0], data[1],dataRep.getName(), 1, data[0].numInstances()).launchDrops();
		}
	}

	public static Instances[] readTrainAndTest(String name) {
		File trainFile = new File("C:\\Users\\leix\\Downloads\\UCR_TS_Archive_2015\\" + name + "/" + name + "_TRAIN");
		if (!new File(trainFile.getAbsolutePath() + ".csv").exists()) {
			UCR2CSV.run(trainFile, new File(trainFile.getAbsolutePath() + ".csv"));
		}
		trainFile = new File(trainFile.getAbsolutePath() + ".csv");
		File testFile = new File("C:\\Users\\leix\\Downloads\\UCR_TS_Archive_2015\\" + name + "/" + name + "_TEST");
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
