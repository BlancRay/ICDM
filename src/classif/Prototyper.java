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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import org.apache.commons.math3.random.RandomDataGenerator;

import items.MonoDoubleItemSet;
import items.Sequence;
import items.ClassedSequence;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public abstract class Prototyper extends AbstractClassifier{
	private static final long serialVersionUID = 922540906465712982L;

	protected ArrayList<ClassedSequence> prototypes;
	protected HashMap<String, ArrayList<Sequence>> classedData;
	protected HashMap<String, ArrayList<Integer>> indexClassedDataInFullData;
	protected int nbPrototypesPerClass;
	protected Sequence[] sequences;
	protected String[] classMap;
	protected Instances trainingData = null;
	protected boolean fillPrototypes = true;

	public void buildClassifier(Instances data) throws Exception {
		trainingData = data;
		Attribute classAttribute = data.classAttribute();
		prototypes = new ArrayList<>();

		classedData = new HashMap<String, ArrayList<Sequence>>();
		indexClassedDataInFullData = new HashMap<String, ArrayList<Integer>>();
		for (int c = 0; c < data.numClasses(); c++) {
			classedData.put(data.classAttribute().value(c), new ArrayList<Sequence>());
			indexClassedDataInFullData.put(data.classAttribute().value(c), new ArrayList<Integer>());
		}

		sequences = new Sequence[data.numInstances()];
		classMap = new String[sequences.length];
		for (int i = 0; i < sequences.length; i++) {
			Instance sample = data.instance(i);
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
			int shift = (sample.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < sequence.length; t++) {
				sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
			}
			sequences[i] = new Sequence(sequence);
			String clas = sample.stringValue(classAttribute);
			classMap[i] = clas;
			classedData.get(clas).add(sequences[i]);
			indexClassedDataInFullData.get(clas).add(i);
//			System.out.println("Element "+i+" of train is classed "+clas+" and went to element "+(indexClassedDataInFullData.get(clas).size()-1));
		}

		buildSpecificClassifier(data);

		if (fillPrototypes) addMissingPrototypesRandom();
	}

	/**
	 * Set if you want to fill the prototypes
	 * 
	 * @param fillPrototypes
	 */
	public void setFillPrototypes(boolean fillPrototypes) {
		this.fillPrototypes = fillPrototypes;
	}

	/**
	 * Predict the accuracy of the prototypes based on the learning set. It uses
	 * cross validation to draw the prediction.
	 * 
	 * @param nbFolds
	 *            the number of folds for the x-validation
	 * @return the predicted accuracy
	 */
	public double predictAccuracyXVal(int nbFolds) throws Exception {
		Evaluation eval = new Evaluation(trainingData);
		eval.crossValidateModel(this, trainingData, nbFolds, new Random(), new Object[] {});
		return eval.errorRate();
	}

	/**
	 * If the Prototyper doesn't return as many prototypes as required, then add
	 * random data to fill the prototype list.
	 */
	private void addMissingPrototypesRandom() {
		int nbPrototypesNeeded = nbPrototypesPerClass * classedData.keySet().size();

		if (prototypes.size() < nbPrototypesNeeded) { // ~ needs to be filled
			RandomDataGenerator randGen = new RandomDataGenerator();
			int nbMissing = nbPrototypesNeeded - prototypes.size();
			int nbToAdd = Math.min(nbMissing, sequences.length);// if less data
																// than
																// prototypes
																// asked
			int[] indexFillins = randGen.nextPermutation(sequences.length, nbToAdd);
			for (int index : indexFillins) {
				ClassedSequence s = new ClassedSequence(sequences[index], classMap[index]);
				prototypes.add(s);
			}
		}
	}

	protected abstract void buildSpecificClassifier(Instances data);

	public double classifyInstance(Instance sample) throws Exception {
		// transform instance to sequence
		MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
		int shift = (sample.classIndex() == 0) ? 1 : 0;
		for (int t = 0; t < sequence.length; t++) {
			sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
		}
		Sequence seq = new Sequence(sequence);

		double minD = Double.MAX_VALUE;
		String classValue = null;
		for (ClassedSequence s : prototypes) {
			double tmpD = seq.distance(s.sequence);
			if (tmpD < minD) {
				minD = tmpD;
				classValue = s.classValue;
			}
		}
		// System.out.println(prototypes.size());
		return sample.classAttribute().indexOfValue(classValue);
	}
	
	public static ClassedSequence[] convertWekaSetToClassedSequence(Instances test){
		
		Attribute classAttribute = test.classAttribute();
		ClassedSequence[] testSequences = new ClassedSequence[test.numInstances()];
		for (int i = 0; i < testSequences.length; i++) {
			Instance sample = test.instance(i);
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
			int shift = (sample.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < sequence.length; t++) {
				sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
			}
			String clas = sample.stringValue(classAttribute);
			testSequences[i] = new ClassedSequence(new Sequence(sequence),clas);
		}
		
		return testSequences;
		
	}

	public double evalErrorRate(ClassedSequence[] testSequences) {
		
		int nbCorrectlyClassified = 0;
		for (int s = 0; s < testSequences.length; s++) {
			Sequence seq = testSequences[s].sequence;
			double minD = Double.MAX_VALUE;
			String classValue = null;
			
			for (ClassedSequence proto : prototypes) {
				double tmpD = seq.distance(proto.sequence);
				if (tmpD < minD) {
					minD = tmpD;
					classValue = proto.classValue;
				}
			}
			
			if(classValue.equals(testSequences[s].classValue)){
				nbCorrectlyClassified++;
			}
			
		}
		
		return 1.0-1.0*nbCorrectlyClassified/(testSequences.length);



	}

	public int getNbPrototypesPerClass() {
		return nbPrototypesPerClass;
	}

	public int getActualNumberOfPrototypesSelected() {
		return prototypes.size();
	}

	public void setNbPrototypesPerClass(int nbPrototypes) {
		this.nbPrototypesPerClass = nbPrototypes;
	}

	public ArrayList<ClassedSequence> getPrototypes() {
		return prototypes;
	}

}
