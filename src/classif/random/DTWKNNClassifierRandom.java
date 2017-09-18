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
package classif.random;

import items.Sequence;
import items.ClassedSequence;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.math3.random.RandomDataGenerator;

import classif.Prototyper;
import weka.core.Instances;

public class DTWKNNClassifierRandom extends Prototyper {
	private static final long serialVersionUID = -7066310318246556613L;
	public ArrayList<Integer> indexPrototypeInTrainData = null;
	double[][]distances=null;
	
	public DTWKNNClassifierRandom() {
		super();
	}
	

	@Override
	protected void buildSpecificClassifier(Instances data) {
		indexPrototypeInTrainData = new ArrayList<Integer>();
		ArrayList<String> classes = new ArrayList<String>(classedData.keySet());
		
		RandomDataGenerator gen = new RandomDataGenerator();
		for (String clas : classes) {
			ArrayList<Sequence> cData = classedData.get(clas);
			// if the class is empty, skip it
			if(cData.isEmpty())
				continue;
			int maxElements = Math.min(nbPrototypesPerClass, cData.size());
			int [] selectedElements = gen.nextPermutation(cData.size(), maxElements);
			for (int i = 0; i < selectedElements.length; i++) {
				int indexInFullData = indexClassedDataInFullData.get(clas).get(selectedElements[i]);
				indexPrototypeInTrainData.add(indexInFullData);
//				System.out.println("prototype "+i+" of class "+clas+" is element "+selectedElements[i]+" index in data="+indexInFullData);
				ClassedSequence prot = new ClassedSequence(cData.get(selectedElements[i]), clas);
				prototypes.add(prot);
			}
		}
		
	}
	
	@Override
	public double evalErrorRate(ClassedSequence[] testSequences) {
		if(distances==null){
			initDistances(testSequences);
		}
		
		int nbCorrectlyClassified = 0;
		for (int s = 0; s < testSequences.length; s++) {
			double minD = Double.MAX_VALUE;
			String classValue = null;
			
			for (int p = 0;p < prototypes.size(); p++) {
				int indexProtoInTrainData = indexPrototypeInTrainData.get(p);
//				double tmpD = testSequences[s].sequence.distance(prototypes.get(p).sequence);
				double tmpD = distances[s][indexProtoInTrainData];
				if (tmpD < minD) {
					minD = tmpD;
					classValue = prototypes.get(p).classValue;
				}
			}
			
			if(classValue.equals(testSequences[s].classValue)){
				nbCorrectlyClassified++;
			}
			
		}
		
		return 1.0-1.0*nbCorrectlyClassified/(testSequences.length);
		
		
	}

	private void initDistances(ClassedSequence[] testSequences) {
		distances = new double[testSequences.length][sequences.length];
		
		for(int s=0;s< testSequences.length; s++) {
			for(int potentialProtoIndex=0;potentialProtoIndex<sequences.length;potentialProtoIndex++){
				
				distances[s][potentialProtoIndex] = testSequences[s].sequence.distance(sequences[potentialProtoIndex]);
			}
		}
		
	}
}
