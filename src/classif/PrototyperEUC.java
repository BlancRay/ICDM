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
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public abstract class PrototyperEUC extends Prototyper {
	private static final long serialVersionUID = 922540906465712982L;

	protected abstract void buildSpecificClassifier(Instances data);

	@Override
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
			double tmpD = seq.distanceEuc(s.sequence);
			if (tmpD < minD) {
				minD = tmpD;
				classValue = s.classValue;
			}
		}
		// System.out.println(prototypes.size());
//		System.out.println(classValue);
		return sample.classAttribute().indexOfValue(classValue);
	}
	
	@Override
	public double evalErrorRate(ClassedSequence[] testSequences) {
		
		int nbCorrectlyClassified = 0;
		for (int s = 0; s < testSequences.length; s++) {
			Sequence seq = testSequences[s].sequence;
			double minD = Double.MAX_VALUE;
			String classValue = null;
			
			for (ClassedSequence proto : prototypes) {
				double tmpD = seq.distanceEuc(proto.sequence);
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

}
