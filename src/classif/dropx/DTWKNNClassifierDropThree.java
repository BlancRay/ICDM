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
package classif.dropx;

import items.Sequence;
import items.ClassedSequence;
import items.SortedSequence;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import weka.core.Instances;

public class DTWKNNClassifierDropThree extends PrototyperSorted  {
	private static final long serialVersionUID = 4228874246068410151L;

	@Override
	protected void buildSortedSequences(Instances data) {		
		sortedSequences = new ArrayList<ClassedSequence>();
				
		// precompute distances
		double[][] distances = new double[data.numInstances()][data.numInstances()];
		for (int i = 0; i < distances.length; i++) {
			for (int j = i+1; j < distances[i].length; j++) {
				distances[i][j] = sequences[i].distance(sequences[j]);
				distances[j][i] = distances[i][j];
			}
		}
		
		ArrayList<SortedSequence> tmpSequences = new ArrayList<SortedSequence>();
		for (int i = 0; i < sequences.length; i++) {
			tmpSequences.add(new SortedSequence(new Sequence(sequences[i]), i, 0.0, classMap[i]));
		}
		
		// compute nearest neighbor and candidates
		ArrayList<Integer> toRemove = new ArrayList<Integer>();
		ArrayList<SortedSequence> tmpSequencesRest = new ArrayList<SortedSequence>();
		
		// remove instance misclassified by NN
		for (int i = 0; i < tmpSequences.size(); i++) {
			// find NN
			double minD = Double.MAX_VALUE;
			int index = 0;
			for (int j = 0; j < tmpSequences.size(); j++) {
				// avoid diagonal
				if(j != i) {
					// check distance
					double tmpD = distances[tmpSequences.get(i).index][tmpSequences.get(j).index];
					// if we found a new NN
					if(tmpD < minD) {
						minD = tmpD;
						index = j;
					}
				}
			}
			// if 'i' is misclassed => remove it !
			if(!tmpSequences.get(i).classValue.equals(tmpSequences.get(index).classValue)) {
				toRemove.add(i);
			} else {
				tmpSequencesRest.add(tmpSequences.get(i));
			}
		}
		
		// sort by nearest enemy (only the remaining instances)
		for (int i = 0; i < tmpSequencesRest.size(); i++) {
			// find nearest enemy
			double minD = Double.MAX_VALUE;
			for (int j = 0; j < tmpSequencesRest.size(); j++) {
				// avoid diagonal and of same class
				if(j != i && !tmpSequencesRest.get(i).classValue.equals(tmpSequencesRest.get(j).classValue)) {
					// check distance
					double tmpD = distances[tmpSequencesRest.get(i).index][tmpSequencesRest.get(j).index];
					// if we found a new NN (of != class)
					if(tmpD < minD) {
						minD = tmpD;
					}
				}
			}
			tmpSequencesRest.get(i).sortingValue = minD;
		}
		Collections.sort(tmpSequencesRest);
			
		for (int i = 0; i < tmpSequencesRest.size(); i++) {
			toRemove.add(tmpSequencesRest.get(i).index);
		}
		
		// create the list used to return the prototypes
		for (int j = toRemove.size() - 1; j >= 0; j--) {
			sortedSequences.add(new ClassedSequence(sequences[toRemove.get(j)],classMap[toRemove.get(j)]));
		}
	}
}
