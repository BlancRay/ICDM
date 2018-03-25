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
package nwafu.dm.tsc.classif.dropx;

import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.ClassedSequence;
import nwafu.dm.tsc.items.SortedSequence;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import weka.core.Instances;

public class DTWKNNClassifierDropTwo extends PrototyperSorted  {
	private static final long serialVersionUID = -634976415435902514L;

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
	
		// init temp structure
		ArrayList<SortedSequence> tmpSequences = new ArrayList<SortedSequence>();
		ArrayList<SortedSequence> tmpSequencesFull = new ArrayList<SortedSequence>();
		for (int i = 0; i < sequences.length; i++) {
			tmpSequences.add(new SortedSequence(new Sequence(sequences[i]), i, 0.0, classMap[i]));
			tmpSequencesFull.add(tmpSequences.get(i));
		}
		
		// sort by nearest enemy
		for (int i = 0; i < tmpSequences.size(); i++) {
			// find nearest enemy
			double minD = Double.MAX_VALUE;
			for (int j = 0; j < tmpSequences.size(); j++) {
				// avoid diagonal
				if(j != i && !tmpSequences.get(i).classValue.equals(tmpSequences.get(j).classValue)) {
					// check distance
					double tmpD = distances[tmpSequences.get(i).index][tmpSequences.get(j).index];
					// if we found a new NN
					if(tmpD < minD) {
						minD = tmpD;
					}
				}
			}
			tmpSequences.get(i).sortingValue = minD;
		}
		Collections.sort(tmpSequences);
				
		// compute nearest neigbor and candidates
		ArrayList<Integer> toRemove = new ArrayList<Integer>();
		
		while(tmpSequences.size() > 2) {
			
			/** compute the NN and candidates **/
			int[] nearestNeighbor = new int[data.numInstances()];
			ArrayList<Integer>[] candidates = new ArrayList[data.numInstances()];
			
			// for each object
			for (int i = 0; i < tmpSequences.size(); i++) {
				// looking for the nearest
				double minD = Double.MAX_VALUE;
				int nn = -1;
				// we look for the NN
				for (int j = 0; j < tmpSequences.size(); j++) {
					// avoid diagonal
					if(j != i) {
						// check distance
						double tmpD = distances[tmpSequences.get(i).index][tmpSequences.get(j).index];
						// if we found a new NN
						if(tmpD < minD) {
							nn = j;
							minD = tmpD;
						}
					}
				}
//				if(candidates[nn] == null) {
//					candidates[nn] = new ArrayList<Integer>();
//				}
//				// we tell to the NN that he is the winner
//				candidates[nn].add(i);
//				// we store the NN
//				nearestNeighbor[i] = nn;
			}
			
			// for each object
			for (int i = 0; i < tmpSequencesFull.size(); i++) {
				// looking for the nearest
				double minD = Double.MAX_VALUE;
				int nn = -1;
				// we look for the NN
				for (int j = 0; j < tmpSequences.size(); j++) {
					// avoid diagonal
					if(j != i) {
						// check distance
						double tmpD = distances[tmpSequencesFull.get(i).index][tmpSequences.get(j).index];
						// if we found a new NN
						if(tmpD < minD) {
							nn = j;
							minD = tmpD;
						}
					}
				}
				if(candidates[nn] == null) {
					candidates[nn] = new ArrayList<Integer>();
				}
				// we tell to the NN that he is the winner
				candidates[nn].add(i);
//				// we store the NN
//				nearestNeighbor[i] = nn;
			}
			

			// remove object one by one according to with/without rule
			int toRem = -1;
			for (int i = 0; i < tmpSequences.size(); i++) {
				int with = 0;
				int without = 0;

				if(candidates[i] == null) {
					toRem = i;
//					System.out.println("candidate null break");
					break;
				} else {
					ArrayList<Integer> candidatesOf = candidates[i];
					
					// compute WITH
					for (int j = 0; j < candidatesOf.size(); j++) {
						if(tmpSequences.get(i).classValue.equals(tmpSequencesFull.get(candidatesOf.get(j)).classValue)) {
							with++;
						}
					}
				
					// compute WITHOUT
					int[] newNearestNeighbor = new int[candidatesOf.size()];
					double[] minForNewNearestNeighbor = new double[candidatesOf.size()];
					for (int k = 0; k < minForNewNearestNeighbor.length; k++) {
						minForNewNearestNeighbor[k] = Double.MAX_VALUE;
					}
					
					// for each object
					for (int k = 0; k < tmpSequences.size(); k++) {
						// if different from current
						if(k != i) {
							// get the object
							SortedSequence tmpSeq = tmpSequences.get(k);
							for (int l = 0; l < newNearestNeighbor.length; l++) {
								if(tmpSequences.get(k).index != tmpSequencesFull.get(candidatesOf.get(l)).index) {
									double tmpD = distances[tmpSeq.index][tmpSequencesFull.get(candidatesOf.get(l)).index];
									if(tmpD < minForNewNearestNeighbor[l]) {
										minForNewNearestNeighbor[l] = tmpD;
										newNearestNeighbor[l] = k;
									}
								}
							}
						}
					}
//					System.out.println("was "+candidatesOf);
//					System.out.println("is "+Arrays.toString(newNearestNeighbor));
					
					for (int j = 0; j < newNearestNeighbor.length; j++) {
						if(tmpSequences.get(newNearestNeighbor[j]).classValue.equals(tmpSequencesFull.get(candidatesOf.get(j)).classValue)) {
							without++;
						}
					}
					
					if(without >= with) {
						toRem = i;
						break;
					}
				}
			}
			// no match found
			if(toRem == -1) { 
				toRem = 0;
			}
			toRemove.add(tmpSequences.get(toRem).index);
			tmpSequences.remove(toRem);
		}
		
		// create the list used to return the prototypes
		for (int j = toRemove.size() - 1; j >= 0; j--) {
			sortedSequences.add(new ClassedSequence(sequences[toRemove.get(j)],classMap[toRemove.get(j)]));
		}
	}
}
