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

import items.ClassedSequence;
import items.IndexedSequence;

import java.util.ArrayList;

import weka.core.Instances;

public class DTWKNNClassifierSimpleRank extends PrototyperSorted {
	private static final long serialVersionUID = -8718040020631038656L;

	@Override
	protected void buildSortedSequences(Instances data) {
		ArrayList<ClassedSequence> sortedSequencesTmp = new ArrayList<ClassedSequence>();
		int nbObjToRemove = data.numInstances();
		int nbClasses = data.numClasses();
		
		double[][] distances = new double[data.numInstances()][data.numInstances()];
		for (int i = 0; i < distances.length; i++) {
			for (int j = i+1; j < distances[i].length; j++) {
				distances[i][j] = sequences[i].distance(sequences[j]);
				distances[j][i] = distances[i][j];
			}
		}
		
		// create temp structure to remove "bad" examples
		ArrayList<IndexedSequence> tmpSequences = new ArrayList<IndexedSequence>();
		ArrayList<String> tmpClassMap = new ArrayList<String>();
		
		// prune tmpSequences and tmpSclassMap
		
		// init temp structure
		for (int i = 0; i < sequences.length; i++) {
			tmpSequences.add(new IndexedSequence(sequences[i], i));
			tmpClassMap.add(classMap[i]);
		}
		
		for (int p = 0; p < nbObjToRemove-2; p++) {
			// score for each point
			int scores[] = new int[tmpSequences.size()];
			// distance to nearest of the point of the same class
			double[] distToNearestOfSameClass = new double[tmpSequences.size()];
			
			for (int k = 0; k < distToNearestOfSameClass.length; k++) {
				distToNearestOfSameClass[k] = Double.MAX_VALUE;
			}
			ArrayList<Integer>[] nearestNeighbor = new ArrayList[tmpSequences.size()];
			
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
						// if object is of same class
						if(tmpClassMap.get(i).equals(tmpClassMap.get(j))) {
							// if it is nearest
							if(minD < distToNearestOfSameClass[i]) {
								distToNearestOfSameClass[i] = minD;
							}
						}
					}
				}
				if(nearestNeighbor[nn] == null) {
					nearestNeighbor[nn] = new ArrayList<Integer>();
				}
				// we tell to the NN that he is the winner
				nearestNeighbor[nn].add(i);
			}
			
			for (int i = 0; i < nearestNeighbor.length; i++) {
				if(nearestNeighbor[i] == null) {
					scores[i] = 0;
				} else {
					ArrayList<Integer> nn = nearestNeighbor[i];
					int tmpScore = 0;
					for (Integer k : nn) {
						// if k is of class i + 1
						if(tmpClassMap.get(k).equals(tmpClassMap.get(i)))
							tmpScore+=1;
						else
							tmpScore-=(2/nbClasses);
						// else -2/(nbC-1)
					}
					scores[i] = tmpScore;
				}
			}
			// find toRemove
			int toRemove = 0;
			for (int i = 1; i < scores.length; i++) {
				if(scores[i] <= scores[toRemove]) {
					if(distToNearestOfSameClass[i] < distToNearestOfSameClass[toRemove])
						toRemove = i;
				}
			}
			sortedSequencesTmp.add(new ClassedSequence(tmpSequences.get(toRemove).sequence, tmpClassMap.get(toRemove)));
			tmpSequences.remove(toRemove);
			tmpClassMap.remove(toRemove);
		}
		
		for (int i = 0; i < tmpSequences.size(); i++) {
			sortedSequencesTmp.add(new ClassedSequence(tmpSequences.get(i).sequence, tmpClassMap.get(i)));
		}
		
		sortedSequences = new ArrayList<ClassedSequence>();
		// reorder
		for (int i = sortedSequencesTmp.size()-1; i >= 0; i--) {
			sortedSequences.add(sortedSequencesTmp.get(i));
		}
	}
}
