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
package nwafu.dm.tsc.classif.pukmeans;

import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.Sequences;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.math3.random.RandomDataGenerator;

public class KMeansCachedSymbolicSequence {
	public int nbClusters;
	public ArrayList<Sequence> data;
	public Sequence[]centers;
	public RandomDataGenerator randGen;
	public double wcss;
	public ArrayList<Integer>[] affectation;
	public double[][] distances;

	public KMeansCachedSymbolicSequence(int nbClusters, ArrayList<Sequence> data, double[][] distances) {
		if (data.size() < nbClusters) {
			this.nbClusters = data.size();
		} else {
			this.nbClusters = nbClusters;
		}
		this.data = data;
		this.randGen = new RandomDataGenerator();
		this.distances = distances;
	}

	public void cluster() {

		Sequence[] initialCenters = new Sequence[nbClusters];
		affectation = new ArrayList[nbClusters];

		// init
		for (int k = 0; k < affectation.length; k++) {
			affectation[k] = new ArrayList<Integer>();
		}

		// pickup centers
		int[] selected = randGen.nextPermutation(data.size(), nbClusters);
		for (int i = 0; i < selected.length; i++) {
			initialCenters[i] = data.get(selected[i]);
		}
		wcss = 0.0;
		// first affectation
		for (int j = 0; j < data.size(); j++) {

			double minDist = Double.MAX_VALUE;
			int bestK=-1;
			// for each cluster k
			for (int k = 0; k < initialCenters.length; k++) {
				// distance between cluster k and data point j
				double currentDist = initialCenters[k].distance(data.get(j));
				if (currentDist < minDist) {
					bestK = k;
					minDist = currentDist;
				}
			}
			wcss+=minDist*minDist;
			// affect data point j to cluster affected to j
			affectation[bestK].add(j);
		}

		// for each iteration i
		for (int i = 0; i < 15; i++) {

			ArrayList<Integer>[] newAffectation = new ArrayList[nbClusters];
			// init
			for (int k = 0; k < newAffectation.length; k++) {
				newAffectation[k] = new ArrayList<Integer>();
			}
			wcss = 0.0;
			// reassign element to cluster
			for (int j = 0; j < data.size(); j++) {
				int bestK = -1;
				double bestDist = Double.POSITIVE_INFINITY;
				// for each cluster k
				for (int k = 0; k < nbClusters; k++) {
					if (affectation[k].size() == 0) continue;
					double distToK = 0.0;
					for (Integer elIndex : affectation[k]) {
						double tmpDist = distances[j][elIndex];
						distToK += tmpDist ;//TODO squared??
					}
					distToK /= affectation[k].size();

					if (distToK < bestDist) {
						bestDist = distToK;
						bestK = k;
					}

				}
				wcss+=bestDist*bestDist;
				

				newAffectation[bestK].add(j);
			}

			affectation = newAffectation;

		}

		
		//find prototypes for classifier
		centers = new Sequence[nbClusters];
		for (int k = 0; k < nbClusters; k++) {
			if (affectation[k].size() == 0) {
				centers[k] = null;
			} else {
				int medoidIndex = Sequences.medoidIndex(affectation[k], distances);
				Sequence medoid = data.get(medoidIndex);
				
				Sequence []sequenceTab = new Sequence[affectation[k].size()];
				for(int i=0;i<sequenceTab.length;i++){
					sequenceTab[i]=data.get(affectation[k].get(i));
				}
				centers[k] = Sequences.meanWithMedoid(medoid, sequenceTab);
			}
		}

	}

	
}
