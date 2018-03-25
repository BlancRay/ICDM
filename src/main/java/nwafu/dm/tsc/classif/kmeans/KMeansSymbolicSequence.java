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
package nwafu.dm.tsc.classif.kmeans;

import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.Sequences;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.math3.random.RandomDataGenerator;

public class KMeansSymbolicSequence {
	public int nbClusters;
	public ArrayList<Sequence> data;
	public Sequence[] centers;
	public int[] clusterMap;
	public RandomDataGenerator randGen;
	public ArrayList<Sequence>[] affectation;

	public KMeansSymbolicSequence(int nbClusters,
			ArrayList<Sequence> data) {
		if(data.size()<nbClusters){
			this.nbClusters = data.size();
		}else{
			this.nbClusters = nbClusters;
		}
		this.data = data;
		this.clusterMap = new int[data.size()];
		this.randGen = new RandomDataGenerator();
	}

	public void cluster() {
		centers = new Sequence[nbClusters];
		affectation = new ArrayList[nbClusters];

		

		// pickup centers
		int[] selected = randGen.nextPermutation(data.size(), nbClusters);
		for (int i = 0; i < selected.length; i++) {
			centers[i] = data.get(selected[i]);
		}

		// for each iteration i
		for (int i = 0; i < 15; i++) {
			// init
			for (int k = 0; k < affectation.length; k++) {
				affectation[k] = new ArrayList<Sequence>();
			}
			// for each data point j
			for (int j = 0; j < data.size(); j++) {

				double minDist = Double.MAX_VALUE;
				// for each cluster k
				for (int k = 0; k < centers.length; k++) {
					// distance between cluster k and data point j
					double currentDist = centers[k].distance(data.get(j));
					if (currentDist < minDist) {
						clusterMap[j] = k;
						minDist = currentDist;
					}
				}

				// affect data point j to cluster affected to j
				affectation[clusterMap[j]].add(data.get(j));
			}

			// redefine
			for (int j = 0; j < nbClusters; j++) {
				if (affectation[j].size() == 0) {
					centers[j]=null;
				}else{
					centers[j] = Sequences.mean(affectation[j].toArray(new Sequence[0]));
				}
			}
		}
	}
}
