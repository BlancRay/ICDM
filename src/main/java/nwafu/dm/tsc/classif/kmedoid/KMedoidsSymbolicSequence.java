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
package nwafu.dm.tsc.classif.kmedoid;

import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.Sequences;

import java.util.ArrayList;

import org.apache.commons.math3.random.RandomDataGenerator;

public class KMedoidsSymbolicSequence {
	public int nbCluster;
	public ArrayList<Sequence> data;
	public int[] indexMedoids;
	public int[] clusterMap;
	public RandomDataGenerator randGen;

	public KMedoidsSymbolicSequence(int nbCluster,
			ArrayList<Sequence> data) {
		this.nbCluster = nbCluster;
		this.data = data;
		this.clusterMap = new int[data.size()];
		this.randGen = new RandomDataGenerator();
	}

	public void cluster() {

		// pickup centers
		int nbSelected = Math.min(data.size(), nbCluster);
		indexMedoids = randGen.nextPermutation(data.size(), nbSelected);
		nbCluster = nbSelected;

		double[][] distances = new double[data.size()][data.size()];
		for (int i = 0; i < distances.length; i++) {
			for (int j = i + 1; j < distances[i].length; j++) {
				distances[i][j] = data.get(i).distance(data.get(j));
				distances[j][i] = distances[i][j];
			}
		}

		ArrayList<Sequence>[] affectation = new ArrayList[nbCluster];
		// init
		for (int i = 0; i < affectation.length; i++) {
			affectation[i] = new ArrayList<Sequence>();
		}

		boolean changed = true;
		// for each iteration i
		for (int i = 0; i < 150 && changed; i++) {
			changed = false;
			// System.out.println(i);
			// for each data point j
			for (int j = 0; j < data.size(); j++) {

				double minDist = Double.MAX_VALUE;
				// for each cluster k
				for (int k = 0; k < indexMedoids.length; k++) {
					if (indexMedoids[k] == -1) {// nothing in cluster k
						continue;
					}
					// distance between cluster k and data point j
					double currentDist = distances[indexMedoids[k]][j];
					if (currentDist < minDist) {
						clusterMap[j] = k;
						minDist = currentDist;
					}
				}

				// affect data point j to cluster affected to j
				affectation[clusterMap[j]].add(data.get(j));
			}

			// redefine
			for (int j = 0; j < nbCluster; j++) {
				int tmpIndex = Sequences.medoidIndex(affectation[j]);
				if (tmpIndex != indexMedoids[j]) {
					indexMedoids[j] = tmpIndex;
					changed = true;
				}
			}

			// reset affect
			for (int k = 0; k < affectation.length; k++) {
				affectation[k] = new ArrayList<Sequence>();
			}
		}

	}

	public Sequence[] getCenters() {
		ArrayList<Sequence> centers = new ArrayList<Sequence>();

		for (int i = 0; i < indexMedoids.length; i++) {
			if(indexMedoids[i]!=-1){
				centers.add(data.get(indexMedoids[i]));
			}
		}

		return centers.toArray(new Sequence[]{});
	}
}
