package classif.kmeans;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.math3.random.RandomDataGenerator;

import items.Sequence;
import items.Sequences;

public class EUCKMeansSymbolicSequence {
	public int nbClusters;
	public ArrayList<Sequence> data;
	public Sequence[] centers;
	public int[] clusterMap;
	public RandomDataGenerator randGen;
	public ArrayList<Sequence>[] affectation;

	public EUCKMeansSymbolicSequence(int nbClusters,
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
					if (centers[k] != null) {
						double currentDist = centers[k].distanceEuc(data.get(j));
						if (currentDist < minDist) {
							clusterMap[j] = k;
							minDist = currentDist;
						}
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
					centers[j] = Sequences.meanEUC(affectation[j].toArray(new Sequence[0]));
				}
			}
		}
	}
}
