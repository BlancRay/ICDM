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
package classif.gmm;

import items.MonoDoubleItemSet;
import items.Sequence;
import items.Sequences;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.math3.random.RandomDataGenerator;

public class EUCGMMSymbolicSequence {
	public int nbClusters;
	public ArrayList<Sequence> data;
	public Sequence[] centers;
	public int[] clusterMap;
	public RandomDataGenerator randGen;
	public ArrayList<Sequence>[] affectation;
	protected Sequence[] centroidsPerCluster = null;
	protected double[] sigmasPerCluster = null;
	protected int dataAttributes;
	
	private double[] dist;
	private Sequence[] tmpcenters;
	private static final double threshold = Math.pow(10, -11);
	private static final double sqrt2Pi = Math.sqrt(2 * Math.PI);
	double[] prior = null;
	double[] nck = null;

	public EUCGMMSymbolicSequence(int nbClusters, ArrayList<Sequence> data, int dataAttributes) {
		if (data.size() < nbClusters) {
			this.nbClusters = data.size();
		} else {
			this.nbClusters = nbClusters;
		}
		this.data = data;
		this.dataAttributes = dataAttributes;
		this.clusterMap = new int[data.size()];
		this.randGen = new RandomDataGenerator();
	}

	
	public void cluster() {
		centers = new Sequence[nbClusters];
		affectation = new ArrayList[nbClusters];
		dist= new double[nbClusters];

		// pickup centers
		int[] selected = randGen.nextPermutation(data.size(), nbClusters);
		for (int i = 0; i < selected.length; i++) {
			centers[i] = data.get(selected[i]);
		}

		double mindist=Double.MAX_VALUE;
		// for each iteration i
		do{
			tmpcenters = centers.clone();
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
					double currentDist = centers[k].distanceEuc(data.get(j));
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
					centers[j] = null;
				} else {
					centers[j] = Sequences.meanEUC(affectation[j].toArray(new Sequence[0]));
				}
			}
			for (int j = 0; j < nbClusters; j++) {
				if (centers[j] != null)
					dist[j] = centers[j].distanceEuc(tmpcenters[j]);
				if(dist[j]<mindist)
					mindist = dist[j];
			}
		}while(mindist>threshold);
		
		GMMprocess();
		
	}
	
	
	public void GMMprocess() {
		centroidsPerCluster = new Sequence[nbClusters];
		sigmasPerCluster = new double[nbClusters];
		prior = new double[nbClusters];
		nck = new double[nbClusters];
		double sumnck = 0.0;
		double sumoflog = 0.0;
		double prevsumoflog = -(Math.exp(308));
		for (int k = 0; k < centers.length; k++) {
			if (centers[k] != null) { // ~ if empty cluster
				// find the center
				centroidsPerCluster[k] = centers[k];
				int nObjectsInCluster = affectation[k].size();

				// compute sigma
				double sumOfSquares = centers[k].EUCsumOfSquares(affectation[k]);
				sigmasPerCluster[k] = Math.sqrt(sumOfSquares / (nObjectsInCluster - 1));
				// System.out.println(sigmasPerClass[c][k]);
				// compute p(k)
				// the P(K) of k
				nck[k] = nObjectsInCluster;
				sumnck += nObjectsInCluster;
				prior[k] = 1.0 * nObjectsInCluster / data.size();
				System.out.println(centroidsPerCluster[k] + "\t" + sigmasPerCluster[k]);
			}
		}

		// computing initial likelihood
		for (Sequence ss : data) {
			double prob = 0.0;
			for (int k = 0; k < centroidsPerCluster.length; k++) {
				double dist = ss.distanceEuc(centroidsPerCluster[k]);
				double p = computeProbaForQueryAndCluster(sigmasPerCluster[k], dist);
				prob += p * prior[k];// probability of every point
				// generated by each cluster
			}
			sumoflog += Math.log(prob);
		}

		clusterMap = new int[data.size()];
		while (Math.abs(sumoflog - prevsumoflog) > threshold) {
			double[][] gammak = new double[data.size()][nbClusters];
			double[] sumofgammak = new double[data.size()];
			double[][] gamma = new double[data.size()][nbClusters];
			// System.out.println("sumoflog="+sumoflog);
			// p(i,k)
			prevsumoflog = sumoflog;
			// get p(i,k)
			ArrayList<Sequence> sequencesForClass = data;
			// for each data point computer gamma
			for (int i = 0; i < sequencesForClass.size(); i++) {
				Sequence s = sequencesForClass.get(i);
				// for each p(k)
				for (int k = 0; k < centroidsPerCluster.length; k++) {
					// nck[k] = affect[k].size();
					double dist = s.distanceEuc(centroidsPerCluster[k]);
					double p = computeProbaForQueryAndCluster(sigmasPerCluster[k], dist);
					gammak[i][k] = p * nck[k] / sumnck;
				}

				// sum of p(k)
				for (int k = 0; k < gammak[i].length; k++) {
					sumofgammak[i] += gammak[i][k];
				}
				// p(i,k)
				for (int k = 0; k < centroidsPerCluster.length; k++) {
					gamma[i][k] = gammak[i][k] / sumofgammak[i];
				}
			}

			// Nk = sum of gamma
			for (int k = 0; k < centroidsPerCluster.length; k++) {
				// for each cluster
				double sumofgammai = 0;
				for (int i = 0; i < data.size(); i++) {
					sumofgammai += gamma[i][k];
				}
				nck[k] = sumofgammai;
			}

			// centroidsPerClass
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[dataAttributes];

			for (int k = 0; k < centroidsPerCluster.length; k++) {
				MonoDoubleItemSet[] sequencetmp = new MonoDoubleItemSet[dataAttributes];
				MonoDoubleItemSet[] sumofSTmp = new MonoDoubleItemSet[dataAttributes];
				// new MonoDoubleItemSet
				for (int t = 0; t < sequence.length; t++) {
					sequencetmp[t] = new MonoDoubleItemSet(0.0);
					sumofSTmp[t] = new MonoDoubleItemSet(0.0);
				}

				for (int i = 0; i < data.size(); i++) {
					sequence = (MonoDoubleItemSet[]) data.get(i).getSequence();
					// gamma(i)*x(i)[t]
					for (int t = 0; t < sequence.length; t++) {
						sequencetmp[t] = new MonoDoubleItemSet(gamma[i][k] * sequence[t].getValue());
					}
					// sum of gamma(i)*x(i)
					for (int t = 0; t < sequence.length; t++) {
						sumofSTmp[t] = new MonoDoubleItemSet(sequencetmp[t].getValue() + sumofSTmp[t].getValue());
					}
				}
				// (sum of gamma*x) / (sum of gamma)
				for (int t = 0; t < sequence.length; t++) {
					sumofSTmp[t] = new MonoDoubleItemSet(sumofSTmp[t].getValue() / nck[k]);
				}
				centroidsPerCluster[k] = new Sequence(sumofSTmp);
			}

			// sigma
			for (int k = 0; k < centroidsPerCluster.length; k++) {
				sigmasPerCluster[k] = 0;
				double inertia = 0.0;
				for (int i = 0; i < data.size(); i++) {
					double dist = data.get(i).distanceEuc(centroidsPerCluster[k]);
					// double dist =
					// affect[k].get(i).distanceEuc(centroidsPerClass[c][k]);
					inertia += (gamma[i][k] * dist * dist);
				}
				sigmasPerCluster[k] = Math.sqrt(inertia / (nck[k] - 1));
			}

			// computer log-likelihood
			sumoflog = 0;
			for (Sequence ss : data) {
				double prob = 0.0;
				for (int k = 0; k < centroidsPerCluster.length; k++) {
					double dist = ss.distanceEuc(centroidsPerCluster[k]);
					double p = computeProbaForQueryAndCluster(sigmasPerCluster[k], dist);
					prob += p * (nck[k] / data.size());
				}
				sumoflog += Math.log(prob);
			}
		}

	}

	private double computeProbaForQueryAndCluster(double sigma, double d) {
		double pqk;
		if (Double.isNaN(sigma)) {
			// System.err.println("alert");
			pqk = 0.0;
		} else
			pqk = Math.exp(-(d * d) / (2 * sigma * sigma)) / (sigma * sqrt2Pi);

		return pqk;
	}
	public Sequence[] getMus() {
	    return centroidsPerCluster;
	}

	public double[] getSigmas() {
	    return sigmasPerCluster;
	}
	
	/*public void cluster() {
		centers = new Sequence[nbClusters];
		affectation = new ArrayList[nbClusters];

		// pickup centers
		int[] selected = randGen.nextPermutation(data.size(), nbClusters);
		for (int i = 0; i < selected.length; i++) {
			centers[i] = data.get(selected[i]);
		}

		// for each iteration i
		for (int i = 0; i < 5; i++) {
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
					double currentDist = centers[k].distanceEuc(data.get(j));
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
					centers[j] = Sequences.meanEUC(affectation[j].toArray(new Sequence[0]));
				}
			}
		}
	}*/
}
