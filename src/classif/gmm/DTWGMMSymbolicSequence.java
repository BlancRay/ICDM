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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.random.RandomDataGenerator;

import classif.kmeans.KMeansSymbolicSequence;

public class DTWGMMSymbolicSequence {
	public int nbClusters;
	final ArrayList<Sequence> data;
	public RandomDataGenerator randGen;

	protected Sequence[] centroidsPerCluster = null;
	protected double[] sigmasPerCluster = null;
	protected int dataAttributes;

	private static final double minObj = 1;
	private static final double sqrt2Pi = Math.sqrt(2 * Math.PI);
	private double[] nck = null;
	private int sumnck;

	public DTWGMMSymbolicSequence(int nbClusters, ArrayList<Sequence> data, int dataAttributes) {
		if (data.size() < nbClusters) {
			this.nbClusters = data.size();
		} else {
			this.nbClusters = nbClusters;
		}
		this.data = data;
		this.dataAttributes = dataAttributes;
		this.randGen = new RandomDataGenerator();
	}

	public void cluster() {
		// init
		boolean isBig;
		ArrayList<Sequence>[] affectation=new ArrayList[nbClusters];
		int runtime=0;
		do {
			if(runtime>10)
				nbClusters-=1;
			isBig = true;
			KMeansSymbolicSequence kmeans = new KMeansSymbolicSequence(nbClusters, data);
			kmeans.cluster();
			centroidsPerCluster = kmeans.centers;
			affectation = kmeans.affectation;
			for (ArrayList<Sequence> Eachaffectation : affectation) {
				if (Eachaffectation.size() < minObj) {
					isBig = false;
					break;
				}
			}
			runtime++;
		} while (isBig == false);

		sigmasPerCluster = new double[nbClusters];
		nck = new double[nbClusters];
		sumnck = data.size();
		for (int k = 0; k < nbClusters; k++) {
			if (centroidsPerCluster[k] != null) { // ~ if empty cluster
				// find the center
				nck[k] = affectation[k].size();
				// compute sigma
				double sumOfSquares = centroidsPerCluster[k].sumOfSquares(affectation[k]);
				sigmasPerCluster[k] = Math.sqrt(sumOfSquares / nck[k]);
			} else
				System.err.println("ERROR");
		}

		double sumoflog = 0.0;
		double prevsumoflog = -(Math.exp(308));

		// computing initial likelihood
		sumoflog = loglikelihood(centroidsPerCluster, sigmasPerCluster, nck);

		/*while (Math.abs(sumoflog - prevsumoflog) > threshold) {
			prevsumoflog = sumoflog;
			sumoflog = gmmprocess();
		}*/
		for (int i = 0; i < 10; i++) {
			gmmprocess();
		}
	}


	
	private double gmmprocess() {
		double[][] gammak = new double[sumnck][nbClusters];
		double[] sumofgammak = new double[sumnck];
		double[][] gamma = new double[sumnck][nbClusters];
		// p(i,k)
		// get p(i,k)
		ArrayList<Sequence> sequencesForClass = data;
		// for each data point computer gamma
		for (int i = 0; i < sequencesForClass.size(); i++) {
			Sequence s = sequencesForClass.get(i);
			// sequence i for each cluster
			for (int k = 0; k < centroidsPerCluster.length; k++) {
				double dist = s.distance(centroidsPerCluster[k]);
				double p = computeProbaForQueryAndCluster(sigmasPerCluster[k], dist);
				gammak[i][k] = p * (nck[k] / sumnck);
			}

			// sum of gamma(k)
			for (int k = 0; k < gammak[i].length; k++) {
				sumofgammak[i] += gammak[i][k];
			}
			// gamma(i,k)
			for (int k = 0; k < centroidsPerCluster.length; k++) {
				gamma[i][k] = gammak[i][k] / sumofgammak[i];
			}
		}
		
		// Nk = sum of gamma
		for (int k = 0; k < centroidsPerCluster.length; k++) {
			// for each cluster
			double sumofgammai = 0;
			for (int i = 0; i < sumnck; i++) {
				sumofgammai += gamma[i][k];
			}
//			System.out.println(sumofgammai);
			nck[k] = sumofgammai;
			if (nck[k] <= minObj) {
				delcluster(k);
				double log=gmmprocess();
				return log;
			}
		}

		// centroidsPerClass
		for (int k = 0; k < centroidsPerCluster.length; k++) {
			centroidsPerCluster[k]= Sequences.weightMean(data.toArray(new Sequence[0]), gamma,k,nck[k]);
		}
		
		// sigma
		for (int k = 0; k < centroidsPerCluster.length; k++) {
			sigmasPerCluster[k] = 0;
			double sumOfSquares = 0.0;
			for (int i = 0; i < sumnck; i++) {
				double dist = data.get(i).distance(centroidsPerCluster[k]);
				sumOfSquares += (gamma[i][k] * dist * dist);
			}
//			sigmasPerCluster[k] = Math.sqrt(sumOfSquares / (nck[k] - 1));
			sigmasPerCluster[k] = Math.sqrt(sumOfSquares / nck[k]);
		}

		// computer log-likelihood
		double log = loglikelihood(centroidsPerCluster, sigmasPerCluster, nck);
		return log;
	}

	private double loglikelihood(Sequence[] mu, double[] sigma, double[] nbObjects) {
		double loglikelihood = 0.0;
		for (Sequence ss : data) {
			double prob = 0.0;
			for (int k = 0; k < mu.length; k++) {
				double dist = ss.distance(centroidsPerCluster[k]);
				double p = computeProbaForQueryAndCluster(sigma[k], dist);
				prob += p * (nbObjects[k] / sumnck);// probability of every point generated by each cluster
			}
			loglikelihood += Math.log(prob);
		}
		return loglikelihood;
	}

	private double computeProbaForQueryAndCluster(double sigma, double d) {
		double pqk = 0.0;
		if (sigma==0) {
			if (d == 0) {
				pqk = 1;
			} else
				pqk = 0;
		} else
			pqk = Math.exp(-(d * d) / (2 * sigma * sigma)) / (sigma * sqrt2Pi);
		return pqk;
	}

	private void delcluster(int k) {
		centroidsPerCluster[k] = null;
		sigmasPerCluster[k] = Double.NaN;
		Sequence[] newcenter = new Sequence[nbClusters - 1];
		double[] newsigma = new double[nbClusters - 1];
		nbClusters = nbClusters - 1;
		int flag =0;
		for (int i = 0; i < centroidsPerCluster.length; i++) {
			if (centroidsPerCluster[i] != null) {
				newcenter[i - flag] = centroidsPerCluster[i];
				newsigma[i - flag] = sigmasPerCluster[i];
			} else
				flag++;
		}
		centroidsPerCluster=newcenter;
		sigmasPerCluster=newsigma;
		ArrayList<Sequence>[] affectation = new ArrayList[nbClusters];
		int[] clusterMap=new int[data.size()];
		for (int i = 0; i < affectation.length; i++) {
			affectation[i] = new ArrayList<Sequence>();
		}
		
		for (int j = 0; j < data.size(); j++) {

			double minDist = Double.MAX_VALUE;
			// for each cluster k
			for (int i = 0; i < centroidsPerCluster.length; i++) {
				// distance between cluster k and data point j
				if (centroidsPerCluster[i] != null) {
					double currentDist = centroidsPerCluster[i].distance(data.get(j));
					if (currentDist < minDist) {
						clusterMap[j] = i;
						minDist = currentDist;
					}
				}
			}

			// affect data point j to cluster affected to j
			affectation[clusterMap[j]].add(data.get(j));
		}
		for (int i = 0; i < nbClusters; i++) {
			nck[i] = affectation[i].size();
		}
	}
	
	public Sequence[] getMus() {
		return centroidsPerCluster;
	}

	public double[] getSigmas() {
		return sigmasPerCluster;
	}
	
	public double[] getNck() {
		return nck;
	}
}
