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

import java.util.ArrayList;

import org.apache.commons.math3.random.RandomDataGenerator;

import classif.kmeans.EUCKMeansSymbolicSequence;

public class EUCGMMSymbolicSequence {
	public int nbClusters;
	public ArrayList<Sequence> data;
	public RandomDataGenerator randGen;

	protected Sequence[] centroidsPerCluster = null;
	protected double[] sigmasPerCluster = null;
	protected int dataAttributes;

	private static final double threshold = Math.pow(10, -6);
	private static final double sqrt2Pi = Math.sqrt(2 * Math.PI);
	private double[] nck = null;
	private int sumnck;

	public EUCGMMSymbolicSequence(int nbClusters, ArrayList<Sequence> data, int dataAttributes) {
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
		EUCKMeansSymbolicSequence kmeans = new EUCKMeansSymbolicSequence(nbClusters, data);
		kmeans.cluster();
		centroidsPerCluster = kmeans.centers;
		ArrayList<Sequence>[] affectation = kmeans.affectation;

		sigmasPerCluster = new double[nbClusters];
		nck = new double[nbClusters];
		sumnck = data.size();
		for (int k = 0; k < nbClusters; k++) {
			if (centroidsPerCluster[k] != null && affectation[k].size()>1) { // ~ if empty cluster
			// find the center
				nck[k] = affectation[k].size();

				// compute sigma
				double sumOfSquares = centroidsPerCluster[k].EUCsumOfSquares(affectation[k]);
				sigmasPerCluster[k] = Math.sqrt(sumOfSquares / (nck[k] - 1));
				// System.out.println(sigmasPerClass[c][k]);
				// compute p(k)
				// the P(K) of k
//				System.out.println(centroidsPerCluster[k] + "\t" + sigmasPerCluster[k]);
			} else {// if empty cluster
				sigmasPerCluster[k] = Double.NaN;
				nck[k] = 1.0;
			}
		}

		double sumoflog = 0.0;
		double prevsumoflog = -(Math.exp(308));

		// computing initial likelihood
		sumoflog = loglikelihood(centroidsPerCluster, sigmasPerCluster, nck);

		while (Math.abs(sumoflog - prevsumoflog) > threshold) {
			prevsumoflog = sumoflog;
			sumoflog = gmmprocess();
		}
	}


	
	private double gmmprocess() {
		double[][] gammak = new double[sumnck][nbClusters];
		double[] sumofgammak = new double[sumnck];
		double[][] gamma = new double[sumnck][nbClusters];
		// System.out.println("sumoflog="+sumoflog);
		// p(i,k)
		// get p(i,k)
		ArrayList<Sequence> sequencesForClass = data;
		// for each data point computer gamma
		for (int i = 0; i < sequencesForClass.size(); i++) {
			Sequence s = sequencesForClass.get(i);
			// for each cluster gammak = N(xi|mu,sigma)*p(k)
			for (int k = 0; k < centroidsPerCluster.length; k++) {
				double dist = s.distanceEuc(centroidsPerCluster[k]);
				double p = computeProbaForQueryAndCluster(sigmasPerCluster[k], dist);
				gammak[i][k] = p * nck[k] / sumnck;
				System.out.println(gammak[i][k]);
			}

			// sum of gammak
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
			for (int i = 0; i < sumnck; i++) {
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

			for (int i = 0; i < sumnck; i++) {
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
			double sumOfSquares = 0.0;
			for (int i = 0; i < sumnck; i++) {
				double dist = data.get(i).distanceEuc(centroidsPerCluster[k]);
				// double dist =
				// affect[k].get(i).distanceEuc(centroidsPerClass[c][k]);
				sumOfSquares += (gamma[i][k] * dist * dist);
			}
			sigmasPerCluster[k] = Math.sqrt(sumOfSquares / (nck[k] - 1));
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
				double dist = ss.distanceEuc(centroidsPerCluster[k]);
				double p = computeProbaForQueryAndCluster(sigma[k], dist);
				prob += p * (nbObjects[k] / sumnck);// probability of every point generated by each cluster
			}
			loglikelihood += Math.log(prob);
		}
		return loglikelihood;
	}

	private double computeProbaForQueryAndCluster(double sigma, double d) {
		double pqk = 0.0;
		if (Double.isNaN(sigma)) {
			// System.err.println("alert");
			if (d == 0)
				pqk = 1.0;
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
	
	public double[] getNck() {
		return nck;
	}
}
