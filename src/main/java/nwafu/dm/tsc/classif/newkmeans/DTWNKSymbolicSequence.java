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
package nwafu.dm.tsc.classif.newkmeans;

import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.Sequences;
import weka.core.DistanceFunction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.random.RandomDataGenerator;

import nwafu.dm.tsc.classif.kmeans.KMeansSymbolicSequence;

public class DTWNKSymbolicSequence {
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
	double[][] dist_i_for_k=null;
	double[][] weight=null;
	double[] sum_weight_i_for_k=null;

	public DTWNKSymbolicSequence(int nbClusters, ArrayList<Sequence> data, int dataAttributes) {
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
				if (Eachaffectation.size() <= minObj) {
					isBig = false;
					break;
				}
			}
			runtime++;
		} while (isBig == false);

		sigmasPerCluster = new double[nbClusters];
		nck = new double[nbClusters];
		sumnck = data.size();
		dist_i_for_k=new double[sumnck][nbClusters];
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

		for (int i = 0; i < 10; i++) {
			gmmprocess(i);
		}
	}


	
	private void gmmprocess(int iteration) {
		weight = new double[sumnck][nbClusters];
		sum_weight_i_for_k= new double[nbClusters];
		double[][] p =new double[sumnck][nbClusters];
		double[] sum_p_k_for_i=new double[sumnck];
		
		//give init weight
		ArrayList<Sequence> sequencesForClass = data;
		// for each data point computer gamma
		for (int i = 0; i < sequencesForClass.size(); i++) {
			Sequence s = sequencesForClass.get(i);
			// gamma(i,k)
			for (int k = 0; k < centroidsPerCluster.length; k++) {
				dist_i_for_k[i][k]=s.distance(centroidsPerCluster[k]);
				weight[i][k] = 1.0 / nbClusters;
				sum_weight_i_for_k[k]+= weight[i][k];
			}
		}
		
		//recaculate weight
		for (int i = 0; i < sequencesForClass.size(); i++) {
			Sequence s = sequencesForClass.get(i);
			for (int k = 0; k < centroidsPerCluster.length; k++) {
				double distance=s.distance(centroidsPerCluster[k]);
				p[i][k] = computeProbaForQueryAndCluster(weight,dist_i_for_k, distance,sum_weight_i_for_k[k],k);
				sum_p_k_for_i[i]+=p[i][k];
			}
		}
		sum_weight_i_for_k= new double[nbClusters];
		for (int i = 0; i < sequencesForClass.size(); i++) {
			for (int k = 0; k < centroidsPerCluster.length; k++) {
				weight[i][k] = p[i][k] / sum_p_k_for_i[i];
				sum_weight_i_for_k[k]+=weight[i][k];
			}
		}
		
		//updata weight or centroid
		// centroidsPerClass
		for (int k = 0; k < centroidsPerCluster.length; k++) {
			centroidsPerCluster[k]= Sequences.weightMean(centroidsPerCluster[k],data.toArray(new Sequence[0]), weight,k,iteration);
		}
		for (int i = 0; i < sequencesForClass.size(); i++) {
			Sequence s = sequencesForClass.get(i);
			for (int k = 0; k < centroidsPerCluster.length; k++) {
				dist_i_for_k[i][k]=s.distance(centroidsPerCluster[k]);
			}
		}
	}

	private double computeProbaForQueryAndCluster(double[][] gamma, double[][] d, double dist, double sumweightk,int k) {
		double sumweight =0;
		for (int i = 0; i <data.size() ; i++) {
			if(d[i][k]<=dist){
				sumweight+=gamma[i][k];
			}
		}
		return (1-sumweight/sumweightk);
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

	public double[][] getDist() {
		return dist_i_for_k;
	}

	public double[][] getGamma() {
		return weight;
	}

	public double[] getSumofgammak() {
		return sum_weight_i_for_k;
	}

}
