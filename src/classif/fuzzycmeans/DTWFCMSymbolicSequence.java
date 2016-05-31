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
package classif.fuzzycmeans;

import items.MonoDoubleItemSet;
import items.Sequence;
import items.Sequences;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.random.RandomDataGenerator;

import classif.kmeans.KMeansSymbolicSequence;

public class DTWFCMSymbolicSequence {
	public int nbClusters;
	final ArrayList<Sequence> data;
	public RandomDataGenerator randGen;

	protected Sequence[] centroidsPerCluster = null;
	protected int dataAttributes;

	private static final double threshold = Math.pow(10,-6);
	private static final double minObj = 2;
	private double[] nck = null;

	public DTWFCMSymbolicSequence(int nbClusters, ArrayList<Sequence> data, int dataAttributes) {
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
			KMeansSymbolicSequence kmeans = new KMeansSymbolicSequence(nbClusters,data);
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
		
		nck = new double[nbClusters];
		
		for (int k = 0; k < nbClusters; k++) {
			if (centroidsPerCluster[k] != null) { // ~ if empty cluster
				nck[k] = affectation[k].size();
			} else
				System.err.println("ERROR");
		}
		
		fcm();
	}

	private void fcm() {

		double[][] uij = new double[data.size()][nbClusters];
		double m = 2;
		ArrayList<Sequence> sequencesForClass = data;
		// init uij
//		for (int i = 0; i < uij.length; i++) {
//			for (int j = 0; j < uij[i].length; j++) {
//				uij[i][j] = 1 / nck[j];
//			}
//		}
		for (int i = 0; i < uij.length; i++) {
			Sequence s = sequencesForClass.get(i);
			for (int j = 0; j < uij[i].length; j++) {
				double d1 = s.distance(centroidsPerCluster[j]);
				double uik = 0;
				for (int k = 0; k < nbClusters; k++) {
					double d2 = s.distance(centroidsPerCluster[k]);
					if(d2==0||d1==0){
						System.out.println("d1	"+d1+"	d2	"+d2);
						System.out.println("uik	"+uik);
						System.out.println("c1	"+centroidsPerCluster[j]);
						System.out.println("c2	"+centroidsPerCluster[k]);
						System.out.println("s	"+s);
						System.out.println(nck[k]);
						
					}
					uik += Math.pow((d1 / d2), (2 / (m - 1)));
				}
				uij[i][j] = 1/uik;
//				System.out.println("init	"+i+"	"+j+"	"+uij[i][j]);
			}
		}

		// calculate centroids
		double diff = 0;
		int iteration=0;
		do {
			double[][] uij_1 = new double[data.size()][nbClusters];
			for (int k = 0; k < nbClusters; k++) {
				centroidsPerCluster[k] = Sequences.weightMean(centroidsPerCluster[k], data.toArray(new Sequence[0]),
						uij, k, 0);
			}
			// updata uij

			for (int i = 0; i < uij.length; i++) {
				Sequence s = sequencesForClass.get(i);
				for (int j = 0; j < uij[i].length; j++) {
					double d1 = s.distance(centroidsPerCluster[j]);
					double uik = 0;
					for (int k = 0; k < nbClusters; k++) {
						double d2 = s.distance(centroidsPerCluster[k]);
						uik += Math.pow((d1 / d2), (2 / (m - 1)));
					}
					uij_1[i][j] = 1/uik;
				}
			}
			diff = maxdiff(uij_1, uij);
			uij = uij_1;
			iteration++;
		} while (diff >= threshold&&iteration<=20);
	}

	private double maxdiff(double[][] m1, double[][] m2) {
		double max = 0;
		for (int i = 0; i < m1.length; i++) {
			for (int j = 0; j < m1[i].length; j++) {
				double diff = Math.abs(m1[i][j] - m2[i][j]);
				if (diff > max) {
					max = diff;
				}
			}
		}
		return max;
	}
	
	public Sequence[] getMus() {
		return centroidsPerCluster;
	}

	public double[] getNck() {
		return nck;
	}
}
