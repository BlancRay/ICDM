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
package nwafu.dm.tsc.classif.Majority;

import nwafu.dm.tsc.items.ClassedSequence;
import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.Sequences;
import weka.core.Attribute;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.math3.random.RandomDataGenerator;

public class KMeansSymbolicSequence {
	public int nbClusters;
	public ArrayList<ClassedSequence> data;
	public ClassedSequence[] centers;
	public int[] clusterMap;
	public RandomDataGenerator randGen;
	public ArrayList<ClassedSequence>[] affectation;
	private Attribute attributes;

	public KMeansSymbolicSequence(int nbClusters,ArrayList<ClassedSequence> data,Attribute attribute) {
		this.data = data;
		this.clusterMap = new int[data.size()];
		this.randGen = new RandomDataGenerator();
		this.attributes=attribute;
		this.nbClusters=nbClusters;
	}

	public void cluster() {
		centers = new ClassedSequence[nbClusters];
		affectation = new ArrayList[nbClusters];

		
		// pickup centers
		int[] selected = randGen.nextPermutation(data.size(), nbClusters);
		for (int i = 0; i < selected.length; i++) {
			centers[i]=new ClassedSequence(data.get(selected[i]).sequence,data.get(selected[i]).classValue);
		}

		// for each iteration i
		for (int i = 0; i < 15; i++) {
			// init
			for (int k = 0; k < affectation.length; k++) {
				affectation[k] = new ArrayList<ClassedSequence>();
			}
			// for each data point j
			for (int j = 0; j < data.size(); j++) {

				double minDist = Double.MAX_VALUE;
				// for each cluster k
				for (int k = 0; k < centers.length; k++) {
					// distance between cluster k and data point j
					double currentDist = centers[k].sequence.distance(data.get(j).sequence);
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
					ArrayList<Sequence> copyaffectation = new ArrayList<Sequence>();
					for (ClassedSequence eachClassedSequence : affectation[j]) {
						copyaffectation.add(eachClassedSequence.sequence);
					}
					centers[j].sequence = Sequences.mean(copyaffectation.toArray(new Sequence[0]));
					centers[j].classValue=MajorityClass(affectation[j]);
				}
			}
		}
	}
	
	protected String MajorityClass(ArrayList<ClassedSequence> Sequences){
		String label=null;
		int[] labels = new int[attributes.numValues()];
		for (ClassedSequence s : Sequences) {
			String classValue=s.classValue;
			labels[(int) Double.parseDouble(classValue)]++;
		}
		//find majority of labels
		int majority=0;
		int max=0;
		for (int i = 0; i < labels.length; i++) {
			if(labels[i]>max)
				majority=i;
		}
		label=attributes.value(majority);
		return  label;
	}
}
