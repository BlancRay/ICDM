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
package classif.kmeans;

import items.ClassedSequence;

import java.util.ArrayList;

import classif.Prototyper;
import weka.core.Instances;

public class DTWKNNClassifierKMeans extends Prototyper {

	private static final long serialVersionUID = 1717176683182910935L;
	
	public DTWKNNClassifierKMeans() {
		super();
	}

	@Override
	protected void buildSpecificClassifier(Instances data) {
		
		ArrayList<String> classes = new ArrayList<String>(classedData.keySet());
		
		for (String clas : classes) {
			// if the class is empty, continue
			if(classedData.get(clas).isEmpty()) 
				continue;
			KMeansSymbolicSequence kmeans = new KMeansSymbolicSequence(nbPrototypesPerClass, classedData.get(clas));
			kmeans.cluster();
			for (int i = 0; i < kmeans.centers.length; i++) {
				if(kmeans.centers[i]!=null){ //~ if empty cluster
					ClassedSequence s = new ClassedSequence(kmeans.centers[i], clas);
					prototypes.add(s);
				}
			}
		}
	}
}
