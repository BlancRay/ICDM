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

import nwafu.dm.tsc.classif.Prototyper;
import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.ClassedSequence;
import weka.core.Instances;

public class DTWKNNClassifierKMedoids extends Prototyper {
	private static final long serialVersionUID = -3409118188128863227L;

	public DTWKNNClassifierKMedoids() {
		super();
	}

	@Override
	protected void buildSpecificClassifier(Instances data){
		for (String clas : classedData.keySet()) {
			// if the class is empty, continue
			if(classedData.get(clas).isEmpty()) 
				continue;
			KMedoidsSymbolicSequence kmedoids = new KMedoidsSymbolicSequence(nbPrototypesPerClass, classedData.get(clas));
			kmedoids.cluster();
			Sequence[]centers = kmedoids.getCenters();
			for (int i = 0; i < centers.length; i++) {
				ClassedSequence s = new ClassedSequence(centers[i], clas);
				prototypes.add(s);
			}
		}
		
	}
}
