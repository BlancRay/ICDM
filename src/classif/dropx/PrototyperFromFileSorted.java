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
package classif.dropx;

import items.ClassedSequence;

import java.util.ArrayList;

import classif.Prototyper;
import weka.core.Capabilities;
import weka.core.Instances;

public class PrototyperFromFileSorted extends Prototyper {
	private static final long serialVersionUID = 3763722724847808466L;
	public ArrayList<ClassedSequence> sortedSequences = null;
	
	public PrototyperFromFileSorted(ArrayList<ClassedSequence> prototypes) {
		this.sortedSequences = prototypes;
	}
	
	@Override
	protected void buildSpecificClassifier(Instances data) {
		int max = nbPrototypesPerClass;
		
		prototypes.clear();
		// add the number of required prototypes
		for (int i = 0; i < sortedSequences.size() && i < max; i++) {
			prototypes.add(sortedSequences.get(i));
		}
	}
}
