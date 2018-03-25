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
package nwafu.dm.tsc.classif.dropx;

import nwafu.dm.tsc.items.ClassedSequence;

import java.util.ArrayList;
import java.util.HashMap;

import weka.core.Attribute;
import weka.core.Instances;
import nwafu.dm.tsc.classif.Prototyper;

public abstract class PrototyperSorted extends Prototyper {
	private static final long serialVersionUID = 5184464370101078127L;
	
	public static ArrayList<ClassedSequence> sortedSequences = null;
	public boolean isNbPrototypesPerClass = true;
	
	public static HashMap<String,ArrayList<ClassedSequence>> sortedSequencesByClass = null;
	
	public static void reset() {
		sortedSequences = null;
	}
	
	public void setIsNbPrototypesPerClass(boolean isNbPrototypesPerClass){
		this.isNbPrototypesPerClass = isNbPrototypesPerClass;
	}
	
	@Override
	protected void buildSpecificClassifier(Instances data) {
		if(sortedSequences == null) {
			buildSortedSequences(data);
		}
		
		int max = nbPrototypesPerClass;
		if(isNbPrototypesPerClass) {
			max = this.nbPrototypesPerClass * data.numClasses();
		}
				
		// add the number of required prototypes
		for (int i = 0; i < sortedSequences.size() && i < max; i++) {
			prototypes.add(sortedSequences.get(i));
		}
	}
	
	/**
	 * Balance the classes in the train test
	 * @param data
	 */
	protected void buildSpecificClassifierByClass(Instances data) {
		if(sortedSequences == null) {
			buildSortedSequences(data);
			buildSortedSequencesPerClass(data);
		}
		
		int max = nbPrototypesPerClass;
		if(isNbPrototypesPerClass) {
			max = this.nbPrototypesPerClass * data.numClasses();
		}
		
		Attribute classAttribute = data.classAttribute();
		for (int i = 0; i < classAttribute.numValues(); i++) {
			ArrayList<ClassedSequence> tmpClassObj = sortedSequencesByClass.get(classAttribute.value(i));
			for (int j = 0; j < max & j < tmpClassObj.size() ; j++) {
				prototypes.add(tmpClassObj.get(j));
			}
		}
	}
	
	private void buildSortedSequencesPerClass(Instances data) {
		sortedSequencesByClass = new HashMap<String,ArrayList<ClassedSequence>>();
		Attribute classAttribute = data.classAttribute();
		
		for (int i = 0; i < classAttribute.numValues(); i++) {
			sortedSequencesByClass.put(classAttribute.value(i), new ArrayList<ClassedSequence>());
		}
		
		for (int i = 0; i < sortedSequences.size(); i++) {
			ClassedSequence instance = sortedSequences.get(i);
			sortedSequencesByClass.get(instance.classValue).add(instance);
		}
	}
	
	protected abstract void buildSortedSequences(Instances data);
}
