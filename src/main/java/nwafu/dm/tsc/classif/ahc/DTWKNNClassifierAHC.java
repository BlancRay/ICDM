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
package nwafu.dm.tsc.classif.ahc;

import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.ClassedSequence;

import java.util.ArrayList;
import java.util.HashMap;

import nwafu.dm.tsc.classif.Prototyper;
import weka.core.Instances;

public class DTWKNNClassifierAHC extends Prototyper {
	
	private static final long serialVersionUID = 3487280631642630599L;
	private transient HashMap<String, AHCSymbolicSequence> ahcs;
	public DTWKNNClassifierAHC() {
		super();
		ahcs = null;
	}
	
	protected void initAHCS() {
		ahcs = new HashMap<String, AHCSymbolicSequence>();
		ArrayList<String> classes = new ArrayList<String>(classedData.keySet());
		
		for (String clas : classes) {
			System.out.println("Compute all clusterings for class "+clas);
			// if the class is empty, continue
			if(classedData.get(clas).isEmpty()) 
				continue;
			AHCSymbolicSequence ahc = new AHCSymbolicSequence(classedData.get(clas));
			ahc.cluster();
			ahcs.put(clas, ahc);
		}
	}

	@Override
	protected void buildSpecificClassifier(Instances data) {
		if(ahcs ==null){
			initAHCS();
		}
		
		ArrayList<String> classes = new ArrayList<String>(classedData.keySet());
		
		for (String clas : classes) {
			// if the class is empty, continue
			if(classedData.get(clas).isEmpty()) 
				continue;
			
			AHCSymbolicSequence ahc = ahcs.get(clas);
			ArrayList<Sequence> centers;
			if(nbPrototypesPerClass<ahc.centroidsForNumberOfClusters.length){
				centers = ahc.centroidsForNumberOfClusters[nbPrototypesPerClass];
			}else{
				centers = ahc.centroidsForNumberOfClusters[ahc.centroidsForNumberOfClusters.length-1];
			}
			
			
			for (int i = 0; i < centers.size(); i++) {
				if(centers.get(i)!=null){
					ClassedSequence s = new ClassedSequence(centers.get(i), clas);
					prototypes.add(s);
				}
			}
		}
	}
}
