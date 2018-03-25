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
package nwafu.dm.tsc.classif;

import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.ClassedSequence;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.StringTokenizer;

import weka.core.Instances;

public class PrototyperUtil {
		
	public static int[] getPrototypesPerClassDistribution(ArrayList<ClassedSequence> prototypes, Instances train) {
		HashMap<String,Integer> distrib = new HashMap<String,Integer>();
		
		Enumeration classValues = train.classAttribute().enumerateValues();
		List<String> list = Collections.list(classValues);
        Collections.sort(list);
        
		for (String integer : list) {
			distrib.put(integer,0);
		}
		
		for (ClassedSequence symbolicSequenceClassed : prototypes) {
			String classValue = symbolicSequenceClassed.classValue;
			distrib.put(classValue,distrib.get(classValue)+1);
		}
		int[] classDistribution = new int[list.size()];
		for (int i = 0; i < classDistribution.length; i++) {
			classDistribution[i] = distrib.get(list.get(i));
		}
		
		return classDistribution;
	}
	
	public static ArrayList<ClassedSequence> loadPrototypes(String path) {
		ArrayList<ClassedSequence> prototypes = new ArrayList<ClassedSequence>();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));
			String line;
			while ((line = br.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(line,";");
				String classValue = st.nextToken();
				ArrayList<MonoDoubleItemSet> tmpSeq = new ArrayList<MonoDoubleItemSet>();
				while(st.hasMoreTokens()) {
					double value = Double.parseDouble(st.nextToken());
					tmpSeq.add(new MonoDoubleItemSet(value));
				}
				MonoDoubleItemSet[] realSeq = new MonoDoubleItemSet[tmpSeq.size()];
				for (int i = 0; i < tmpSeq.size(); i++) {
					realSeq[i] = tmpSeq.get(i);
				}
				prototypes.add(new ClassedSequence(new Sequence(realSeq),classValue));
				if (classValue == null) {
					System.out.println("NULL");
				}
			}
			br.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return prototypes;
	}
	
	public static void savePrototypes(ArrayList<ClassedSequence> prototypes, String path) {		
		savePrototypes(prototypes, path, false);
	}
	
	public static void savePrototypes(ArrayList<ClassedSequence> prototypes, String path, boolean append) {		
		try {
			PrintStream out = new PrintStream(new BufferedOutputStream(new FileOutputStream(path)), append);
			
			for (ClassedSequence seq : prototypes) {
				out.print(seq.classValue+";");
				for (int i = 0; i < seq.sequence.sequence.length; i++) {
					MonoDoubleItemSet item = (MonoDoubleItemSet)seq.sequence.sequence[i];
					out.print(item.getValue());
					if(i < seq.sequence.sequence.length-1)
						out.print(";");
				}
				out.print("\n");
			}
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
