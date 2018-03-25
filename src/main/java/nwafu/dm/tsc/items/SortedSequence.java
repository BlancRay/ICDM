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
package nwafu.dm.tsc.items;

public class SortedSequence implements Comparable<Object> {
	public Sequence sequence;
	public int index;
	public double sortingValue;
	public String classValue;
	
	public SortedSequence(Sequence sequence,int index, double sortingValue, String classValue) {
		this.sequence = sequence;
		this.index = index;
		this.sortingValue = sortingValue;
		this.classValue = classValue;
	}

	public int compareTo(Object o) {
		SortedSequence tmp = (SortedSequence)o;
		int res = 0;
		if(tmp.sortingValue-sortingValue > 0) res = 1;
		if(tmp.sortingValue-sortingValue < 0) res = -1;
		return res;
	}
	
	public String toString() {
		return index+" "+sortingValue;
	}
}