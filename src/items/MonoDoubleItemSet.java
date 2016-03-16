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
package items;
import static java.lang.Math.abs;
public class MonoDoubleItemSet extends Itemset implements java.io.Serializable {
	private static final long serialVersionUID = 5103879297281957601L;
	
	double value;
	
	public MonoDoubleItemSet(double value){
		this.value = value;
	}
	
	@Override
	public Itemset clone() {
		return new MonoDoubleItemSet(value);
	}

	@Override
	public double distance(Itemset o) {
		MonoDoubleItemSet o1 = (MonoDoubleItemSet)o;
		return abs(o1.value-value);
	}
	
	public Itemset pow2() {
		return new MonoDoubleItemSet(value*value);
	}

	@Override
	public Itemset mean(Itemset[] tab) {
		if (tab.length < 1) {
			throw new RuntimeException("Empty tab");
		}
		double sum = 0.0;
		for (Itemset itemset : tab) {
			MonoDoubleItemSet item = (MonoDoubleItemSet)itemset;
			sum += item.value;
		}
		return new MonoDoubleItemSet(sum / tab.length);
	}
	
	@Override
	public Itemset weightmean(Itemset[] tab,double weight) {
		if (tab.length < 1) {
			throw new RuntimeException("Empty tab");
		}
		double sum = 0.0;
		for (Itemset itemset : tab) {
			MonoDoubleItemSet item = (MonoDoubleItemSet)itemset;
			sum += item.value*weight;
		}
		return new MonoDoubleItemSet(sum / tab.length);
	}

	@Override
	public String toString() {
		return new Double(value).toString();
	}
	
	public double getValue(){
		return value;
	}
}
