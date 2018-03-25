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

public class Tools {
	public final static double Min3(final double a, final double b, final double c) {
		return (a <= b) ? ((a <= c) ? a : c) : (b <= c) ? b : c;
	}

	public static int ArgMin3(final double a, final double b, final double c) {
		return (a <= b) ? ((a <= c) ? 0 : 2) : (b <= c) ? 1 : 2;
	}
	
	public static double sum(final double... tab) {
		double res = 0.0;
		for (double d : tab)
			res += d;
		return res;
	}
}
