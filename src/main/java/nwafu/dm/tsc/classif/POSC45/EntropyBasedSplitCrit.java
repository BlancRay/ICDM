/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    EntropyBasedSplitCrit.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

package nwafu.dm.tsc.classif.POSC45;

import weka.core.Utils;

/**
 * "Abstract" class for computing splitting criteria based on the entropy of a
 * class distribution.
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.7 $
 */
public abstract class EntropyBasedSplitCrit extends SplitCriterion
{

	/** for serialization */
	private static final long serialVersionUID = -2618691439791653056L;

	/** The log of 2. */
	protected static double log2 = Math.log(2);

	/**
	 * Help method for computing entropy. ����log
	 */
	public final double logFunc(double num)
	{
		// Constant hard coded for efficiency reasons
		if (num < 1e-6)
			return 0;
		else
			return num * Math.log(num) / log2;
	}

	/**
	 * Computes entropy of distribution before splitting.
	 */
	public final double oldEnt(Distribution bags)
	{
		double d = OneClassEntropy(bags.perClass(0), bags.perClass(1));

		//Out.println("POS SAMPLES : " + bags.perClass(0));
		///Out.println("UN  SAMPLES : " + bags.perClass(1));
		
		//Out.println("ԭʼ�أ�" + d);
		
		//Out.println("=====================================================");
	
		return d;
	}

	/**
	 * Computes entropy of distribution after splitting.
	 */
	public final double newEnt(Distribution bags)
	{

		double returnValue = 0;
		int i, j;

		for (i = 0; i < bags.numBags(); i++)
		{
			double d = OneClassEntropy(bags.perClassPerBag(i, 0), bags
					.perClassPerBag(i, 1));
			returnValue = returnValue + bags.perBag(i) / bags.total() * d;
			//Out.println("new Entropy : " + bags.perBag(i) + "*" + d + "="
					//+ bags.perBag(i) * d);
		}
		
		//Out.println("=====================================================");
		//Out.println();
		//Out.println();
		//Out.println();

		return returnValue;
	}

	/**
	 * Computes entropy after splitting without considering the class values.
	 */
	public final double splitEnt(Distribution bags)
	{

		double returnValue = 0;
		int i;

		for (i = 0; i < bags.numBags(); i++)
			returnValue = returnValue + logFunc(bags.perBag(i));
		return logFunc(bags.total()) - returnValue;
	}

	public double OneClassEntropy(double dPosWeight, double dUnWeight)
	{
		
		//Out.println("==============POSC45���㹫ʽ�Ĳ�����====================");
		//Out.println("��ǰ����������Ŀ��" + dPosWeight);
		//Out.println("��������Ŀ��" + C45PosUnl.nPosSize);
		//Out.println("δ��ע���������ݣ�" + C45PosUnl.nUnSize);
		//Out.println("��ǰ��δ��ע��������Ŀ��" + dUnWeight);
		//Out.println("��ǰ��D(f)��" + C45PosUnl.dDF);
		
		
		
		double p1 = (dPosWeight / ClassifyPOSC45.nPosSize) * ClassifyPOSC45.dDF
				* (ClassifyPOSC45.nUnSize / dUnWeight);
		if ((p1 > 1) || (Utils.eq(dUnWeight, 0)))
			p1 = 1;

		double p0 = 1 - p1;

		double d = -logFunc(p0) - logFunc(p1);
		
		
		//Out.println("p0 : " + p0 + " p1 : " + p1 + " POS WEIGHT : " + dPosWeight + " NEG WEIGHT : " + dUnWeight
		//		+ " ENTROPY : " + d);
		// Exception e=new Exception();
		// e.printStackTrace();
		return d;
	}
}
