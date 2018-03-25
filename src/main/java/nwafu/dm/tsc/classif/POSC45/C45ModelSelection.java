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
 *    C45ModelSelection.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

package nwafu.dm.tsc.classif.POSC45;

import java.util.Enumeration;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
/**
 * Class for selecting a C4.5-type split for a given dataset.
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.10 $
 */
public class C45ModelSelection extends ModelSelection
{

	/** for serialization */
	private static final long serialVersionUID = 3372204862440821989L;

	/** Minimum number of objects in interval. */
	private int m_minNoObj;

	/** All the training data */
	private Instances m_allData; // 

	/**
	 * Initializes the split selection method with the given parameters.
	 * 
	 * @param minNoObj
	 *            minimum number of instances that have to occur in at least two
	 *            subsets induced by split
	 * @param allData
	 *            FULL training dataset (necessary for selection of split
	 *            points).
	 */
	public C45ModelSelection(int minNoObj, Instances allData)
	{
		m_minNoObj = minNoObj;
		m_allData = allData;
	}

	/**
	 * Sets reference to training data to null.
	 */
	public void cleanup()
	{

		m_allData = null;
	}

	/**
	 * Selects C4.5-type split for the given dataset.
	 */
	public final ClassifierSplitModel selectModel(Instances data)
	{

		//Out.println("�����Ѿ�����C45ModelSelection�Ĳ��֣������Ҫ��ɸ��������������ķ��ѡ�");
		
		double minResult;
//		double currentResult;
		C45Split[] currentModel;
		C45Split bestModel = null;
		NoSplit noSplitModel = null;
		double averageInfoGain = 0;
		int validModels = 0;
		boolean multiVal = true;
		Distribution checkDistribution;
		Attribute attribute;
		double sumOfWeights;
		int i;

		try
		{

			// Check if all Instances belong to one class or if not
			// enough Instances to split.
			checkDistribution = new Distribution(data);
			noSplitModel = new NoSplit(checkDistribution);
			if (Utils.sm(checkDistribution.total(), 2 * m_minNoObj)
					|| Utils.eq(checkDistribution.total(), checkDistribution
							.perClass(checkDistribution.maxClass())))
				return noSplitModel;

			// Check if all attributes are nominal and have a
			// lot of values.
			if (m_allData != null)
			{
				Enumeration<Attribute> enu = data.enumerateAttributes();
				while (enu.hasMoreElements())
				{
					attribute = (Attribute) enu.nextElement();
					if ((attribute.isNumeric())
							|| (Utils.sm((double) attribute.numValues(),
									(0.3 * (double) m_allData.numInstances()))))
					{
						multiVal = false;
						break;
					}
				}
			}

			currentModel = new C45Split[data.numAttributes()];
			sumOfWeights = data.sumOfWeights();

			//Out.println("�������Ե���Ŀ����ѭ�����ҵ���õ����ԣ����Եĸ�����" + data.numAttributes());
			
			// For each attribute.
			//Out.println("==== IN ONE SPLIT ==================================== {");
			//Out.println(noSplitModel);
			for (i = 0; i < data.numAttributes(); i++)
			{
				//Out.println("==== SUB-SPLIT " + i + "==================================== {");
			
				//Out.println("===================��" + i + "�����Կ�ʼ======================================");
				
				// Apart from class attribute.
				if (i != (data).classIndex())
				{

					// Get models for current attribute.
					currentModel[i] = new C45Split(i, m_minNoObj, sumOfWeights);
					currentModel[i].buildClassifier(data);

					// Check if useful split for current attribute
					// exists and check for enumerated attributes with
					// a lot of values.
					if (currentModel[i].checkModel())
						if (m_allData != null)
						{
							if ((data.attribute(i).isNumeric())
									|| (multiVal || Utils.sm((double) data
											.attribute(i).numValues(),
											(0.3 * (double) m_allData
													.numInstances()))))
							{
								averageInfoGain = averageInfoGain
										+ currentModel[i].infoGain();
								validModels++;
							}
						}
						else
						{
							averageInfoGain = averageInfoGain
									+ currentModel[i].infoGain();
							validModels++;
						}
				}
				else
				{
					currentModel[i] = null;
				}
				if (currentModel[i] != null)
				{
					//Out.println("InfoGain  : " + currentModel[i].infoGain());
					//Out.println("GainRatio : " + currentModel[i].gainRatio());
					//Out.println("ValidModel: " + validModels);
				}
				//Out.println("==== SUB-SPLIT " + i
				//		+ "==================================== }");
				
				//Out.println("===================��" + i + "�����Խ���======================================");
				
				//Out.println();

			}

			// Check if any useful split was found.
			if (validModels == 0)
				return noSplitModel;
			averageInfoGain = averageInfoGain / (double) validModels;

			// Find "best" attribute to split on.
			minResult = 0;
			for (i = 0; i < data.numAttributes(); i++)
			{
				if ((i != (data).classIndex())
						&& (currentModel[i].checkModel()))
				{
					//Out.println(currentModel[i].infoGain() + "  "
					//		+ (averageInfoGain - 1E-3) + "   " + minResult);
					// Use 1E-3 here to get a closer approximation to the
					// original
					// implementation.
					if ((currentModel[i].infoGain() >= (averageInfoGain - 1E-3))
							&& Utils.gr(currentModel[i].gainRatio(), minResult))
					{
						bestModel = currentModel[i];
						minResult = currentModel[i].gainRatio();
					}
				}
			}

			// Check if useful split was found.
			if (Utils.eq(minResult, 0))
			{
				//Out.println("NO BEST SPLIT, minResult=" + minResult);
				//Out.println("AVERAGE-GAIN : " + averageInfoGain);
				//Out.println("==== IN ONE SPLIT ==================================== }");
				return noSplitModel;
			}

			// Add all Instances with unknown values for the corresponding
			// attribute to the distribution for the model, so that
			// the complete distribution is stored with the model.
			bestModel.distribution().addInstWithUnknown(data,
					bestModel.attIndex());

			// Set the split point analogue to C45 if attribute numeric.
			if (m_allData != null)
				bestModel.setSplitPoint(m_allData);
			//Out.println("BEST SPLIT IS : ");
			//Out.println("==== IN ONE SPLIT ==================================== }");

			return bestModel;
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * Selects C4.5-type split for the given dataset.
	 */
	public final ClassifierSplitModel selectModel(Instances train,
			Instances test)
	{

		return selectModel(train);
	}
}
