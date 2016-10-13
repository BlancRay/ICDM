/**
 *    Date : 2007-12-9
 *      by : Yang ZHANG  @ UQ
 *   Email : mokuram@itee.uq.edu.au
 *           zhangyang@nwsuaf.edu.cn
 *    
 * Project : Classification of Data Streams
 *
 * Description:
 *        There is no such LIB CLASS in WEKA that helps to cut a dataset into two parts.
 *        This class helps to do that.
 *        A sample of how to use this LIB CLASS is given in the main() method.
 */
package classif.pu;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.instance.*;

public class CutInto2 extends Resample
{
	private static final long serialVersionUID = -8835068474731851218L;

	protected TreeSet setA = new TreeSet();
	protected TreeSet setB = new TreeSet();

	/**
	 * @param args
	 */
	public static void main(String[] args)
	{
		try
		{
			Instances originalDataset = new Instances(
    				new BufferedReader(
    						new FileReader("ionosphere.arff")));
			originalDataset.setClassIndex(originalDataset.numAttributes() - 1);

			String strOption = "-S 100 -Z 25 -no-replacement";
			String[] options = Utils.splitOptions(strOption);
			CutInto2 resample = new CutInto2();
			resample.setOptions(options);
			resample.setInputFormat(originalDataset);
			Filter.useFilter(originalDataset, resample);

		}
		catch (Exception e)
		{
System.err.println(e);		}
	}

	public Instances[] randomize(int nRandomInt, int nPercent,
			Instances originalDataset)
	{
		try
		{
			originalDataset.setClassIndex(originalDataset.numAttributes() - 1);
			String strOption = "-S " + nRandomInt + " -Z " + nPercent
					+ " -no-replacement";
			String[] options = Utils.splitOptions(strOption);
			CutInto2 resample = new CutInto2();
			resample.setOptions(options);
			resample.setInputFormat(originalDataset);
			Filter.useFilter(originalDataset, resample);

			Instances[] two = new Instances[2];
			two[0] = resample.getPartZ(originalDataset);
			two[1] = resample.getPartNotZ(originalDataset);

			return two;
		}
		catch (Exception e)
		{
			System.err.println(e);
		}

		return null;
	}

	/**
	 * creates the subsample without replacement
	 * 
	 * @param random
	 *            the random number generator to use
	 * @param origSize
	 *            the original size of the dataset
	 * @param sampleSize
	 *            the size to generate
	 * @param actualClasses
	 *            the number of classes found in the data
	 * @param classIndices
	 *            the indices where classes start
	 */
	public void createSubsampleWithoutReplacement(Random random, int origSize,
			int sampleSize, int actualClasses, int[] classIndices)
	{

		if (sampleSize > origSize)
		{
			sampleSize = origSize;
			System.err
					.println("Resampling with replacement can only use percentage <=100% - "
							+ "Using full dataset!");
		}

		Vector<Integer>[] indices = new Vector[actualClasses];
		Vector<Integer>[] indicesNew = new Vector[actualClasses];

		// generate list of all indices to draw from
		for (int i = 0; i < actualClasses; i++)
		{
			indices[i] = new Vector<Integer>(classIndices[i + 1]
					- classIndices[i]);
			indicesNew[i] = new Vector<Integer>(indices[i].capacity());
			for (int n = classIndices[i]; n < classIndices[i + 1]; n++)
				indices[i].add(n);
		}

		// draw X samples
		int currentSize = origSize;
		for (int i = 0; i < sampleSize; i++)
		{
			int index = 0;
			if (random.nextDouble() < m_BiasToUniformClass)
			{
				// Pick a random class (of those classes that actually appear)
				int cIndex = random.nextInt(actualClasses);
				for (int j = 0, k = 0; j < classIndices.length - 1; j++)
				{
					if ((classIndices[j] != classIndices[j + 1])
							&& (k++ >= cIndex))
					{
						// Pick a random instance of the designated class
						index = random.nextInt(indices[j].size());
						indicesNew[j].add(indices[j].get(index));
						indices[j].remove(index);
						break;
					}
				}
			}
			else
			{
				index = random.nextInt(currentSize);
				for (int n = 0; n < actualClasses; n++)
				{
					if (index < indices[n].size())
					{
						indicesNew[n].add(indices[n].get(index));
						indices[n].remove(index);
						break;
					}
					else
					{
						index -= indices[n].size();
					}
				}
				currentSize--;
			}
		}

		// ADD BY ZHANG YANG
		// -------------------------------------------------------
		readIndexes(indicesNew, setA);
		readIndexes(indices, setB);
		// ADD BY ZHANG YANG END
		// -------------------------------------------------------
		// sort indices
		if (getInvertSelection())
		{
			indicesNew = indices;
		}
		else
		{
			for (int i = 0; i < indicesNew.length; i++)
				Collections.sort(indicesNew[i]);
		}

		// add to ouput
		for (int i = 0; i < indicesNew.length; i++)
		{
			for (int n = 0; n < indicesNew[i].size(); n++)
				push((Instance) getInputFormat().instance(indicesNew[i].get(n))
						.copy());
		}

		// clean up
		for (int i = 0; i < indices.length; i++)
		{
			indices[i].clear();
			indicesNew[i].clear();
		}
		indices = null;
		indicesNew = null;
	}

	// ADD BY ZHANG YANG -------------------------------------------------------
	protected void readIndexes(Vector[] v, Set set)
	{
		for (int i = 0; i < v.length; i++)
		{
			for (int j = 0; j < v[i].size(); j++)
			{
				set.add(v[i].get(j));
			}
		}
	}

	protected Instances getPart(TreeSet set, Instances dataset)
	{
		Instances newDataset = new Instances(dataset, 10);

		Iterator it = set.iterator();
		while (it.hasNext())
		{
			int index = ((Integer) it.next()).intValue();
			newDataset.add(dataset.instance(index));
		}

		return newDataset;
	}

	public Instances getPartZ(Instances dataset)
	{
		return getPart(setA, dataset);
	}

	public Instances getPartNotZ(Instances dataset)
	{
		return getPart(setB, dataset);
	}
	// ADD BY ZHANG YANG END
	// -------------------------------------------------------
}