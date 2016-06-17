package classif.DT;

import weka.core.Instance;
import weka.core.Instances;

public abstract class ClassifierSplitModel {

	/** Number of created subsets. */
	protected int m_numSubsets;
	public Instances m_splitPoint;
	protected Distribution m_distribution; 

	/**
	 * Allows to clone a model (shallow copy).
	 */
	public Object clone() {

		Object clone = null;

		try {
			clone = super.clone();
		} catch (CloneNotSupportedException e) {
		}
		return clone;
	}

	/**
	 * Builds the classifier split model for the given set of instances.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	public abstract void buildClassifier(Instances instances) throws Exception;

	/**
	 * Checks if generated model is valid.
	 */
	public final boolean checkModel() {

		if (m_numSubsets > 0)
			return true;
		else
			return false;
	}

	/**
	 * Classifies a given instance.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	public final double classifyInstance(Instance instance) throws Exception {

		int theSubset;

		theSubset = whichSubset(instance);
		return instance.classAttribute().indexOfValue(instance.classAttribute().value(theSubset));
	}

	/**
	 * Returns the number of created subsets for the split.
	 */
	public final int numSubsets() {

		return m_numSubsets;
	}

	/**
	 * Splits the given set of instances into subsets.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	public final Instances[] split(Instances data) throws Exception {

		Instances[] instances = new Instances[m_numSubsets];
		Instance instance;
		int subset, i, j;

		for (j = 0; j < m_numSubsets; j++)
			instances[j] = new Instances((Instances) data, data.numInstances());
		for (i = 0; i < data.numInstances(); i++) {
			instance = ((Instances) data).instance(i);
			subset = whichSubset(instance);
			instances[subset].add(instance);
		}
		return instances;
	}

	/**
	 * Returns index of subset instance is assigned to. Returns -1 if instance
	 * is assigned to more than one subset.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	public abstract int whichSubset(Instance instance) throws Exception;

	public double[] weights(Instance instance) {
		// TODO Auto-generated method stub
		return null;
	}

	public abstract Instances getSplitPoint();

	public abstract void setSplitPoint(Instances m_splitPoint);
	
}
