package nwafu.dm.tsc.classif.BIGDT;

import java.io.Serializable;
import java.util.Stack;
import nwafu.dm.tsc.items.Pairs;
import weka.core.Instance;
import weka.core.Instances;

public abstract class ClassifierSplitModel implements Serializable{
	private static final long serialVersionUID = 758462424142684777L;
	/** Number of created subsets. */
	protected int m_numSubsets;
	protected Instances m_splitPoint;
	protected Distribution m_distribution;
//	protected Stack<Pairs> m_Pairs;

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
	 * Gets class probability for instance.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	public double classProb(int classIndex, Instance instance, int theSubset) throws Exception {

		if (theSubset > -1) {
			return m_distribution.prob(classIndex, theSubset);
		} else {
			double[] weights = weights(instance);
			if (weights == null) {
				return m_distribution.prob(classIndex);
			} else {
				double prob = 0;
				for (int i = 0; i < weights.length; i++) {
					prob += weights[i] * m_distribution.prob(classIndex, i);
				}
				return prob;
			}
		}
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
		return null;
	}
	  /**
	   * Sets distribution associated with model.
	   */
	  public void resetDistribution(Instances data) throws Exception {

	    m_distribution = new Distribution(data, this);
	  }
	  /**
	   * Returns the distribution of class values induced by the model.
	   */
	  public final Distribution distribution() {

	    return m_distribution;
	  }

	public abstract Instances getSplitPoint();

	public abstract void setSplitPoint(Instances m_splitPoint);
	
}
