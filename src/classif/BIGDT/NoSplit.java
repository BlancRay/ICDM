
package classif.BIGDT;

import items.MonoDoubleItemSet;
import items.Sequence;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;

public final class NoSplit extends ClassifierSplitModel {

	/**
	 * Creates "no-split"-split for given distribution.
	 */
	public NoSplit(Distribution distribution) {

		m_distribution = new Distribution(distribution);
		m_numSubsets = 1;
	}

	/**
	 * Creates a "no-split"-split for a given set of instances.
	 *
	 * @exception Exception
	 *                if split can't be built successfully
	 */
	public final void buildClassifier(Instances instances) throws Exception {

		m_distribution = new Distribution(instances);
		m_numSubsets = 1;
	}

	/**
	 * Always returns splitPoint class label because only there is only one subset.
	 */
	public final int whichSubset(Instance instance) {
		int classlable = -1;
		Sequence[] splitsequences = new Sequence[getSplitPoint().numInstances()];
		for (int i = 0; i < splitsequences.length; i++) {
			Instance splitInstance = getSplitPoint().instance(i);
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[splitInstance.numAttributes() - 1];
			int shift = (splitInstance.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < sequence.length; t++) {
				sequence[t] = new MonoDoubleItemSet(splitInstance.value(t + shift));
			}
			splitsequences[i] = new Sequence(sequence);
		}

		MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[instance.numAttributes() - 1];
		int shift = (instance.classIndex() == 0) ? 1 : 0;
		for (int t = 0; t < sequence.length; t++) {
			sequence[t] = new MonoDoubleItemSet(instance.value(t + shift));
		}
		Sequence seq = new Sequence(sequence);

		double minD = Double.MAX_VALUE;
		int locatesplitpoint = -1;
		for (int i = 0; i < splitsequences.length; i++) {
			double tmpD = seq.distance(splitsequences[i]);
			if (tmpD < minD) {
				minD = tmpD;
				locatesplitpoint = i;
			}
		}
		classlable = (int) m_splitPoint.instance(locatesplitpoint).classValue();
//		classlable=m_splitPoint.instance(locatesplitpoint).classAttribute().indexOfValue(Double.toString(m_splitPoint.instance(locatesplitpoint).classValue()));
		return classlable;
	}

	/**
	 * Always returns null because there is only one subset.
	 */
	public final double[] weights(Instance instance) {

		return null;
	}

	/**
	 * Does nothing because no condition has to be satisfied.
	 */
	public final String leftSide(Instances instances) {

		return "";
	}

	/**
	 * Does nothing because no condition has to be satisfied.
	 */
	public final String rightSide(int index, Instances instances) {

		return "";
	}

	/**
	 * Returns a string containing java source code equivalent to the test made
	 * at this node. The instance being tested is called "i".
	 *
	 * @param index
	 *            index of the nominal value tested
	 * @param data
	 *            the data containing instance structure info
	 * @return a value of type 'String'
	 */
	public final String sourceExpression(int index, Instances data) {

		return "true"; // or should this be false??
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 1.9 $");
	}

	@Override
	public Instances getSplitPoint() {
		
		return m_splitPoint;
	}

	@Override
	public void setSplitPoint(Instances splitPoint) {
		this.m_splitPoint = splitPoint;
		
	}
	
}
