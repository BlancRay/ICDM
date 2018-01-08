package classif.pu;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import java.util.ArrayList;
import java.util.Enumeration;
import items.ClassedSequence;
import items.MonoDoubleItemSet;
import items.Sequence;

public class C45Split extends ClassifierSplitModel {

	/** Attribute to split on. */
	private int[] m_pairIndex;

	/** Value of split point. */
	public Instances m_splitPoint;

	/** InfoGain of split. */
	private double m_infoGain;

	/** GainRatio of split. */
	private double m_gainRatio;

	/** Static reference to splitting criterion. */
	private static double log2 = Math.log(2);
	/** Desired number of branches. */
	private int m_complexityIndex;

	/**
	 * Initializes the split model.
	 */
	public C45Split(int[] pairIndex) {

		// Get index of attribute to split on.
		m_pairIndex = pairIndex;
	}

	/**
	 * Creates a C4.5-type split on the given data. Assumes that none of the
	 * class values is missing.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	public void buildClassifier(Instances trainInstances) throws Exception {

		// Initialize the remaining instance variables.
		m_numSubsets = 0;
		m_splitPoint = new Instances(trainInstances, m_pairIndex.length);
		m_infoGain = 0;
		m_gainRatio = 0;
		m_complexityIndex = m_pairIndex.length;

		m_distribution = new Distribution(m_complexityIndex, trainInstances.numClasses());
		Instance instance;
		// Only Instances with known values are relevant.
		Enumeration<Instance> enu = trainInstances.enumerateInstances();
		while (enu.hasMoreElements()) {
			instance = (Instance) enu.nextElement();
			m_distribution.add(whichSubset(instance), instance);
		}

		ArrayList<ClassedSequence> prototypes = new ArrayList<>();
		for (int i = 0; i < m_pairIndex.length; i++) {
			Instance sample = trainInstances.instance(m_pairIndex[i]);
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
			int shift = (sample.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < sequence.length; t++) {
				sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
			}
			ClassedSequence s = new ClassedSequence(new Sequence(sequence),
					sample.classAttribute().value((int) sample.classValue()));
			prototypes.add(s);
			m_splitPoint.add(sample);
		}

		// computer error rate
		int[][] nbObjPreClass_afterSplit = new int[trainInstances.numClasses()][trainInstances.numClasses()];
		nbObjPreClass_afterSplit = classifyInstancesintoClass(trainInstances, prototypes);

		int[] error = new int[trainInstances.numClasses()];
		error = evalerror(trainInstances, prototypes);
		// computer infoGain
		m_infoGain = evalInfoGain(trainInstances, nbObjPreClass_afterSplit);
		// m_gainRatio = evalGainRatio(nbObjPreClass_afterSplit, m_infoGain);

		// System.out.println("info\t"+m_infoGain);
		// System.out.println("gain\t"+m_gainRatio);

		for (int i = 0; i < error.length; i++) {
			if (error[i] != 0)
				m_numSubsets = 2;
		}
	}

	/*
	 * public double evalGainRatio(int[] nbObjeachClass, double infoGain) {
	 * double gainRatio = 0.0; double sum = 0.0; sum=Utils.sum(nbObjeachClass);
	 * for (int i = 0; i < nbObjeachClass.length; i++) { gainRatio -= log2(1.0 *
	 * nbObjeachClass[i] / sum); } return (infoGain/gainRatio); }
	 */

	public double evalInfoGain(Instances instances, int[][] nbObj_aftersplit_eachClass) {
		double parent_entropy = 0.0;
		double avg_child_entropy = 0.0;
		double[] child_entropy = new double[instances.numClasses()];
		double[] parent_nbObjPreClass = new double[instances.numClasses()];
		double[] child_nbObjPreClass = new double[instances.numClasses()];
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance Obj = instances.instance(i);
			parent_nbObjPreClass[(int) Obj.classValue()]++;
		}
		for (int i = 0; i < parent_nbObjPreClass.length; i++) {
			parent_entropy -= log2(parent_nbObjPreClass[i] / instances.numInstances());
		}

		for (int i = 0; i < nbObj_aftersplit_eachClass.length; i++) {
			// sum Objs in each branch
			child_nbObjPreClass[i] = Utils.sum(nbObj_aftersplit_eachClass[i]);
			for (int j = 0; j < nbObj_aftersplit_eachClass[i].length; j++) {
				// entropy for each child
				child_entropy[i] -= (log2(nbObj_aftersplit_eachClass[i][j] / child_nbObjPreClass[i]));
			}
		}
		for (int i = 0; i < child_nbObjPreClass.length; i++) {
			avg_child_entropy += child_nbObjPreClass[i] / instances.numInstances() * child_entropy[i];
		}
		return parent_entropy - avg_child_entropy;
	}

	public int[][] classifyInstancesintoClass(Instances data, ArrayList<ClassedSequence> prototypes) {
		int[][] nbObjeachclass = new int[data.numClasses()][data.numClasses()];
		for (int i = 0; i < data.numInstances(); i++) {
			Instance sample = data.instance(i);
			// transform instance to sequence
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
			int shift = (sample.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < sequence.length; t++) {
				sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
			}
			Sequence seq = new Sequence(sequence);
			double minD = Double.MAX_VALUE;
			String classValue = null;
			for (ClassedSequence s : prototypes) {
				double tmpD = seq.distance(s.sequence);
				if (tmpD < minD) {
					minD = tmpD;
					classValue = s.classValue;
				}
			}
			nbObjeachclass[sample.classAttribute().indexOfValue(classValue)][(int) sample.classValue()]++;
		}
		return nbObjeachclass;
	}

	public int[] evalerror(Instances data, ArrayList<ClassedSequence> prototypes) {

		int[] errorclassifyObj = new int[data.numClasses()];
		for (int i = 0; i < data.numInstances(); i++) {
			Instance sample = data.instance(i);
			// transform instance to sequence
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
			int shift = (sample.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < sequence.length; t++) {
				sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
			}
			Sequence seq = new Sequence(sequence);
			double minD = Double.MAX_VALUE;
			String classValue = null;
			for (ClassedSequence s : prototypes) {
				double tmpD = seq.distance(s.sequence);
				if (tmpD < minD) {
					minD = tmpD;
					classValue = s.classValue;
				}
			}
			if (sample.classAttribute().indexOfValue(classValue) != (int) sample.classValue())
				errorclassifyObj[sample.classAttribute().indexOfValue(classValue)]++;
		}
		return errorclassifyObj;

	}

	/**
	 * Returns (C4.5-type) gain ratio for the generated split.
	 */
	public final double gainRatio() {
		return m_gainRatio;
	}

	/**
	 * Returns (C4.5-type) information gain for the generated split.
	 */
	public final double infoGain() {

		return m_infoGain;
	}

	/**
	 * Sets split point to greatest value in given data smaller or equal to old
	 * split point. (C4.5 does this for some strange reason).
	 * 
	 * @return
	 */
	public final Instances setSplitPoint() {

		return m_splitPoint;
	}

	/**
	 * Returns index of subset instance is assigned to. Returns -1 if instance
	 * is assigned to more than one subset.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	public final int whichSubset(Instance sample) throws Exception {
		int classlable = -1;
		Sequence[] splitsequences = new Sequence[m_splitPoint.numInstances()];
		for (int i = 0; i < splitsequences.length; i++) {
			Instance splitInstance = m_splitPoint.instance(i);
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[splitInstance.numAttributes() - 1];
			int shift = (splitInstance.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < sequence.length; t++) {
				sequence[t] = new MonoDoubleItemSet(splitInstance.value(t + shift));
			}
			splitsequences[i] = new Sequence(sequence);
		}

		MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
		int shift = (sample.classIndex() == 0) ? 1 : 0;
		for (int t = 0; t < sequence.length; t++) {
			sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
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
		// classlable=m_splitPoint.instance(locatesplitpoint).classAttribute().indexOfValue(Double.toString(m_splitPoint.instance(locatesplitpoint).classValue()));
		return classlable;
	}

	public double log2(double a) {
		if (a == 0.0)
			return 0.0;
		else
			return a * Math.log(a) / log2;
	}

	@Override
	public Instances getSplitPoint() {
		// TODO Auto-generated method stub
		return m_splitPoint;
	}

	@Override
	public void setSplitPoint(Instances splitPoint) {
		this.m_splitPoint = splitPoint;
	}

	@Override
	public double[] weights(Instance instance) {
		// TODO Auto-generated method stub
		return null;
	}
}
