package classif.DT;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.ArrayList;
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
	public static double log2 = Math.log(2);

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
		int[] nbObjPreClass = new int[trainInstances.numClasses()];
		nbObjPreClass = classifyInstancesintoClass(trainInstances, prototypes);
		
		
		
		
		int[] error = new int[trainInstances.numClasses()];
		error = evalerror(trainInstances, prototypes);
		// computer infoGain
		m_infoGain = evalInfoGain(trainInstances,nbObjPreClass);
		m_gainRatio = evalGainRatio(nbObjPreClass, m_infoGain);

		// System.out.println("info\t"+m_infoGain);
		// System.out.println("gain\t"+m_gainRatio);

		for (int i = 0; i < error.length; i++) {
			if (error[i] != 0)
				m_numSubsets=2;
		}
	}

	public double evalGainRatio(int[] nbObjeachClass, double infoGain) {
		double gainRatio = 0.0;
		double sum = 0.0;
		sum=Utils.sum(nbObjeachClass);
		for (int i = 0; i < nbObjeachClass.length; i++) {
			gainRatio -= log2(1.0 * nbObjeachClass[i] / sum);
		}
		return (infoGain/gainRatio);
	}

	public double evalInfoGain(Instances instances,int[] nbObjeachClass) {
		double Gain = 0.0;
		double sum = 0.0;
		double infoGain = 0.0;
		double[] nbObjPreClass = new double[instances.numClasses()];
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance Obj = instances.instance(i);
			nbObjPreClass[(int) Obj.classValue()]++;
		}
		for (int i = 0; i < nbObjPreClass.length; i++) {
			infoGain -= log2(nbObjPreClass[i] / instances.numInstances());
		}
		sum=Utils.sum(nbObjeachClass);
		for (int i = 0; i < nbObjeachClass.length; i++) {
			Gain -= (1.0 * nbObjeachClass[i] / sum) * (log2(1.0 * nbObjeachClass[i] / sum));
		}
		return infoGain-Gain;
	}

	public int[] classifyInstancesintoClass(Instances data, ArrayList<ClassedSequence> prototypes) {
		int[] nbObjeachclass = new int[data.numClasses()];
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
			nbObjeachclass[sample.classAttribute().indexOfValue(classValue)]++;
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
//		classlable=m_splitPoint.instance(locatesplitpoint).classAttribute().indexOfValue(Double.toString(m_splitPoint.instance(locatesplitpoint).classValue()));
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
		this.m_splitPoint=splitPoint;
	}
}
