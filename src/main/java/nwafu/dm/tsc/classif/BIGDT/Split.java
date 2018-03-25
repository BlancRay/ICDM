package nwafu.dm.tsc.classif.BIGDT;

import java.util.ArrayList;
import java.util.Enumeration;

import nwafu.dm.tsc.items.ClassedSequence;
import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.Pairs;
import nwafu.dm.tsc.items.Sequence;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class Split extends ClassifierSplitModel{
	private static final long serialVersionUID = -2324062930716795258L;
	private Pairs m_pair;
	/** InfoGain of split. */
	private double m_infoGain;
	/** Desired number of branches. */
	private int m_complexityIndex;
	
	/** Static reference to splitting criterion. */
	private static double log2 = Math.log(2);
	public Split(Pairs pair) {

		// Get index of attribute to split on.
		m_pair = pair;
	}
	
	
	public void buildClassifier(Instances trainInstances) throws Exception {

		// Initialize the remaining instance variables.
		m_numSubsets = 0;
		m_splitPoint = new Instances(trainInstances, 2);
		m_infoGain = 0;
		m_complexityIndex = trainInstances.numClasses();
		
		ArrayList<ClassedSequence> prototypes = new ArrayList<ClassedSequence>();
		for (int i = 0; i < 2; i++) {
			Sequence seq=m_pair.getPair()[i];
			Instance sample = new DenseInstance(trainInstances.numAttributes());
			sample.setDataset(trainInstances);
			sample.setClassValue(m_pair.getClasslable()[i]);
			for (int j = 1; j < sample.numAttributes(); j++) {
				sample.setValue(sample.attribute(j),seq.getItem(j-1).getValue());
			}
			ClassedSequence s = new ClassedSequence(seq,sample.classAttribute().value((int) sample.classValue()));
			prototypes.add(s);
			m_splitPoint.add(sample);
		}

		m_distribution = new Distribution(m_complexityIndex, trainInstances.numClasses());
		Instance instance;
		// Only Instances with known values are relevant.
		Enumeration enu = trainInstances.enumerateInstances();
		while (enu.hasMoreElements()) {
			instance = (Instance) enu.nextElement();
			m_distribution.add(whichSubset(instance), instance);
		}
		
		

		// computer error rate
//		int[][] nbObjPreClass_afterSplit = new int[trainInstances.numClasses()][trainInstances.numClasses()];
//		nbObjPreClass_afterSplit = classifyInstancesintoClass(trainInstances, prototypes);
		
		// computer infoGain
		m_infoGain = evalInfoGain(trainInstances);

		// System.out.println("info\t"+m_infoGain);

		int[] error = new int[trainInstances.numClasses()];
		error = evalerror(trainInstances, prototypes);
		for (int i = 0; i < error.length; i++) {
			if (error[i] != 0) {
				m_numSubsets = trainInstances.numClasses();
				break;
			}
		}
//		setSplitPoint(m_splitPoint);
	}
	public double evalInfoGain(Instances instances) {
		double parent_entropy = 0.0;
		double avg_child_entropy = 0.0;
		double[] child_entropy = new double[instances.numClasses()];
		double[] parent_nbObjPreClass = m_distribution.getM_perClass();
		double[] child_nbObjPreClass = m_distribution.getM_perBag();
		for (int i = 0; i < parent_nbObjPreClass.length; i++) {
			parent_entropy -= log2(parent_nbObjPreClass[i] / instances.numInstances());
		}

		for (int i = 0; i < m_distribution.matrix().length; i++) {
			if(child_nbObjPreClass[i]==0.0){
				child_entropy[i]= 0.0;
				continue;
			}
			for (int j = 0; j < m_distribution.matrix()[i].length; j++) {
				child_entropy[i] -= (log2(m_distribution.matrix()[i][j] / child_nbObjPreClass[i]));//entropy for each child
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
	public double log2(double a) {
		if (a == 0.0)
			return 0.0;
		else
			return a * Math.log(a) / log2;
	}

	public Instances getSplitPoint() {
		// TODO Auto-generated method stub
		return m_splitPoint;
	}

	public void setSplitPoint(Instances splitPoint) {
		this.m_splitPoint=splitPoint;
	}
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
	
	public final int whichbag(Instance sample) throws Exception {
		int baglable = -1;
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
		baglable = locatesplitpoint;
		return baglable;
	}
}
