package classif.pu;

import java.util.ArrayList;
import java.util.Enumeration;
import items.ClassedSequence;
import items.MonoDoubleItemSet;
import items.Pairs;
import items.Sequence;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class Split extends ClassifierSplitModel{
	private Pairs m_pair;
	/** InfoGain of split. */
	private double m_infoGain;
	private double m_gainRatio; 
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
		
		ArrayList<ClassedSequence> prototypes = new ArrayList<>();
		for (int i = 0; i < 2; i++) {
			Sequence seq=m_pair.getPair()[i];
			Instance sample = new Instance(trainInstances.numAttributes());
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
//		int[][] nbObjPreBagPreClass_afterSplit = new int[trainInstances.numClasses()][trainInstances.numClasses()];
//		nbObjPreBagPreClass_afterSplit = classifyInstancesintoClass(trainInstances, prototypes);
		
		// computer infoGain
		m_infoGain = evalInfoGain(trainInstances, m_distribution.getperClassPerBag());
		m_gainRatio = splitCritValue(m_distribution, trainInstances.numInstances(), m_infoGain);
//		System.out.println("info\t" + m_infoGain);

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
	public double evalInfoGain(Instances instances,double[][] nbObjPreBagPreClass_afterSplit) {
		double parent_entropy = 0.0;
		double avg_child_entropy = 0.0;
		double[] child_entropy = new double[instances.numClasses()];
		double[] parent_weightPreClass = m_distribution.getperClass();
		double[] child_nbObjPreBag = m_distribution.getperBag();
		
		parent_entropy=OneClassEntropy(parent_weightPreClass[0],parent_weightPreClass[1]);
		
		for (int i = 0; i < nbObjPreBagPreClass_afterSplit.length; i++) {
			if (child_nbObjPreBag[i] == 0.0) {
				child_entropy[i] = 0.0;
				continue;
			}
			child_entropy[i] = OneClassEntropy(nbObjPreBagPreClass_afterSplit[i][0],
					nbObjPreBagPreClass_afterSplit[i][1]);
		}
		for (int i = 0; i < child_nbObjPreBag.length; i++) {
			avg_child_entropy += nbObjPreBagPreClass_afterSplit[i][1] / parent_weightPreClass[1] * child_entropy[i];
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
		for (int i = 0; i < errorclassifyObj.length; i++) {
			errorclassifyObj[i]=(int) m_distribution.numIncorrect(i);
		}
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
				double tmpD = seq.LB_distance(s.sequence,minD);
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
		return m_splitPoint;
	}

	public void setSplitPoint(Instances splitPoint) {
		this.m_splitPoint=splitPoint;
	}
	public final double infoGain() {

		return m_infoGain;
	}
	  public final double gainRatio() {
		    return m_gainRatio;
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
			double tmpD = seq.LB_distance(splitsequences[i], minD);
			if (tmpD < minD) {
				minD = tmpD;
				locatesplitpoint = i;
			}
		}
		classlable = (int) m_splitPoint.instance(locatesplitpoint).classValue();
//		classlable=m_splitPoint.instance(locatesplitpoint).classAttribute().indexOfValue(Double.toString(m_splitPoint.instance(locatesplitpoint).classValue()));
		return classlable;
	}
	
	public double OneClassEntropy(double dPosWeight, double dUnlWeight) {
		double p1 = (dPosWeight / ClassifyPOSC45.nPosSize) * ClassifyPOSC45.dDF * (ClassifyPOSC45.nUnlSize / dUnlWeight);
		if ((p1 > 1) || (Utils.eq(dUnlWeight, 0)))
			p1 = 1;
		double p0 = 1 - p1;
//		System.out.println("P1:"+p1+"\tP0:"+p0);
		double entropy = -log2(p0) - log2(p1);
		return entropy;
	}

	public final double splitCritValue(Distribution bags, double totalnoInst, double numerator) {

		double denumerator;
		double noUnknown;
		double unknownRate;
		int i;

		// Compute split info.
		denumerator = splitEnt(bags, totalnoInst);

		// Test if split is trivial.
		if (Utils.eq(denumerator, 0))
			return 0;
		denumerator = denumerator / totalnoInst;

		return numerator / denumerator;
	}
	  private final double splitEnt(Distribution bags,double totalnoInst){
		    
		    double returnValue = 0;
		    double noUnknown;
		    int i;
		    
		    noUnknown = totalnoInst-bags.total();
		    if (Utils.gr(bags.total(),0)){
		      for (i=0;i<bags.numBags();i++)
			returnValue = returnValue-log2(bags.perBag(i));
		      returnValue = returnValue-log2(noUnknown);
		      returnValue = returnValue+log2(totalnoInst);
		    }
		    return returnValue;
		  }
	  
	public final double classProb(int classIndex, Instance instance, int theSubset) throws Exception {

		if (theSubset <= -1) {
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
		} else {
			if (Utils.gr(m_distribution.perBag(theSubset), 0)) {
				return m_distribution.prob(classIndex, theSubset);
			} else {
				return m_distribution.prob(classIndex);
			}
		}
	}
}
