package classif.DT;

import java.util.ArrayList;
import java.util.HashMap;

import items.ClassedSequence;
import items.MonoDoubleItemSet;
import items.Sequence;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class Node {
	protected Instances nodeInstance;
	protected int numSubsets;
	protected ArrayList<ClassedSequence> prototypes;
	private Instances allData;

	public Node(Instances data) {
		this.allData=data;
	}
	public Node selectModel(Instances data) throws Exception {
		HashMap<String, ArrayList<Sequence>> classedData;
		HashMap<String, ArrayList<Integer>> indexClassedDataInFullData;
		Sequence[] sequences;
		String[] classMap;
		Instances trainingData = null;
		if(data.numInstances()==1)
			return null;
		trainingData = data;
		Attribute classAttribute = data.classAttribute();

		classedData = new HashMap<String, ArrayList<Sequence>>();
		indexClassedDataInFullData = new HashMap<String, ArrayList<Integer>>();
		for (int c = 0; c < data.numClasses(); c++) {
			classedData.put(data.classAttribute().value(c), new ArrayList<Sequence>());
			indexClassedDataInFullData.put(data.classAttribute().value(c), new ArrayList<Integer>());
		}

		sequences = new Sequence[data.numInstances()];
		classMap = new String[sequences.length];
		for (int i = 0; i < sequences.length; i++) {
			Instance sample = data.instance(i);
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
			int shift = (sample.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < sequence.length; t++) {
				sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
			}
			sequences[i] = new Sequence(sequence);
			String clas = sample.stringValue(classAttribute);
			classMap[i] = clas;
			classedData.get(clas).add(sequences[i]);
			indexClassedDataInFullData.get(clas).add(i);
		}
		String[] classes = new String[trainingData.numClasses()];
		for (int i = 0; i < trainingData.numClasses(); i++) {
			classes[i] = trainingData.classAttribute().value(i);
		}

		int classtrue = 0;
		int classfalse = 0;
		double purityall = 0.0;
		int[] locate = new int[2];
		double[][] purity = new double[classedData.get(classes[0]).size()][classedData.get(classes[1]).size()];
		for (int index_c0 = 0; index_c0 < classedData.get(classes[0]).size(); index_c0++) {
			for (int index_c1 = 0; index_c1 < classedData.get(classes[1]).size(); index_c1++) {
				Sequence seqs_0 = classedData.get(classes[0]).get(index_c0);
				Sequence seqs_1 = classedData.get(classes[0]).get(index_c1);
				ClassedSequence c0 = new ClassedSequence(seqs_0, classes[0]);
				ClassedSequence c1 = new ClassedSequence(seqs_1, classes[1]);
				ArrayList<ClassedSequence>prototype = new ArrayList<>();
				prototype.add(c0);
				prototype.add(c1);
				for (int i = 0; i < trainingData.numInstances(); i++) {
					Instance sample = trainingData.instance(i);
					double lable = 0;
					lable = classifyInstance(sample,prototype);
					if (lable == sample.classValue()) {
						classtrue++;
					} else
						classfalse++;
				}
				purity[index_c0][index_c1] = classtrue / (classtrue + classfalse);
				if (purity[index_c0][index_c1] > purityall) {
					purityall = purity[index_c0][index_c1];
					locate[0] = index_c0;
					locate[1] = index_c1;
				}
			}
		}
		Instances node_Ins = null;
		for (int i = 0; i < trainingData.numClasses(); i++) {
			Instance inst = null;
			inst.setClassValue(classes[i]);
			for (int j = 1; j < data.instance(0).numAttributes(); j++) {
				inst.setValue(j, classedData.get(classes[i]).get(locate[i]).getItem(j).toString());
				ClassedSequence cq = new ClassedSequence(classedData.get(classes[i]).get(locate[i]), classes[i]);
				prototypes.add(cq);
			}
			node_Ins.add(inst);
		}
		Node node=new Node(data);
		node.setNodeInstance(node_Ins);
		node.setNumSubsets(classfalse);
		return node;
	}

	
	public Instances[] split(Instances data) throws Exception {
		Instances[] splits = new Instances[2];
		for (int i = 0; i < data.numInstances(); i++) {
			double lable = classifyInstance(data.instance(i), prototypes);
			if (lable == allData.classAttribute().indexOfValue(prototypes.get(0).classValue))
				splits[0].add(data.instance(i));
			else
				splits[1].add(data.instance(i));
		}
		return splits;
	}

	public Instances getNodeInstance() {
		return nodeInstance;
	}
	public void setNodeInstance(Instances nodeInstance) {
		this.nodeInstance = nodeInstance;
	}
	public int getNumSubsets() {
		return numSubsets;
	}
	public void setNumSubsets(int numSubsets) {
		this.numSubsets = numSubsets;
	}
	public double classifyInstance(Instance sample,ArrayList<ClassedSequence>prototype) throws Exception {
		// transform instance to sequence
		MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
		int shift = (sample.classIndex() == 0) ? 1 : 0;
		for (int t = 0; t < sequence.length; t++) {
			sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
		}
		Sequence seq = new Sequence(sequence);

		double minD = Double.MAX_VALUE;
		String classValue = null;
		for (ClassedSequence s : prototype) {
			double tmpD = seq.distance(s.sequence);
			if (tmpD < minD) {
				minD = tmpD;
				classValue = s.classValue;
			}
		}
		return sample.classAttribute().indexOfValue(classValue);
	}

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
	  public double [] weights(Instance instance) {
		return null;
	}
	  public int whichSubset(Instance instance) throws Exception {
		return 0;
	}
	
	
	
}
