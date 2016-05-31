package classif.DT;

import java.util.ArrayList;
import java.util.HashMap;

import items.ClassedSequence;
import items.MonoDoubleItemSet;
import items.Sequence;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class DecisionTreeClassifier extends Classifier{
	private static final long serialVersionUID = 922540906465712982L;

	protected ArrayList<ClassedSequence> prototypes;
	protected HashMap<String, ArrayList<Sequence>> classedData;
	protected HashMap<String, ArrayList<Integer>> indexClassedDataInFullData;
	protected int nbPrototypesPerClass[];
	protected Sequence[] sequences;
	protected String[] classMap;
	protected Instances trainingData = null;
	protected boolean fillPrototypes = true;

	public DecisionTreeClassifier() {
		super();
	}
	public void buildClassifier(Instances data) throws Exception {
		trainingData = data;
		Attribute classAttribute = data.classAttribute();
		prototypes = new ArrayList<>();

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
//			System.out.println("Element "+i+" of train is classed "+clas+" and went to element "+(indexClassedDataInFullData.get(clas).size()-1));
		}
		
		
		foreach class1{
			foreach class2{
				set obj 1,2 as prototype;
				classifyInstance(instance);
				computer purity;
				if (purity>prepurity){
					store protptype;
					purityall=purity;
					prepurity=purity;
				}
			}
		}
		
		//TODO for each prototype,add Objs to each affectation 
		classifyInstance(instance);
		
		store 
	}

	public double classifyInstance(Instance sample) throws Exception {
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
		// System.out.println(prototypes.size());
		return sample.classAttribute().indexOfValue(classValue);
	}
	
	protected double purity(ArrayList<ClassedSequence>[] affectation) {
		double purity = 0.0;
		double tureaffectation = 0.0;
		for (int i = 0; i < affectation.length; i++) {
			int[] labels = new int[trainingData.classAttribute().numValues()];
			for (ClassedSequence s : affectation[i]) {
				String classValue = s.classValue;
				labels[(int) Double.parseDouble(classValue)]++;
			}
			
			int majority = 0;
			int max = 0;
			for (int j = 0; j < labels.length; j++) {
				if (labels[j] > max)
					majority = j;
			}
			tureaffectation += labels[majority];
		}
		purity = tureaffectation / trainingData.numInstances();
		return purity;
	}
}
