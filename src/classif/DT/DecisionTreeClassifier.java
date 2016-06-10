package classif.DT;

import java.util.ArrayList;
import java.util.HashMap;

import items.ClassedSequence;
import items.Itemset;
import items.MonoDoubleItemSet;
import items.Sequence;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class DecisionTreeClassifier extends Classifier {
	private static final long serialVersionUID = 922540906465712982L;

	protected ArrayList<ClassedSequence> prototypes;
	protected HashMap<String, ArrayList<Sequence>> classedData;
	protected HashMap<String, ArrayList<Integer>> indexClassedDataInFullData;
	protected int nbPrototypesPerClass[];
	protected Sequence[] sequences;
	protected String[] classMap;
	protected Instances trainingData = null;
	protected boolean fillPrototypes = true;
	private Tree root;
	private int m_minNumObj = 2;

	public DecisionTreeClassifier() {
		super();
	}

	public void buildClassifier(Instances data) throws Exception {
		Node modSelection;
		modSelection = new Node(data);
		root = new Tree(modSelection);
		root.buildClassifier(data);
	}

	public double classifyInstance(Instance instance) throws Exception {

		return root.classifyInstance(instance);
	}
}
