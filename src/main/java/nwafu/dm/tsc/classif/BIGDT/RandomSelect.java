package nwafu.dm.tsc.classif.BIGDT;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Stack;

import org.apache.commons.math3.analysis.function.Max;
import org.apache.commons.math3.random.RandomDataGenerator;

import nwafu.dm.tsc.items.ClassedSequence;
import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.Pairs;
import nwafu.dm.tsc.items.Sequence;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class RandomSelect {

	protected HashMap<String, ArrayList<Sequence>> classedData;
	protected HashMap<String, ArrayList<Integer>> indexClassedDataInFullData;
	protected Sequence[] sequences;
	protected String[] classMap;
	protected Instances trainingData = null;
	protected int nbPairs;

	public RandomSelect() {
		super();
	}

	protected double sample = 0.05;

	public Stack<Pairs> buildClassifier(Instances data) {
		trainingData = data;
		Attribute classAttribute = trainingData.classAttribute();

		classedData = new HashMap<String, ArrayList<Sequence>>();
		indexClassedDataInFullData = new HashMap<String, ArrayList<Integer>>();
		for (int c = 0; c < trainingData.numClasses(); c++) {
			classedData.put(trainingData.classAttribute().value(c), new ArrayList<Sequence>());
			indexClassedDataInFullData.put(trainingData.classAttribute().value(c), new ArrayList<Integer>());
		}

		sequences = new Sequence[trainingData.numInstances()];
		classMap = new String[sequences.length];
		for (int i = 0; i < sequences.length; i++) {
			Instance sample = trainingData.instance(i);
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
		ArrayList<String> values = new ArrayList<String>(classAttribute.numValues());
		for (String clas : classedData.keySet()) {
			if (classedData.get(clas).size() != 0)
				values.add(clas);
		}
		Attribute newclassAttribute=new Attribute("class",values);
		int totle = 0;
		for (int i = 0; i < newclassAttribute.numValues() - 1; i++) {
			for (int j = i + 1; j < newclassAttribute.numValues(); j++) {
				totle += classedData.get(newclassAttribute.value(i)).size()
						* classedData.get(newclassAttribute.value(j)).size();
			}
		}
//		nbPairs = Math.min((int)Math.ceil(totle /10)+1, 100);
//		nbPairs = totle>0?Math.min(Math.min(totle, 5),100):0;
//		nbPairs = (int) Math.min(totle * 0.5, 100);
		nbPairs = (totle > 2 ? ((int) Math.ceil(totle * sample) == 1 ? 2 : Math.min((int) (totle * sample), 100)) : totle);
//		System.out.println(totle * sample+"\t"+nbPairs);
		Stack<Pairs> stack = new Stack<Pairs>();
		while (stack.size() < nbPairs) {
			RandomDataGenerator randGen = new RandomDataGenerator();
			int[] classselected = randGen.nextPermutation(newclassAttribute.numValues(), 2);
			ArrayList<Sequence> class1 = classedData.get(newclassAttribute.value(classselected[0]));
			ArrayList<Sequence> class2 = classedData.get(newclassAttribute.value(classselected[1]));
			Pairs pairs = new Pairs();
			Sequence[] pair_Sequence = new Sequence[2];
			if (class1.size() > 1)
				pair_Sequence[0] = class1.get(randGen.nextInt(0, class1.size() - 1));
			else
				pair_Sequence[0] = class1.get(0);
			if (class2.size() > 1)
				pair_Sequence[1] = class2.get(randGen.nextInt(0, class2.size() - 1));
			else
				pair_Sequence[1] = class2.get(0);
			pairs.setPair(pair_Sequence);
			pairs.setClasslable(
					new String[] { newclassAttribute.value(classselected[0]), newclassAttribute.value(classselected[1]) });
//			pairs.setDistance(pairs.Distance());
//			if (pairs.getDistance() == 0.0)
//				continue;
			stack.push(pairs);
		}
		return stack;
	}
	
}
