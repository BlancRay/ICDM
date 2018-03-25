package nwafu.dm.tsc.classif.pu;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Stack;

import nwafu.dm.tsc.items.ClassedSequence;
import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.Pairs;
import nwafu.dm.tsc.items.Sequence;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class Dataselect {
	protected ArrayList<ClassedSequence> prototypes;
	protected HashMap<String, ArrayList<Sequence>> classedData;
	HashMap<String, ArrayList<Integer>> indexClassedDataInFullData;
	protected int nClustersPerClass;
	Sequence[] sequences;
	String[] classMap;
	Instances trainingData = null;
	int nbPairs;
	public Dataselect() {
		super();
	}
	int sample=50;
	public Stack<Pairs> buildClassifier(Instances data) {
//		if (data.numInstances() > sample) {
//			trainingData=new Instances(data);
//			RandomDataGenerator randGen = new RandomDataGenerator();
//			int[] selected = randGen.nextPermutation(data.numInstances(), data.numInstances() / sample);
//			for (int i = 0; i < selected.length; i++) {
//				trainingData.add(data.instance(selected[i]));
//			}
//		} else
			trainingData = data;
		// nbPairs=(int) Math.pow(data.numInstances()/10, 2);
		nbPairs = 100;
		
		Attribute classAttribute = trainingData.classAttribute();
		prototypes = new ArrayList<nwafu.dm.tsc.items.ClassedSequence>();

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
		int totle=0;
		for (int i = 0; i < classAttribute.numValues()-1; i++) {
			for (int j = i+1; j < classAttribute.numValues(); j++) {
				totle+=classedData.get(classAttribute.value(i)).size()*classedData.get(classAttribute.value(j)).size();
			}
		}
		nbPairs=Math.max(totle/sample, 100);
		
		Stack<Pairs> stack=new Stack<Pairs>();
		Stack<Pairs> stack_sort=new Stack<Pairs>();
		for (int i = 0; i < classAttribute.numValues()-1; i++) {
			for (int j = i+1; j < classAttribute.numValues(); j++) {
				ArrayList<Sequence> class1 = classedData.get(classAttribute.value(i));
				ArrayList<Sequence> class2 = classedData.get(classAttribute.value(j));
				for (int n = 0; n < class1.size(); n++) {
					for (int m = 0; m < class2.size(); m++) {
						Pairs pairs=new Pairs();
						Sequence[] pair_Sequence = new Sequence[2];
						pair_Sequence[0]=class1.get(n);
						pair_Sequence[1]=class2.get(m);
						pairs.setPair(pair_Sequence);
						pairs.setClasslable(new String[]{classAttribute.value(i),classAttribute.value(j)});
						if(stack.size()<nbPairs){
							pairs.setDistance(pairs.Distance());
							if(pairs.getDistance()==0.0)
								continue;
							while (!stack.isEmpty() && stack.peek().getDistance() > pairs.getDistance()) {
								stack_sort.push(stack.pop());
							}
							stack.push(pairs);
							while (!stack_sort.isEmpty()) {
								stack.push(stack_sort.pop());
							}
						}
						else{
							double d=pair_Sequence[0].LB_distance(pair_Sequence[1], stack.peek().getDistance());
							if(d<stack.peek().getDistance()&&d!=0.0){
								pairs.setDistance(pairs.Distance());
								stack.pop();
								while (!stack.isEmpty() && stack.peek().getDistance() >d) {
									stack_sort.push(stack.pop());
								}
								stack.push(pairs);
								while (!stack_sort.isEmpty()) {
									stack.push(stack_sort.pop());
								}
							}
						}
					}
				}
			}
		}
//		Stack<Pairs>stack_copy=(Stack<Pairs>) stack.clone();
//		while (!stack_copy.isEmpty()) {
//			Pairs pair=stack_copy.pop();
//			System.out.println("Dist: "+pair.getDistance());
//			System.out.println("Sequence: ");
//			for (int i = 0; i < pair.getPair().length; i++) {
//				System.out.println(pair.getPair()[i]);
//				
//			}
//			System.out.println("Lable: "+Arrays.toString(pair.getClasslable()));
//			System.out.println();
//		}
		return stack;
		
		
//		Computer infogain for each pairs
/*
		Instances data;
		classedData:C1 C2 C3 C4
		for C1 to C3
			for C1+1 to C4
				for Sequence S1 in C1
					for Sequence S1 in C1+1
						if  nb of Pairs !> 100
							pair[S1,S2].dist=DTW(S1,S2)
							push pair in Pairs
						else
							pair[S1,S2].dist=LB_DTW(S1,S2,Pairs.longestdist)
							if  pair[S1,S2].dist < Pairs.longestdist
								pop pair with Pairs.longestdist
								push pair[S1,S2]
								recomputer Pairs.longestdist
		*/
		
		
	}
}
