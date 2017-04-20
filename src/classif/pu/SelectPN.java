package classif.pu;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.Stack;

import org.apache.commons.math3.random.RandomDataGenerator;

import classif.ensemble.FindKNN;
import items.MonoDoubleItemSet;
import items.Pairs;
import items.Sequence;
import items.Sequences;
import weka.core.Instance;
import weka.core.Instances;

public class SelectPN {
	private Instances trainingData = null;
	private int total;

	public SelectPN() {
		super();
	}
	public Stack<Pairs> buildClassifier(Instances data) {
		trainingData = new Instances(data);
		
		/*
		 * split data to P U
		 */
		Instances p = new Instances(trainingData,0);
		Instances u = new Instances(trainingData, 0);
		for (int i = 0; i < trainingData.numInstances(); i++) {
			Instance sample = trainingData.instance(i);
			if(sample.classValue()==0.0)
				p.add(sample);
			else
				u.add(sample);
		}
		
		/*
		 * find N and KNN
		 */
		double dist = -1;
		int sign = -1;
		if (p.numInstances() == 1) {
			for (int i = 0; i < u.numInstances(); i++) {
				double d = convert(u.instance(i)).distance(convert(p.instance(0)));
				if (d > dist) {
					dist = d;
					sign = i;
				}
			}
		} else {
			for (int i = 0; i < u.numInstances(); i++) {
				double d = convert(u.instance(i))
						.distance(convert(p.instance(new RandomDataGenerator().nextInt(0, p.numInstances() - 1))));
				if (d > dist) {
					dist = d;
					sign = i;
				}
			}
		}
		
		Instance n=u.instance(sign);
		Instances knn_N = new Instances(trainingData, 0);
		total=Math.max(1,u.numInstances()/2);
		FindKNN findKNN = new FindKNN(n, u, total);
		knn_N = findKNN.KNN();
		
		/*
		 * use P and knn_N to combine pairs
		 */
		ArrayList<Sequence> class_P = new ArrayList<Sequence>();
		for (int i = 0; i < p.numInstances(); i++) {
			class_P.add(convert(p.instance(i)));
		}
		ArrayList<Sequence> class_N = new ArrayList<Sequence>();
		for (int i = 0; i < knn_N.numInstances(); i++) {
			class_N.add(convert(knn_N.instance(i)));
		}
		Stack<Pairs> stack = new Stack<Pairs>();
		Pairs pairs = new Pairs();
		/*for (int i = 0; i < class_P.size(); i++) {
			Sequence[] pair_Sequence = new Sequence[2];
			pair_Sequence[0] = class_P.get(i);
			for (int j = 0; j < class_N.size(); j++) {
				pair_Sequence[1] = class_N.get(j);
				pairs.setPair(pair_Sequence);
				pairs.setClasslable(new String[] { "1.0", "-1.0" });
				stack.push(pairs);
			}
		}*/
		
		while (stack.size()<100) {
			Sequence[] pair_Sequence = new Sequence[2];
			if (class_P.size() > 1)
				pair_Sequence[0]=class_P.get(new RandomDataGenerator().nextInt(0,class_P.size()-1));
			else
				pair_Sequence[0] = class_P.get(0);
			if (class_N.size() > 1)
				pair_Sequence[1]=class_N.get(new RandomDataGenerator().nextInt(0,class_N.size()-1));
			else
				pair_Sequence[1] = class_N.get(0);
			pairs.setPair(pair_Sequence);
			pairs.setClasslable(new String[] { "1.0", "-1.0" });
			stack.push(pairs);
		}
		return stack;
	}
		
		private Sequence convert(Instance instance) {
			Sequence sequence;
			MonoDoubleItemSet[] MonoDoubleItemSet = new MonoDoubleItemSet[instance.numAttributes() - 1];
			int shift = (instance.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < MonoDoubleItemSet.length; t++) {
				MonoDoubleItemSet[t] = new MonoDoubleItemSet(instance.value(t + shift));
			}
			sequence=new Sequence(MonoDoubleItemSet);
			return sequence;
			
		}
		
		private double distToInstances(Instance instance,Instances instances) {
			double dist=Double.MAX_VALUE;
			for (int i = 0; i < instances.numInstances(); i++) {
				double d= convert(instance).distance(convert(instances.instance(i)));
				if(d<dist) {
					dist=d;
				}
			}
			return dist;
		}
}
