package nwafu.dm.tsc.classif.ensemble;

import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.Sequence;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class FindKNN {
	private Instance query;
	private Instances train;
	private int k;
	public FindKNN(Instance query,Instances train,int k) {
		this.query=query;
		this.train=train;
		this.k=k;
	}

	public Instances KNN() {
		double[] dist=new double[train.numInstances()];
		Instances knn=new Instances(train, k);
		for (int i = 0; i < train.numInstances(); i++) {
			Instance trainInstance=train.instance(i);
			dist[i]=convert(query).distance(convert(trainInstance));
		}
		int[] sorted=Utils.sort(dist);
		for (int i = 0; i < k; i++) {
			knn.add(train.instance(sorted[i]));
		}
//		knn.add(query);
		return knn;
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
}
