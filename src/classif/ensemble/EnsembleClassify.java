package classif.ensemble;

import java.util.Arrays;

import classif.BIGDT.ClassifyBigDT;
import classif.fastkmeans.DTWKNNClassifierKMeansCached;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class EnsembleClassify extends Classifier{
	private static final long serialVersionUID = 272706678590256204L;
	public EnsembleClassify() {
		super();
	}
	DTWKNNClassifierKMeansCached fkm = new DTWKNNClassifierKMeansCached();
	ClassifyBigDT dt = new ClassifyBigDT();
	public void buildClassifier(Instances data) throws Exception {
		fkm.setNbPrototypesPerClass(4);
		fkm.setFillPrototypes(true);
		fkm.buildClassifier(data);
		System.out.println("Fast K-Means build finished");
		dt.buildClassifier(data);
		System.out.println("Decision Tree build finished");
	}
	
	public double classifyInstance(Instance sample) throws Exception {
		int[] classlabel = new int[sample.numClasses()];
		classlabel[(int) fkm.classifyInstance(sample)] ++;
		classlabel[(int) dt.classifyInstance(sample)] ++;
//		System.out.println(Arrays.toString(classlabel));
		return Utils.maxIndex(classlabel);
	}

	public DTWKNNClassifierKMeansCached getFkm() {
		return fkm;
	}

	public ClassifyBigDT getDt() {
		return dt;
	}

}
