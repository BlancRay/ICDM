package classif.BIGDT;

import java.util.Arrays;
import java.util.Collections;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class ClassifyBigDT extends Classifier{

	/** for serialization */
	static final long serialVersionUID = -217733168393644444L;

	/** The decision tree */
	private ClassifierTree[] root=new ClassifierTree[10];

	/**
	 * Generates the classifier.
	 *
	 * @param instances
	 *            the data to train the classifier with
	 * @throws Exception
	 *             if classifier can't be built successfully
	 */
	public void buildClassifier(Instances instances) throws Exception {

		ModelSelection modSelection;
		modSelection = new C45ModelSelection(instances);
		for (int i = 0; i < root.length; i++) {
			root[i] = new C45tree(modSelection);
			root[i].buildClassifier(instances);
			System.out.println(i+"th tree has been built.");
		}
	}

	/**
	 * Classifies an instance.
	 *
	 * @param instance
	 *            the instance to classify
	 * @return the classification for the instance
	 * @throws Exception
	 *             if instance can't be classified successfully
	 */
	public double classifyInstance(Instance instance) throws Exception {
		int[] classlabel=new int[instance.numClasses()];
		for (int j = 0; j < root.length; j++) {
			classlabel[(int) root[j].classifyInstance(instance)]++;
		}
//		System.out.println("classlabel:"+Arrays.toString(classlabel));
		return Utils.maxIndex(classlabel);
	}
}
