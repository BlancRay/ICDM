package nwafu.dm.tsc.classif.DT;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class ClassifyDT extends AbstractClassifier{
	private static final long serialVersionUID = 8295336430552350575L;
	/** The decision tree */
	private ClassifierTree root;

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
		modSelection = new C45ModelSelection( instances);
		root = new C45tree(modSelection);
		root.buildClassifier(instances);
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
//		System.out.println("");
		return root.classifyInstance(instance);
	}
}
