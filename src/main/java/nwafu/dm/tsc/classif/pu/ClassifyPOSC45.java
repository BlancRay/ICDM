package nwafu.dm.tsc.classif.pu;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class ClassifyPOSC45 extends AbstractClassifier{

	/** for serialization */
	static final long serialVersionUID = -217733168393644444L;

	/** The decision tree */
//	private ClassifierTree[] root=new ClassifierTree[10];
	private ClassifierTree root;
	public static int nPosSize;
	public static int nUnlSize;
	public static double dDF = 0.5;
	private Instances posDataset;
	private Instances unlDataset;
	public ClassifyPOSC45(double d)
	{
		dDF = d;
	}

	public ClassifyPOSC45()
	{
		dDF = 0.5; // default value, this is OK when POS:NEG is expected to be
					// 1:1
	}
	public void setDataset(Instances posData, Instances unlData)
	{
		posDataset = new Instances(posData);
		unlDataset = new Instances(unlData);
	}

	public Instances createTrainingData() {
		nPosSize = posDataset.numInstances();
		nUnlSize = unlDataset.numInstances();
		Instances dataset = new Instances(posDataset);
		for (int i = 0; i < nUnlSize; i++) {
			dataset.add(unlDataset.instance(i));
		}
		return dataset;
	}
	/**
	 * Generates the classifier.
	 *
	 * @param instances
	 *            the data to train the classifier with
	 * @throws Exception
	 *             if classifier can't be built successfully
	 */
	public void buildClassifier(Instances instances) throws Exception {

		Instances dataset = createTrainingData();
		ModelSelection modSelection;
		modSelection = new C45ModelSelection(dataset);
		root =new C45tree(modSelection);
		root.buildClassifier(dataset);
//		for (int i = 0; i < root.length; i++) {
//			root[i] = new C45tree(modSelection);
//			root[i].buildClassifier(dataset);
//			System.out.println(i+"th tree has been built.");
//		}
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
//		int[] classlabel=new int[instance.numClasses()];
//		for (int j = 0; j < root.length; j++) {
//			classlabel[(int) root[j].classifyInstance(instance)]++;
//		}
////		System.out.println("classlabel:"+Arrays.toString(classlabel));
//		return Utils.maxIndex(classlabel);
		return root.classifyInstance(instance);
	}
}
