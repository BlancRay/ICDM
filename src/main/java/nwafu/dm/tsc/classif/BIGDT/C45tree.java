package nwafu.dm.tsc.classif.BIGDT;

import java.util.Stack;
import nwafu.dm.tsc.items.Pairs;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class C45tree extends ClassifierTree {
	private static final long serialVersionUID = 968301386365352742L;
	/** Cleanup after the tree has been built. */
	boolean m_cleanup = true;
	/** Confidence level */
	private float m_CF = 0.25f;
	/** Subtree raising to be performed? */
	private boolean m_subtreeRaising = true;

	/**
	 * Constructor for pruneable tree structure. Stores reference to associated
	 * training data at each node.
	 *
	 * @param toSelectLocModel
	 *            selection method for local splitting model
	 * @param pruneTree
	 *            true if the tree is to be pruned
	 * @param cf
	 *            the confidence factor for pruning
	 * @param raiseTree
	 * @param cleanup
	 * @throws Exception
	 *             if something goes wrong
	 */
	public C45tree(ModelSelection toSelectLocModel) throws Exception {
		super(toSelectLocModel);
	}

	/**
	 * Method for building a pruneable classifier tree.
	 *
	 * @param data
	 *            the data for building the tree
	 * @throws Exception
	 *             if something goes wrong
	 */
	public void buildClassifier(Instances data) throws Exception {
		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();
		// Dataselect dataselect = new Dataselect();
		// Stack<Pairs> pairstack = dataselect.buildClassifier(data);

		// buildTree(data, pairstack, 0, "./" + data.relationName() + "/");
		buildTree(data, 0, "./" + data.relationName() + "/");
//		collapse();
//		prune();
		if (m_cleanup) {
			cleanup(new Instances(data, 0));
		}
	}

	/**
	 * Collapses a tree to a node if training error doesn't increase.
	 */
	public final void collapse() {

		double errorsOfSubtree;
		double errorsOfTree;
		int i;

		if (!m_isLeaf) {
			errorsOfSubtree = getTrainingErrors();
			errorsOfTree = localModel().distribution().numIncorrect();
			if (errorsOfSubtree >= errorsOfTree - 1E-3) {

				// Free adjacent trees
				m_sons = null;
				m_isLeaf = true;

				// Get NoSplit Model for tree.
				m_localModel = new NoSplit(localModel().distribution());
				//set most class as Split point
				double[] nbObjPreClass = new double[m_train.numClasses()];
				for (int nbins = 0; nbins < m_train.numInstances(); nbins++) {
					Instance Obj = m_train.instance(nbins);
					nbObjPreClass[(int) Obj.classValue()]++;
				}
				int classlable=Utils.maxIndex(nbObjPreClass);
				Instances splitpoint=new Instances(m_train, m_train.numInstances());
				for (int j = 0; j < m_train.numInstances(); j++) {
					if (m_train.instance(j).classValue()==classlable) {
						splitpoint.add(m_train.instance(j));
					}
				}
				m_localModel.setSplitPoint(splitpoint);
			} else
				for (i = 0; i < m_sons.length; i++)
					if(son(i)!=null)
					son(i).collapse();
		}
	}

	/**
	 * Returns a newly created tree.
	 *
	 * @param data
	 *            the data to work with
	 * @return the new tree
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected ClassifierTree getNewTree(Instances data, int runtime, String dir) throws Exception {
		C45tree newTree = new C45tree(m_toSelectModel);
		newTree.buildTree((Instances) data, ++runtime, dir);

		return newTree;
	}

	protected ClassifierTree getNewTree(Instances data, Stack<Pairs> pairstack, int runtime, String dir)
			throws Exception {
		C45tree newTree = new C45tree(m_toSelectModel);
		newTree.buildTree((Instances) data, pairstack, ++runtime, dir);

		return newTree;
	}

	/**
	 * Prunes a tree using C4.5's pruning procedure.
	 *
	 * @throws Exception
	 *             if something goes wrong
	 */
	public void prune() throws Exception {

		double errorsLargestBranch;
		double errorsLeaf;
		double errorsTree;
		int indexOfLargestBranch;
		C45tree largestBranch;
		int i;

		if (!m_isLeaf) {

			// Prune all subtrees.
			for (i = 0; i < m_sons.length; i++)
				if(son(i)!=null)
				son(i).prune();

			// Compute error for largest branch
			indexOfLargestBranch = localModel().distribution().maxBag();
			if (m_subtreeRaising) {
				errorsLargestBranch = son(indexOfLargestBranch).getEstimatedErrorsForBranch((Instances) m_train);
			} else {
				errorsLargestBranch = Double.MAX_VALUE;
			}

			// Compute error if this Tree would be leaf
			errorsLeaf = getEstimatedErrorsForDistribution(localModel().distribution());

			// Compute error for the whole subtree
			errorsTree = getEstimatedErrors();

			// Decide if leaf is best choice.
			if (Utils.smOrEq(errorsLeaf, errorsTree + 0.1) && Utils.smOrEq(errorsLeaf, errorsLargestBranch + 0.1)) {

				// Free son Trees
				m_sons = null;
				m_isLeaf = true;

				// Get NoSplit Model for node.
				m_localModel = new NoSplit(localModel().distribution());
				//set most class as Split point
				double[] nbObjPreClass = new double[m_train.numClasses()];
				for (int nbins = 0; nbins < m_train.numInstances(); nbins++) {
					Instance Obj = m_train.instance(nbins);
					nbObjPreClass[(int) Obj.classValue()]++;
				}
				int classlable=Utils.maxIndex(nbObjPreClass);
				Instances splitpoint=new Instances(m_train, m_train.numInstances());
				for (int j = 0; j < m_train.numInstances(); j++) {
					if (m_train.instance(j).classValue()==classlable) {
						splitpoint.add(m_train.instance(j));
					}
				}
				m_localModel.setSplitPoint(splitpoint);
				if(m_localModel.m_splitPoint.numInstances()==0)
					System.err.println("111111");
				return;
			}

			// Decide if largest branch is better choice
			// than whole subtree.
			if (Utils.smOrEq(errorsLargestBranch, errorsTree + 0.1)) {
				largestBranch = son(indexOfLargestBranch);
				m_sons = largestBranch.m_sons;
				m_localModel = largestBranch.localModel();
				m_isLeaf = largestBranch.m_isLeaf;
				newDistribution(m_train);
				prune();
			}
		}
	}

	private C45tree son(int index) {

		return (C45tree) m_sons[index];
	}

	/**
	 * Method just exists to make program easier to read.
	 * 
	 * @return the local split model
	 */
	private ClassifierSplitModel localModel() {

		return m_localModel;
	}

	/**
	 * Computes estimated errors for leaf.
	 * 
	 * @param theDistribution
	 *            the distribution to use
	 * @return the estimated errors
	 */
	private double getEstimatedErrorsForDistribution(Distribution theDistribution) {

		if (Utils.eq(theDistribution.total(), 0))
			return 0;
		else
			return theDistribution.numIncorrect()
					+ Stats.addErrs(theDistribution.total(), theDistribution.numIncorrect(), m_CF);
	}

	/**
	 * Computes errors of tree on training data.
	 * 
	 * @return the training errors
	 */
	private double getTrainingErrors() {

		double errors = 0;
		int i;

		if (m_isLeaf)
			return localModel().distribution().numIncorrect();
		else {
			for (i = 0; i < m_sons.length; i++)
				if(son(i)!=null)
				errors = errors + son(i).getTrainingErrors();
			return errors;
		}
	}

	/**
	 * Computes estimated errors for tree.
	 * 
	 * @return the estimated errors
	 */
	private double getEstimatedErrors() {

		double errors = 0;
		int i;

		if (m_isLeaf)
			return getEstimatedErrorsForDistribution(localModel().distribution());
		else {
			for (i = 0; i < m_sons.length; i++)
				if(son(i)!=null)
				errors = errors + son(i).getEstimatedErrors();
			return errors;
		}
	}

	/**
	 * Computes estimated errors for one branch.
	 *
	 * @param data
	 *            the data to work with
	 * @return the estimated errors
	 * @throws Exception
	 *             if something goes wrong
	 */
	private double getEstimatedErrorsForBranch(Instances data) throws Exception {

		Instances[] localInstances;
		double errors = 0;
		int i;

		if (m_isLeaf)
			return getEstimatedErrorsForDistribution(new Distribution(data));
		else {
			Distribution savedDist = localModel().m_distribution;
			localModel().resetDistribution(data);
			localInstances = (Instances[]) localModel().split(data);
			localModel().m_distribution = savedDist;
			for (i = 0; i < m_sons.length; i++)
				if(son(i)!=null)
				errors = errors + son(i).getEstimatedErrorsForBranch(localInstances[i]);
			return errors;
		}
	}

	/**
	 * Computes new distributions of instances for nodes in tree.
	 *
	 * @param data
	 *            the data to compute the distributions for
	 * @throws Exception
	 *             if something goes wrong
	 */
	private void newDistribution(Instances data) throws Exception {

		Instances[] localInstances;
		localModel().resetDistribution(data);
		if(data==null)
			System.out.println("@@@@@@@@@@@@@");
		m_train = data;
		if (!m_isLeaf) {
			localInstances = (Instances[]) localModel().split(data);
			for (int i = 0; i < m_sons.length; i++)
				m_sons[i] = getNewTree(localInstances[i], 0, null);
		} else {

			// Check whether there are some instances at the leaf now!
			if (!Utils.eq(data.sumOfWeights(), 0)) {
				m_isEmpty = false;
			}
		}
	}
}
