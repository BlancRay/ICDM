package classif.DT;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class Tree {
	protected Node m_toSelectModel;
	protected Tree[] m_sons;
	protected boolean m_isLeaf;
	protected boolean m_isEmpty;
	protected Instances m_train;
	protected Distribution m_test;
	protected Node m_localModel;

	public Tree(Node toSelectLocModel) throws Exception {
		super();
	}

	protected Tree getNewTree(Instances data) throws Exception {

		Tree newTree = new Tree(m_toSelectModel);
		newTree.buildTree((Instances) data);

		return newTree;
	}

	public void buildClassifier(Instances data) throws Exception {
		buildTree(data);
	}

	public void buildTree(Instances data) throws Exception {

		Instances[] localInstances;

		m_test = null;
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;
		m_localModel = m_toSelectModel.selectModel(data);
		if (m_localModel.getNumSubsets() > 1) {
			localInstances = m_localModel.split(data);
			data = null;
			m_sons = new Tree[m_localModel.getNumSubsets()];
			for (int i = 0; i < m_sons.length; i++) {
				m_sons[i] = getNewTree(localInstances[i]);
				localInstances[i] = null;
			}
		} else {
			m_isLeaf = true;
			if (Utils.eq(data.sumOfWeights(), 0))
				m_isEmpty = true;
			data = null;
		}
	}

	private Tree son(int index) {

		return (Tree) m_sons[index];
	}

	public double classifyInstance(Instance instance) throws Exception {

		double maxProb = -1;
		double currentProb;
		int maxIndex = 0;
		int j;

		for (j = 0; j < instance.numClasses(); j++) {
			currentProb = getProbs(j, instance, 1);
			if (Utils.gr(currentProb, maxProb)) {
				maxIndex = j;
				maxProb = currentProb;
			}
		}

		return (double) maxIndex;
	}

	private double getProbs(int classIndex, Instance instance, double weight) throws Exception {

		double prob = 0;

		if (m_isLeaf) {
			return weight * localModel().classProb(classIndex, instance, -1);
		} else {
			int treeIndex = localModel().whichSubset(instance);
			if (treeIndex == -1) {
				double[] weights = localModel().weights(instance);
				for (int i = 0; i < m_sons.length; i++) {
					if (!son(i).m_isEmpty) {
						prob += son(i).getProbs(classIndex, instance, weights[i] * weight);
					}
				}
				return prob;
			} else {
				if (son(treeIndex).m_isEmpty) {
					return weight * localModel().classProb(classIndex, instance, treeIndex);
				} else {
					return son(treeIndex).getProbs(classIndex, instance, weight);
				}
			}
		}
	}
	
	  private Node localModel() {
		    
		    return (Node)m_localModel;
		  }
}
