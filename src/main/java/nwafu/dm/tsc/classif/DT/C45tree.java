package nwafu.dm.tsc.classif.DT;

import weka.core.Instances;

public class C45tree 
  extends ClassifierTree {

/** True if the tree is to be pruned. */
  boolean m_pruneTheTree = false;

  /** The confidence factor for pruning. */
  float m_CF = 0.25f;

  /** Is subtree raising to be performed? */
  boolean m_subtreeRaising = true;

  /** Cleanup after the tree has been built. */
  boolean m_cleanup = true;

  /**
   * Constructor for pruneable tree structure. Stores reference
   * to associated training data at each node.
   *
   * @param toSelectLocModel selection method for local splitting model
   * @param pruneTree true if the tree is to be pruned
   * @param cf the confidence factor for pruning
   * @param raiseTree
   * @param cleanup
   * @throws Exception if something goes wrong
   */
  public C45tree(ModelSelection toSelectLocModel)
       throws Exception {

    super(toSelectLocModel);
  }


  /**
   * Method for building a pruneable classifier tree.
   *
   * @param data the data for building the tree
   * @throws Exception if something goes wrong
   */
  public void buildClassifier(Instances data) throws Exception {

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
   buildTree(data,0,"./"+data.relationName()+"/");
   if (m_cleanup) {
     cleanup(new Instances(data, 0));
   }
  }

  /**
   * Returns a newly created tree.
   *
   * @param data the data to work with
   * @return the new tree
   * @throws Exception if something goes wrong
   */
  protected ClassifierTree getNewTree(Instances data,int runtime,String dir) throws Exception {
    
		C45tree newTree = new C45tree(m_toSelectModel);
		newTree.buildTree((Instances) data, ++runtime,dir);

		return newTree;
  }
}
