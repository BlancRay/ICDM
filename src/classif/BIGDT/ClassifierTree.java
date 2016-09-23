package classif.BIGDT;

import java.io.File;
import java.util.Stack;

import items.MonoDoubleItemSet;
import items.Pairs;
import items.Sequence;
import items.SymbolicSequence;
import tools.DrawAllSequences;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class for handling a tree structure used for
 * classification.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 10256 $
 */
public class ClassifierTree{

  /** for serialization */
  static final long serialVersionUID = -8722249377542734193L;
  
  /** The model selection method. */  
  protected ModelSelection m_toSelectModel;     

  /** Local model at node. */
  protected ClassifierSplitModel m_localModel;  

  /** References to sons. */
  protected ClassifierTree [] m_sons;           

  /** True if node is leaf. */
  protected boolean m_isLeaf;                   

  /** True if node is empty. */
  protected boolean m_isEmpty;                  

  /** The training instances. */
  protected Instances m_train;                  

  /** The pruning instances. */
  protected Distribution m_test;     

  /** The id for the node. */


  /**
   * Constructor. 
   */
  public ClassifierTree(ModelSelection toSelectLocModel) {
    
    m_toSelectModel = toSelectLocModel;
  }


  /**
   * Method for building a classifier tree.
   *
   * @param data the data to build the tree from
   * @throws Exception if something goes wrong
   */
  public void buildClassifier(Instances data) throws Exception {

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    Dataselect dataselect=new Dataselect();
	Stack<Pairs> pairstack= dataselect.buildClassifier(data);
    buildTree(data,pairstack,0,"./"+data.relationName()+"/");
  }

  /**
   * Builds the tree structure.
   *
   * @param data the data for which the tree structure is to be
   * generated.
   * @param keepData is training data to be kept?
   * @throws Exception if something goes wrong
   */

	public void buildTree(Instances data,Stack<Pairs> pairstack,int runtime,String dir) throws Exception {
		String loc=null;
		Instances[] localInstances;
		m_test = null;
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;
		m_train = data;
		
		m_localModel = m_toSelectModel.selectModel(data, pairstack);
		
		Instances ins = m_localModel.getSplitPoint();
		Sequence[] sequences = new Sequence[ins.numInstances()];
		for (int i = 0; i < sequences.length; i++) {
			Instance sample = ins.instance(i);
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
			int shift = (sample.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < sequence.length; t++) {
				sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
			}
			sequences[i] = new Sequence(sequence);
		}
		Stack<Pairs> pairstack_store = new Stack<Pairs>();
		while (!pairstack.isEmpty()) {
			if (pairstack.peek().getDistance() == sequences[0].distance(sequences[1])){
				pairstack.pop();
				continue;}
			pairstack_store.push(pairstack.pop());
		}
		
		while(!pairstack_store.isEmpty()){
			pairstack.push(pairstack_store.pop());
		}
		
		
//		if (runtime == 0)
//			plot(data, dir, runtime,"root node");
//		System.out.println(m_localModel.numSubsets());
		if (m_localModel.numSubsets() > 1) {
//			plot(m_localModel.getSplitPoint(),dir, runtime,"split data");
			localInstances = m_localModel.split(data);
			data = null;
			m_sons = new ClassifierTree[m_localModel.numSubsets()];
			int flg=0;
			for (int i = 0; i < m_sons.length; i++) {
				if (localInstances[i].numInstances() != 0){
					if (flg == 0) {
						System.out.println("left branch:");
						// loc = dir + "left/";
						// plot(localInstances[i], loc, runtime,"left branch");
					} else {
						System.out.println("right branch:");
						// loc = dir + "right/";
						// plot(localInstances[i], loc, runtime,"right branch");
					}
					flg = 1;
//					m_sons[i] = getNewTree(localInstances[i], runtime, loc);
					m_sons[i] = getNewTree(localInstances[i],pairstack ,runtime, loc);
					localInstances[i] = null;
				}
			}
		} else {
			m_isLeaf = true;
			data = null;
		}
	}
	public void buildTree(Instances data,int runtime,String dir) throws Exception {
		String loc=null;
		Instances[] localInstances;
		m_test = null;
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;
		m_train = data;
		RandomSelect dataselect = new RandomSelect();
		Stack<Pairs> pairstack = dataselect.buildClassifier(data);
		
		m_localModel = m_toSelectModel.selectModel(data,pairstack);
//		if (runtime == 0)
//			plot(data, dir, runtime,"root node");
//		System.out.println(m_localModel.numSubsets());
		if (m_localModel.numSubsets() > 1) {
//			plot(m_localModel.getSplitPoint(),dir, runtime,"split data");
			localInstances = m_localModel.split(data);
			data = null;
			m_sons = new ClassifierTree[m_localModel.numSubsets()];
			int flg=0;
			for (int i = 0; i < m_sons.length; i++) {
				if (localInstances[i].numInstances() != 0){
//					if (flg == 0) {
//						System.out.println("left branch:");
//						// loc = dir + "left/";
//						// plot(localInstances[i], loc, runtime,"left branch");
//					} else {
//						System.out.println("right branch:");
//						// loc = dir + "right/";
//						// plot(localInstances[i], loc, runtime,"right branch");
//					}
					flg = 1;
					m_sons[i] = getNewTree(localInstances[i], runtime, loc);
					localInstances[i] = null;
				}
				else {
					System.out.println(localInstances[i].numInstances());
				}
			}
		} else {
			m_isLeaf = true;
			data = null;
		}
	}


  /** 
   * Classifies an instance.
   *
   * @param instance the instance to classify
   * @return the classification
   * @throws Exception if something goes wrong
   */
	public double classifyInstance(Instance instance) throws Exception {
		double classlable = -1;
		if (m_isLeaf){
			if (localModel().m_splitPoint==null) {
				System.out.println("!!!");
			}
			if (localModel().getSplitPoint()==null) {
				System.out.println("!!!123123");
			}
			classlable = localModel().whichSubset(instance);
//			for (int j2 = 0; j2 < localModel().getSplitPoint().numInstances(); j2++) {
//				System.out.println(localModel().getSplitPoint().instance(j2));
//			}
//			System.out.println(classlable);
//			System.out.println(localModel().m_splitPoint.classAttribute().value((int)classlable));
			
		return instance.classAttribute().indexOfValue(localModel().m_splitPoint.classAttribute().value((int)classlable));
		}
		else{
			int treeindex= localModel().whichSubset(instance);
//			if (treeindex == 0)
//				System.out.println("<--");
//			else
//				System.out.println("-->");
			return son(treeindex).classifyInstance(instance);
		}
	}

	/**
	 * Cleanup in order to save memory.
	 * 
	 * @param justHeaderInfo
	 */
  public final void cleanup(Instances justHeaderInfo) {

		m_train = justHeaderInfo;
		m_test = null;
		if (!m_isLeaf)
			for (int i = 0; i < m_sons.length; i++)
				if(m_sons[i]!=null)
				m_sons[i].cleanup(justHeaderInfo);
	}

  /** 
   * Returns class probabilities for a weighted instance.
   *
   * @param instance the instance to get the distribution for
   * @param useLaplace whether to use laplace or not
   * @return the distribution
   * @throws Exception if something goes wrong
   */

  /**
   * Assigns a uniqe id to every node in the tree.
   * 
   * @param lastID the last ID that was assign
   * @return the new current ID
   */

  /**
   *  Returns the type of graph this classifier
   *  represents.
   *  @return Drawable.TREE
   */   
  public int graphType() {
      return Drawable.TREE;
  }

  /**
   * Returns graph describing the tree.
   *
   * @throws Exception if something goes wrong
   * @return the tree as graph
   */

  /**
   * Returns a newly created tree.
   *
   * @param data the training data
   * @return the generated tree
   * @throws Exception if something goes wrong
   */
	protected ClassifierTree getNewTree(Instances data, int runtime, String dir)
			throws Exception {

		ClassifierTree newTree = new ClassifierTree(m_toSelectModel);
		newTree.buildTree(data,  ++runtime, dir);

		return newTree;
	}
	protected ClassifierTree getNewTree(Instances data,Stack<Pairs> pairstack, int runtime, String dir)
			throws Exception {
		C45tree newTree = new C45tree(m_toSelectModel);
		newTree.buildTree((Instances) data, pairstack, ++runtime, dir);

		return newTree;
	}
  /**
   * Method just exists to make program easier to read.
   */
  private ClassifierSplitModel localModel() {
    
    return (ClassifierSplitModel)m_localModel;
  }
  
  /**
   * Method just exists to make program easier to read.
   */
  private ClassifierTree son(int index) {
    
    return (ClassifierTree)m_sons[index];
  }
  
  private void plot(Instances instances,String dir,int runtime,String branch){
	  File repSave=new File(dir);
	  SymbolicSequence[] m = new SymbolicSequence[instances.numInstances()];
		for (int i = 0; i < instances.numInstances(); i++) {
			 System.out.println(instances.instance(i));
			Instance sample = instances.instance(i);
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
			int shift = (sample.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < sequence.length; t++) {
				sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
			}
			m[i] = new SymbolicSequence(sequence);
		}
		new DrawAllSequences(repSave,m, 0).plot(branch+"_"+runtime);
  }
}
