
package nwafu.dm.tsc.classif.pu;

import java.io.Serializable;
import java.util.Stack;

import nwafu.dm.tsc.items.Pairs;
import weka.core.Instances;
import weka.core.RevisionHandler;

public abstract class ModelSelection
  implements Serializable, RevisionHandler {

  /** for serialization */
  private static final long serialVersionUID = -4850147125096133642L;

  /**
   * Selects a model for the given dataset.
   *
   * @exception Exception if model can't be selected
   */
  public abstract ClassifierSplitModel selectModel(Instances data,Stack<Pairs> stack) throws Exception;
  public abstract ClassifierSplitModel selectModel(Instances data) throws Exception;
}
