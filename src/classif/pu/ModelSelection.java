
package classif.pu;

import weka.core.Instances;
import weka.core.RevisionHandler;

import java.io.Serializable;
import java.util.Stack;

import items.Pairs;

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
