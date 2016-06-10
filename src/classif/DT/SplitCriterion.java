package classif.DT;

import weka.core.RevisionHandler;

import java.io.Serializable;

/**
 * Abstract class for computing splitting criteria
 * with respect to distributions of class values.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.8 $
 */
public abstract class SplitCriterion
  implements Serializable, RevisionHandler {

  /** for serialization */
  private static final long serialVersionUID = 5490996638027101259L;

  /**
   * Computes result of splitting criterion for given distribution.
   *
   * @return value of splitting criterion. 0 by default
   */
  public double splitCritValue(Distribution bags){

    return 0;
  }

  /**
   * Computes result of splitting criterion for given training and
   * test distributions.
   *
   * @return value of splitting criterion. 0 by default
   */
  public double splitCritValue(Distribution train, Distribution test){

    return 0;
  }

  /**
   * Computes result of splitting criterion for given training and
   * test distributions and given number of classes.
   *
   * @return value of splitting criterion. 0 by default
   */
  public double splitCritValue(Distribution train, Distribution test,
			       int noClassesDefault){

    return 0;
  }

  /**
   * Computes result of splitting criterion for given training and
   * test distributions and given default distribution.
   *
   * @return value of splitting criterion. 0 by default
   */
  public double splitCritValue(Distribution train, Distribution test,
			       Distribution defC){

    return 0;
  }
}


