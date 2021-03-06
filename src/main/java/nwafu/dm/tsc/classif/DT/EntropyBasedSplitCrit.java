package nwafu.dm.tsc.classif.DT;

/**
 * "Abstract" class for computing splitting criteria
 * based on the entropy of a class distribution.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.8 $
 */
public abstract class EntropyBasedSplitCrit
  extends SplitCriterion {

  /** for serialization */
  private static final long serialVersionUID = -2618691439791653056L;

  /** The log of 2. */
  protected static double log2 = Math.log(2);

  /**
   * Help method for computing entropy.
   */
  public final double logFunc(double num) {

    // Constant hard coded for efficiency reasons
    if (num < 1e-6)
      return 0;
    else
      return num*Math.log(num)/log2;
  }

  /**
   * Computes entropy of distribution before splitting.
   */
  public final double oldEnt(Distribution bags) {

    double returnValue = 0;
    int j;

    for (j=0;j<bags.numClasses();j++)
      returnValue = returnValue+logFunc(bags.perClass(j));
    return logFunc(bags.total())-returnValue; 
  }

  /**
   * Computes entropy of distribution after splitting.
   */
  public final double newEnt(Distribution bags) {
    
    double returnValue = 0;
    int i,j;

    for (i=0;i<bags.numBags();i++){
      for (j=0;j<bags.numClasses();j++)
	returnValue = returnValue+logFunc(bags.perClassPerBag(i,j));
      returnValue = returnValue-logFunc(bags.perBag(i));
    }
    return -returnValue;
  }

  /**
   * Computes entropy after splitting without considering the
   * class values.
   */
  public final double splitEnt(Distribution bags) {

    double returnValue = 0;
    int i;

    for (i=0;i<bags.numBags();i++)
      returnValue = returnValue+logFunc(bags.perBag(i));
    return logFunc(bags.total())-returnValue;
  }
}

