// I changed the codes here for BUG here.
//     by YANG ZHANG @ ITEE

package classif.POSC45;

import weka.core.Utils;

/**
 * Class for computing the information gain for a given distribution.
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.9 $
 */
public final class InfoGainSplitCrit extends EntropyBasedSplitCrit
{

	/** for serialization */
	private static final long serialVersionUID = 4892105020180728499L;

	/**
	 * This method is a straightforward implementation of the information gain
	 * criterion for the given distribution.
	 */
	public final double splitCritValue(Distribution bags)
	{
		System.out.println("OOOOOOOOPS, U have reached here");
		System.out
				.println("the original version of J48 does not show bugs because u cannot reach here.");
		System.exit(1);
		double numerator;

		numerator = oldEnt(bags) - newEnt(bags);

		// Splits with no gain are useless.
		if (Utils.eq(numerator, 0))
			return 0;

		// We take the reciprocal value because we want to minimize the
		// splitting criterion's value.
		return numerator / bags.total();
	}

	/**
	 * This method computes the information gain in the same way C4.5 does.
	 * 
	 * @param bags
	 *            the distribution
	 * @param totalNoInst
	 *            weight of ALL instances (including the ones with missing
	 *            values).
	 */
	public final double splitCritValue(Distribution bags, double totalNoInst)
	{

		double numerator;

		//Out.println("������InfoGainSplitCrit���У���ʹ��splitCritValue����");

		//Out.println("���ϼ���oldEnt(bags)��newEnt(bags)");
		
		numerator = (oldEnt(bags) - newEnt(bags));
		//Out.println("InfoGainCrit: " + oldEnt(bags) + "-" + newEnt(bags) + "="
		//		+ numerator);
		// Splits with no gain are useless.
		if (Utils.eq(numerator, 0))
			return 0;

		return numerator;
	}

	/**
	 * This method computes the information gain in the same way C4.5 does.
	 * 
	 * @param bags
	 *            the distribution
	 * @param totalNoInst
	 *            weight of ALL instances
	 * @param oldEnt
	 *            entropy with respect to "no-split"-model.
	 */
	public final double splitCritValue(Distribution bags, double totalNoInst,
			double oldEnt)
	{

		double numerator;
		numerator = (oldEnt(bags) - newEnt(bags));

		// Splits with no gain are useless.
		if (Utils.eq(numerator, 0))
			return 0;

		return numerator;
	}
}
