package classif.DT;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.HashMap;

import items.ClassedSequence;
import items.Sequence;

/**
 * Class for selecting a C4.5-type split for a given dataset.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.11 $
 */
public class C45ModelSelection extends ModelSelection {

	/** for serialization */
	private static final long serialVersionUID = 3372204862440821989L;

	/** Minimum number of objects in interval. */
	private int m_minNoObj=1;

	/** All the training data */
	private Instances m_allData; //
	protected ArrayList<ClassedSequence> prototypes;
	protected HashMap<String, ArrayList<Sequence>> classedData;
	protected HashMap<String, ArrayList<Integer>> indexClassedDataInFullData;
	protected int nbPrototypesPerClass[];
	protected Sequence[] sequences;
	protected String[] classMap;
	protected Instances trainingData = null;
	protected boolean fillPrototypes = true;

	/**
	 * Initializes the split selection method with the given parameters.
	 *
	 * @param minNoObj
	 *            minimum number of instances that have to occur in at least two
	 *            subsets induced by split
	 * @param allData
	 *            FULL training dataset (necessary for selection of split
	 *            points).
	 */
	public C45ModelSelection(Instances allData) {
		m_allData = allData;
	}

	/**
	 * Sets reference to training data to null.
	 */
	public void cleanup() {

		m_allData = null;
	}

	/**
	 * Selects C4.5-type split for the given dataset.
	 */
	public final ClassifierSplitModel selectModel(Instances data) {
		NoSplit noSplitModel = null;
		Distribution checkDistribution;
		C45Split bestModel = null;
		C45Split[][] currentModel;
		double minResult;
		double averageInfoGain = 0;
		int validModels = 0;
		int i, j;

		try {
			
			// Check if all Instances belong to one class or if not
			// enough Instances to split.
			checkDistribution = new Distribution(data);
			noSplitModel = new NoSplit(checkDistribution);
			noSplitModel.setSplitPoint(data);

			currentModel = new C45Split[data.numInstances()][data.numInstances()];
			// For each attribute.
			for (i = 0; i < data.numInstances(); i++) {
				for (j = i; j < data.numInstances(); j++) {

					// Apart from class attribute.
					if (data.instance(i).classValue() != data.instance(j).classValue()) {

						// Get models for current attribute.
						int[] pair = new int[data.numClasses()];
						pair[0] = i;
						pair[1] = j;
						currentModel[i][j] = new C45Split(pair);
						currentModel[i][j].buildClassifier(data);
						if (currentModel[i][j].checkModel())
							if (m_allData != null) {
								averageInfoGain = averageInfoGain + currentModel[i][j].infoGain();
								validModels++;
							}
						
					} else
						currentModel[i][j] = null;
				}
			}

			// Check if any useful split was found.
			if (validModels == 0){
//				for (int j2 = 0; j2 < noSplitModel.getSplitPoint().numInstances(); j2++) {
//					System.out.println(noSplitModel.getSplitPoint().instance(j2).classValue());
//				}
				return noSplitModel;}
			averageInfoGain = averageInfoGain / (double) validModels;

			// Find "best" attribute to split on.
			minResult = 0;
			for (i = 0; i < data.numInstances(); i++) {
				for (j = i; j < data.numInstances(); j++) {
					if ((data.instance(i).classValue() != data.instance(j).classValue())
							&& (currentModel[i][j].checkModel()))

						// Use 1E-3 here to get a closer approximation to the
						// original implementation.
//						if ((currentModel[i][j].infoGain() >= (averageInfoGain - 1E-3))
//								&& Utils.gr(currentModel[i][j].gainRatio(), minResult)) {
//						bestModel = currentModel[i][j];
//						minResult = currentModel[i][j].gainRatio();
//						}
						if(currentModel[i][j].infoGain()>minResult){
							minResult=currentModel[i][j].infoGain();
							bestModel = currentModel[i][j];
						}
				}
			}

			// Check if useful split was found.
			if (Utils.eq(minResult, 0)){
//				for (int j2 = 0; j2 < noSplitModel.getSplitPoint().numInstances(); j2++) {
//					System.out.println(noSplitModel.getSplitPoint().instance(j2).classValue());
//				}
				return noSplitModel;}

			// Set the split point analogue to C45 if attribute numeric.
			if (m_allData != null)
				bestModel.setSplitPoint();
//			for (int j2 = 0; j2 < bestModel.setSplitPoint().numInstances(); j2++) {
//				System.out.println(bestModel.setSplitPoint().instance(j2).classValue());
//			}
			return bestModel;

		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * Selects C4.5-type split for the given dataset.
	 */
	public final ClassifierSplitModel selectModel(Instances train, Instances test) {

		return selectModel(train);
	}

	public String getRevision() {
		return RevisionUtils.extract("$Revision: 1.11 $");
	}

}
