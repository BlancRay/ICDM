package nwafu.dm.tsc.classif.pu;

import java.util.Stack;

import nwafu.dm.tsc.items.Pairs;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

public class C45ModelSelection extends ModelSelection {

	/** for serialization */
	private static final long serialVersionUID = 3372204862440821989L;

	/** All the training data */
	private Instances m_allData; //
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
		Split bestModel = null;
		Split[] currentModel;
		double minResult;
		int validModels = 0;
		double averageInfoGain = 0;
		try {

			// Check if all Instances belong to one class or if not
			// enough Instances to split.
			checkDistribution = new Distribution(data);
			noSplitModel = new NoSplit(checkDistribution);
			if (data == null) {
				System.out.println("!!!!!!!!");
			}
			noSplitModel.setSplitPoint(data);
			if (Utils.sm(checkDistribution.total(), 3)
					|| Utils.eq(checkDistribution.total(), checkDistribution.perClass(checkDistribution.maxClass())))
				return noSplitModel;
//			 RandomSelect dataselect = new RandomSelect();
//			 Stack<Pairs> pairs = dataselect.buildClassifier(data);
			SelectPN dataselect = new SelectPN();
			Stack<Pairs> pairs = dataselect.buildClassifier(data);

			currentModel = new Split[pairs.size()];
			// For each attribute.
			for (int i = 0; i < currentModel.length; i++) {
				currentModel[i] = new Split(pairs.pop());
				currentModel[i].buildClassifier(data);
				averageInfoGain = averageInfoGain + currentModel[i].infoGain();
				validModels++;
			}

			// Check if any useful split was found.
			if (validModels == 0) {
				// for (int j2 = 0; j2 <
				// noSplitModel.getSplitPoint().numInstances(); j2++) {
				// System.out.println(noSplitModel.getSplitPoint().instance(j2));
				// }
				return noSplitModel;
			}
			averageInfoGain = averageInfoGain / (double) validModels;
			// Find "best" attribute to split on.
			minResult = 0;

			for (int j = 0; j < currentModel.length; j++) {
				if (currentModel[j].checkModel())
					/*
					 * if ((currentModel[j].infoGain() >= (averageInfoGain -
					 * 1E-3)) && Utils.gr(currentModel[j].gainRatio(),
					 * minResult)) { bestModel = currentModel[j]; minResult =
					 * currentModel[j].gainRatio(); }
					 */
					if ((currentModel[j].infoGain() >= minResult)) {
						minResult = currentModel[j].infoGain();
						bestModel = currentModel[j];
					}
			}

			// Check if useful split was found.
			if (Utils.eq(minResult, 0)) {
				// for (int j2 = 0; j2 <
				// noSplitModel.getSplitPoint().numInstances(); j2++) {
				// System.out.println(noSplitModel.getSplitPoint().instance(j2));
				// }
				return noSplitModel;
			}

			// Set the split point analogue to C45 if attribute numeric.
			if (m_allData != null)
				// bestModel.setSplitPoint();
				// for (int j2 = 0; j2 <
				// bestModel.setSplitPoint().numInstances(); j2++) {
				// System.out.println(bestModel.setSplitPoint().instance(j2));
				// }
				return bestModel;

		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	public String getRevision() {
		return RevisionUtils.extract("$Revision: 1.11 $");
	}

	@SuppressWarnings("unchecked")
	@Override
	public ClassifierSplitModel selectModel(Instances data, Stack<Pairs> stack) throws Exception {

		NoSplit noSplitModel = null;
		Distribution checkDistribution;
		Split bestModel = null;
		Split[] currentModel;
		Stack<Pairs> pairs=(Stack<Pairs>) stack.clone();
		double minResult;
		int validModels = 0;
		double averageInfoGain = 0;
		try {
			
			// Check if all Instances belong to one class or if not
			// enough Instances to split.
			checkDistribution = new Distribution(data);
			noSplitModel = new NoSplit(checkDistribution);
			if(data==null)
			{
				System.out.println("!!!!!!!!");
			}
			noSplitModel.setSplitPoint(data);
			if (Utils.sm(checkDistribution.total(),2) ||
					  Utils.eq(checkDistribution.total(),
						   checkDistribution.perClass(checkDistribution.maxClass())))
					return noSplitModel;

			currentModel = new Split[pairs.size()];
			// For each attribute. 
			for (int i = 0; i < currentModel.length; i++) {
				currentModel[i] = new Split(pairs.pop());
				currentModel[i].buildClassifier(data);
				if (currentModel[i].checkModel())
					if (m_allData != null) {
						averageInfoGain = averageInfoGain+ currentModel[i].infoGain();
						validModels++;
					}
					else
						currentModel[i] = null;
			}

			// Check if any useful split was found.
			if (validModels == 0) {
//				for (int j2 = 0; j2 < noSplitModel.getSplitPoint().numInstances(); j2++) {
//					System.out.println(noSplitModel.getSplitPoint().instance(j2));
//				}
				return noSplitModel;
			}
			averageInfoGain = averageInfoGain / (double) validModels;
			// Find "best" attribute to split on.
			minResult = 0;
			
			for (int j = 0; j < currentModel.length; j++) {
				if (currentModel[j].checkModel())
				/*if ((currentModel[j].infoGain() >= (averageInfoGain - 1E-3))
						&& Utils.gr(currentModel[j].gainRatio(), minResult))
				{
					bestModel = currentModel[j];
					minResult = currentModel[j].gainRatio();
				}*/
					if ((currentModel[j].infoGain() >= minResult)){
						minResult = currentModel[j].infoGain();
						bestModel = currentModel[j];
					}
			}

			// Check if useful split was found.
			if (Utils.eq(minResult, 0)) {
//				for (int j2 = 0; j2 < noSplitModel.getSplitPoint().numInstances(); j2++) {
//					System.out.println(noSplitModel.getSplitPoint().instance(j2));
//				}
				return noSplitModel;
			}

			// Set the split point analogue to C45 if attribute numeric.
			if (m_allData != null)
//				bestModel.setSplitPoint();
//			for (int j2 = 0; j2 < bestModel.setSplitPoint().numInstances(); j2++) {
//				System.out.println(bestModel.setSplitPoint().instance(j2));
//			}
			return bestModel;

		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

}
