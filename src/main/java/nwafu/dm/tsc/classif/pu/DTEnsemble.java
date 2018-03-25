package nwafu.dm.tsc.classif.pu;

import java.util.Arrays;
import java.util.Random;

import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.Sequence;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class DTEnsemble extends AbstractClassifier {
	private static final long serialVersionUID = 2907908860776080478L;
	// the classifier
	private ClassifyPOSC45 c45posunl[];
	private ClassifyPOSC45 puc45;
	private static final String pClass = "1.0";
	private static final String uClass = "-1.0";
	protected double ratio = 0.1;
	protected int nbInitPos = 1;
	protected int nbPOStree = 1;

	// training
	public void buildClassifier(Instances data) throws Exception {
		nbPOStree=Math.max(data.numInstances()/data.numClasses()/10, 25);
		c45posunl = new ClassifyPOSC45[nbPOStree];
		for (int i = 0; i < c45posunl.length; i++) {
			ratio=(i+1.0)/nbPOStree;
			Instances traindata = new Instances(data);
			// split data to P and U sets
			Instances posData = new Instances(traindata, 0);
			Instances unlData = new Instances(traindata, 0);

			posData=init(traindata,nbInitPos);
//			Instances posDatacopy = new Instances(posData);
//			Instances traindatacopy = new Instances(traindata);
//			Instances unlDatacopy = new Instances(traindatacopy,0);
			double[] dist = new double[traindata.numInstances()];
			int flg = 0;
/*//			PrintStream outPoscopy=new PrintStream("/stor9000/apps/users/NWSUAF/2008117287/ICDM/test/DTWD");
			while (traindatacopy.numInstances() != 0) {
				double[] mindist = findUnlabeledNNDTWD(posDatacopy, traindatacopy);
				int unlpos = Utils.minIndex(mindist);
				double now_dist = mindist[unlpos];
				dist[flg] = now_dist;
//				outPoscopy.println(traindatacopy.instance(unlpos));
				traindatacopy.instance(unlpos).setClassValue(pClass);
				posDatacopy.add(traindatacopy.instance(unlpos));
				traindatacopy.delete(unlpos);
				flg++;
			}
//			outPoscopy.close();
			double[] scccopy = SCC(dist, posDatacopy.numInstances() - nbInitPos);
			System.out.println("DTWD pos:\t" + Utils.maxIndex(scccopy) + "\t" + Utils.minIndex(scccopy));
			for (int j = posDatacopy.numInstances() - 1; j > Utils.maxIndex(scccopy); j--) {
				posDatacopy.instance(j).setClassValue(uClass);
				unlDatacopy.add(posDatacopy.instance(j));
				posDatacopy.delete(j);
			}

			*//**
			 * test of leaf node with 50% positive samples
			 *//*
			dist = new double[traindata.numInstances()];
			flg = 0;*/
//			PrintStream outPos = new PrintStream("/stor9000/apps/users/NWSUAF/2008117287/ICDM/test/DTW");
			while (traindata.numInstances() != 0) {
				double[] mindist = findUnlabeledNN(posData, traindata);
				int unlpos = Utils.minIndex(mindist);
				double now_dist = mindist[unlpos];
				dist[flg] = now_dist;
//				outPos.println(traindata.instance(unlpos));
				traindata.instance(unlpos).setClassValue(pClass);
				posData.add(traindata.instance(unlpos));
				traindata.delete(unlpos);
				flg++;
			}
//			outPos.close();
			double[] scc = SCC(dist, posData.numInstances() - nbInitPos);
			int index = Utils.maxIndex(scc);
			System.out.println("DTW pos:\t" + Utils.maxIndex(scc) + "\t" + Utils.minIndex(scc));
			for (int j = posData.numInstances() - 1; j > index; j--) {
				posData.instance(j).setClassValue(uClass);
				unlData.add(posData.instance(j));
				posData.delete(j);
			}
			System.out.println(ratio);
			c45posunl[i] = new ClassifyPOSC45(ratio);
			c45posunl[i].setDataset(posData, unlData);
			c45posunl[i].buildClassifier(null);
			/*if (Utils.maxIndex(scc) > Utils.maxIndex(scccopy)) {
				c45posunl[i] = new ClassifyPOSC45(ratio);
				c45posunl[i].setDataset(posData, unlData);
				c45posunl[i].buildClassifier(null);
				// Evaluation eval = new Evaluation(data);
				// eval.evaluateModel(c45posunl[i], data);
				// System.out.println(eval.errorRate()+eval.fMeasure(0));
			}
			else{
				c45posunl[i] = new ClassifyPOSC45(ratio);
				c45posunl[i].setDataset(posDatacopy, unlDatacopy);
				c45posunl[i].buildClassifier(null);
			}*/
		}
		double[] fmeasures = new double[nbPOStree];
		for (int j = 0; j < c45posunl.length; j++) {
			Evaluation evaltrain = new Evaluation(data);
			evaltrain.evaluateModel(c45posunl[j],data);
			fmeasures[j]=evaltrain.fMeasure(data.classAttribute().indexOfValue(pClass));
		}
		puc45=c45posunl[Utils.maxIndex(fmeasures)];
	}

	// classify
	public double classifyInstance(Instance instance) throws Exception {
		/*int[] classlabel = new int[instance.numClasses()];
		for (int i = 0; i < c45posunl.length; i++) {
			classlabel[(int) c45posunl[i].classifyInstance(instance)]++;
		}
		// System.out.println(claC45posunl.classifyInstance(instance));
		return Utils.maxIndex(classlabel);*/
		return puc45.classifyInstance(instance);
	}

	/**
	 * Minimum Distances between each U and P sets
	 * @param p Labeled Positive example
	 * @param u Unlabeled data
	 * @return Every u to P mindistance
	 */
	private double[] findUnlabeledNN(Instances p, Instances u) {
		double[][] updist = new double[u.numInstances()][p.numInstances()];
		double[] mindist = new double[u.numInstances()];
		for (int i = 0; i < u.numInstances(); i++) {
			for (int j = 0; j < p.numInstances(); j++) {
				Sequence[] splitsequences = new Sequence[2];
				splitsequences[0] = InsToSeq(u.instance(i));
				splitsequences[1] = InsToSeq(p.instance(j));
				updist[i][j] = splitsequences[0].distance(splitsequences[1]);
			}
			// min distance from u to p
			mindist[i] = updist[i][Utils.minIndex(updist[i])];
		}
		return mindist;
	}
	
	private double[] findUnlabeledNNDTWD(Instances p, Instances u) {
		double[][] updist = new double[u.numInstances()][p.numInstances()];
		double[] mindist = new double[u.numInstances()];
		for (int i = 0; i < u.numInstances(); i++) {
			for (int j = 0; j < p.numInstances(); j++) {
				Sequence[] splitsequences = new Sequence[2];
				splitsequences[0] = InsToSeq(u.instance(i));
				splitsequences[1] = InsToSeq(p.instance(j));
				updist[i][j] = splitsequences[0].distanceDTWD(splitsequences[1]);
			}
			// min distance from u to p
			mindist[i] = updist[i][Utils.minIndex(updist[i])];
		}
		return mindist;
	}


	/**
	 * Convert Instance to Sequence
	 * @param sample Instance to be convert
	 * @return Converted Sequence
	 */
	private Sequence InsToSeq(Instance sample) {
		MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
		int shift = (sample.classIndex() == 0) ? 1 : 0;
		for (int t = 0; t < sequence.length; t++) {
			sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
		}
		return new Sequence(sequence);
	}

	private double[] SCC(double[] mindist, int nbUNL) {
		double[] scc = new double[mindist.length - 1];
		int scci = 0;
		for (int i = 1; i < mindist.length; i++) {
			double abs = Math.abs(mindist[i] - mindist[i - 1]);
			double std = Math.sqrt(Utils.variance(Arrays.copyOfRange(mindist, 0, i + 1)));
			scc[scci] = (abs / std * (1.0 * (nbUNL - (i - 1)) / nbUNL));
			scci++;
		}
		return scc;
	}

	/**
	 * random select one positive example
	 * 
	 * @return positive example
	 */
	protected Instances init(Instances data,int nbinition) {
		Instances iniPosSample = new Instances(data,0);
		while (iniPosSample.numInstances()<nbinition) {
			int rd = new Random().nextInt(data.numInstances());
			if (data.instance(rd).classValue() == data.classAttribute().indexOfValue(pClass)) {
				iniPosSample.add(data.instance(rd));
				data.delete(rd);
			}
		}
		// ClassedSequence s = new ClassedSequence(InsToSeq(posSample), pClass);
		// prototypes.add(s);
		return iniPosSample;
	}
}
