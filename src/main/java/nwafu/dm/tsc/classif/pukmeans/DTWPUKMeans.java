package nwafu.dm.tsc.classif.pukmeans;

import nwafu.dm.tsc.items.ClassedSequence;
import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.Sequences;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import nwafu.dm.tsc.classif.kmeans.KMeansSymbolicSequence;
import weka.classifiers.AbstractClassifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class DTWPUKMeans extends AbstractClassifier {

	private static final long serialVersionUID = 1717176683182910935L;
	protected ArrayList<ClassedSequence> prototypes;
	protected Sequence[] clusteredUNL = null;
	protected ArrayList<Sequence> unlabeledData = new ArrayList<Sequence>();
	protected ArrayList<Sequence> posData = new ArrayList<Sequence>();
	protected int nbClustersinUNL;
	private double df = 0.25;
	protected double[][] distances;
	private static final double minObj = 1;
	private static final String pClass="1.0";
	private static final String uClass="-1.0";

	Sequence[] sequences;
	Instances trainingData = null;

	public DTWPUKMeans() {
		super();
	}

	/**
	 * train from dataset
	 * 
	 * @param data
	 */
	public void buildClassifier(Instances data) {
		trainingData = new Instances(data);
		prototypes = new ArrayList<ClassedSequence>();
		Instances posIns=new Instances(trainingData,0);
		posIns.add(init());
		for (int i = 0; i < trainingData.numInstances(); i++) {
			trainingData.instance(i).setClassValue(uClass);
		}
		sequences = new Sequence[trainingData.numInstances()];
		for (int i = 0; i < sequences.length; i++) {
			Instance sample = trainingData.instance(i);
			sequences[i] = InsToSeq(sample);
			unlabeledData.add(sequences[i]);
		}
//		Sequence[] pos=new Sequence[100];
//		pos[0] = InsToSeq(posIns.instance(0));
		posData.add(InsToSeq(posIns.instance(0)));
		for (int i = 1; i < data.numInstances()*df; i++) {
			int unlpos = findUnlabeledNN(posIns, unlabeledData);
			posIns.add(trainingData.instance(unlpos));
			trainingData.delete(unlpos);
//			pos[i] = unlabeledData.get(unlpos);
			posData.add(unlabeledData.get(unlpos));
			unlabeledData.remove(unlpos);
//			ClassedSequence s = new ClassedSequence(pos[i], pClass);
//			prototypes.add(s);
		}
		
		KMeansSymbolicSequence km =new KMeansSymbolicSequence(nbClustersinUNL, posData);
		km.cluster();
		Sequence[] pos = new Sequence[nbClustersinUNL];
		pos = km.centers;
		for (int i = 0; i < pos.length; i++) {
			if (pos[i] != null) {
				ClassedSequence s = new ClassedSequence(pos[i], pClass);
				prototypes.add(s);
			}
		}

		if (distances == null) {
			initDistances();
		}
		clusteredUNL = new Sequence[nbClustersinUNL];
		// if the class is empty, continue
		boolean isBig;
		ArrayList<Integer>[] affectation = new ArrayList[nbClustersinUNL];
		int runtime = 0;
		do {
			if (runtime > 10)
				nbClustersinUNL -= 1;
			isBig = true;
			KMeansCachedSymbolicSequence kmeans = new KMeansCachedSymbolicSequence(nbClustersinUNL, unlabeledData, distances);
			kmeans.cluster();
			clusteredUNL = kmeans.centers;
			affectation = kmeans.affectation;
			for (ArrayList<Integer> Eachaffectation : affectation) {
				if (Eachaffectation.size() < minObj) {
					isBig = false;
					break;
				}
			}
			for (int k = 0; k < kmeans.centers.length; k++) {
				if (kmeans.centers[k] != null) { // ~ if empty cluster
					// find the center
					clusteredUNL[k] = kmeans.centers[k];
				}
			}
			runtime++;
		} while (isBig == false);

		for (int i = 0; i < clusteredUNL.length; i++) {
			if(clusteredUNL[i]!=null){
				ClassedSequence s=new ClassedSequence(clusteredUNL[i], uClass);
				prototypes.add(s);
			}
		}
//		findN(pos, clusteredUNL);
//		searchPOSNEG(clusteredUNL,prototypes);
		
		
//		for (ClassedSequence s2 : prototypes) {
//			System.out.println(s2.classValue+"  "+s2.sequence);
//		}
	}

	/**
	 * test on one sample
	 * 
	 * @param sample
	 * @return p(y|sample) forall y
	 * @throws Exception
	 */
	@Override
	public double classifyInstance(Instance sample) throws Exception {
		// transform instance to sequence
		MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
		int shift = (sample.classIndex() == 0) ? 1 : 0;
		for (int t = 0; t < sequence.length; t++) {
			sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
		}
		Sequence seq = new Sequence(sequence);

		double minD = Double.MAX_VALUE;
		String classValue = null;
		for (ClassedSequence s : prototypes) {
			double tmpD = seq.distance(s.sequence);
			if (tmpD < minD) {
				minD = tmpD;
				classValue = s.classValue;
			}
		}
		// System.out.println(classValue);
		return sample.classAttribute().indexOfValue(classValue);
	}

	public void setDf(double df) {
		this.df = df;
	}

	public void setNbClustersinUNL(int nbClustersinUNL) {
		this.nbClustersinUNL = nbClustersinUNL;
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

	protected void initDistances() {
		// if the class is empty, continue
		int nObjectsInClass = unlabeledData.size();
		distances = new double[nObjectsInClass][nObjectsInClass];
		for (int i = 0; i < nObjectsInClass; i++) {
			for (int j = i + 1; j < nObjectsInClass; j++) {
				distances[i][j] = unlabeledData.get(i).distance(unlabeledData.get(j));
				distances[j][i] = distances[i][j];
			}
		}
		// System.out.println("all distances cached");
	}
	
	protected void findN(Sequence[] pos, ArrayList<Sequence> unl) {
		double bigdist = Double.MIN_NORMAL;
		int Nposition = -1;
		for (int i = 0; i < unl.size(); i++) {
			double[] d = new double[pos.length];
			for (int j = 0; j < pos.length; j++) {
				d[j] = unl.get(i).distance(pos[j]);
			}
			if (d[Utils.minIndex(d)] > bigdist) {
				bigdist = d[Utils.minIndex(d)];
				Nposition = i;
			}
		}
		ClassedSequence s = new ClassedSequence(unl.get(Nposition), uClass);
		prototypes.add(s);
		System.out.println(unl.get(Nposition));
		unlabeledData.remove(Nposition);
	}
	
	protected void findN(Sequence[] pos, Sequence[] cluster) {
		double[] dist_pu = new double[cluster.length];
		for (int i = 0; i < cluster.length; i++) {
			double[] d=new double[pos.length];
			for (int j = 0; j < pos.length; j++) {
				d[j] = cluster[i].distance(pos[j]);
			}
			dist_pu[i] = Utils.kthSmallestValue(d, 1);
		}
		
		ClassedSequence s = new ClassedSequence(cluster[Utils.maxIndex(dist_pu)], uClass);
		prototypes.add(s);

//		for (int k = 0; k < dist_pu.length; k++) {
//			if (dist_pu[k] > Utils.kthSmallestValue(dist_pu, (int) (dist_pu.length*0.5))) {
//				ClassedSequence s = new ClassedSequence(cluster[k], uClass);
//				prototypes.add(s);
//			}
//		}
	}
	
	/**
	 * Split Unlabeled data to P or N
	 * @param sequences Unlabeled Sequence[]
	 * @param prototypes Prototypes of P and N
	 */
	protected void searchPOSNEG(Sequence[] sequences,ArrayList<ClassedSequence> prototypes) {
		ArrayList<ClassedSequence> pn=new ArrayList<ClassedSequence>(prototypes);
		for (Sequence seq : sequences) {
			double minD = Double.MAX_VALUE;
			String classValue = null;
			for (ClassedSequence s : pn) {
				double tmpD = seq.distance(s.sequence);
				if (tmpD < minD) {
					minD = tmpD;
					classValue = s.classValue;
				}
			}
			if (minD==0.0)
				continue;
			ClassedSequence s = new ClassedSequence(seq, classValue);
			prototypes.add(s);
		}
	}
	
	/**
	 * Find a reliable P from U with a positive example
	 * @param p
	 * @param u
	 * @return a reliable P
	 */
	private int findUnlabeledNN(Instances p, ArrayList<Sequence> u) {
		int unlpos = -1;
		double[][]updist=new double[u.size()][p.numInstances()];
		double[]mindist=new double[u.size()];
		for (int i = 0; i < u.size(); i++) {
			for (int j = 0; j < p.numInstances(); j++) {
				Sequence[] splitsequences = new Sequence[2];
				splitsequences[0] = u.get(i);
				splitsequences[1] = InsToSeq(p.instance(j));
				updist[i][j] = splitsequences[0].distance(splitsequences[1]);
			}
			// min distance from u to p
			mindist[i]=updist[i][Utils.minIndex(updist[i])];
		}
		unlpos=Utils.minIndex(mindist);
		return unlpos;
	}
	
	/**
	 * random select one positive example
	 * @return positive example
	 */
	protected Instance init() {
		Instance posSample = new DenseInstance(trainingData.numAttributes());
		posSample.setDataset(trainingData);
		while (posSample.classValue() != trainingData.classAttribute().indexOfValue(pClass)) {
			int rd = new Random().nextInt(trainingData.numInstances());
			if (trainingData.instance(rd).classValue() == trainingData.classAttribute().indexOfValue(pClass)) {
				posSample = trainingData.instance(rd);
				trainingData.delete(rd);
			}
		}
//		ClassedSequence s = new ClassedSequence(InsToSeq(posSample), pClass);
//		prototypes.add(s);
		return posSample;
	}
	
}
