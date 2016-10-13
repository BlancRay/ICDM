package classif.pukmeans;

import items.ClassedSequence;
import items.MonoDoubleItemSet;
import items.Sequence;
import items.Sequences;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import classif.kmeans.KMeansSymbolicSequence;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class DTWPUKMeans extends Classifier {

	private static final long serialVersionUID = 1717176683182910935L;
	protected ArrayList<ClassedSequence> prototypes;
	protected Sequence[] clusteredUNL = null;
	protected ArrayList<Sequence> unlabeledData = new ArrayList<>();
	protected int nbClustersinUNL;
	private double df = 0.25;
	protected double[][] distances;
	private int nbClusters;
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
	@Override
	public void buildClassifier(Instances data) {
		trainingData = new Instances(data);
		prototypes = new ArrayList<>();
		Instance posData=init();
		for (int i = 0; i < trainingData.numInstances(); i++) {
			trainingData.instance(i).setClassValue(uClass);
		}
		sequences = new Sequence[trainingData.numInstances()];
		for (int i = 0; i < sequences.length; i++) {
			Instance sample = trainingData.instance(i);
			sequences[i] = InsToSeq(sample);
			unlabeledData.add(sequences[i]);
		}
		//find N
//		findN(pos, unlabeledData);
		
		// split data to P and U sets
		Sequence[] pos=new Sequence[5];
		pos[0] = InsToSeq(posData);
		for (int i = 1; i < pos.length; i++) {
			int unlpos = findUnlabeledNN(posData, unlabeledData);
			pos[i] = unlabeledData.get(unlpos);
			unlabeledData.remove(unlpos);
			ClassedSequence s = new ClassedSequence(pos[i], pClass);
			prototypes.add(s);
		}

		if (distances == null) {
			initDistances();
		}
		clusteredUNL = new Sequence[nbClustersinUNL];
		// if the class is empty, continue
		boolean isBig;
		ArrayList<Integer>[] affectation = new ArrayList[nbClusters];
		int runtime = 0;
		do {
			if (runtime > 10)
				nbClusters -= 1;
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

		findN(pos, clusteredUNL);
		// search positive from unlabeled
		searchPOSNEG(clusteredUNL,prototypes);
		
		
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
		double[] dist_pu= new double[cluster.length];
		for (int i = 0; i < cluster.length; i++) {
			dist_pu[i]=cluster[i].distance(Sequences.mean(pos));
		}
		for (int k = 0; k < dist_pu.length; k++) {
			if (dist_pu[k] > Utils.kthSmallestValue(dist_pu, dist_pu.length/2)) {
				ClassedSequence s = new ClassedSequence(cluster[k], uClass);
				prototypes.add(s);
			}
		}
	}
	
	/**
	 * Split Unlabeled data to P or N
	 * @param sequences Unlabeled Sequence[]
	 * @param prototypes Prototypes of P and N
	 */
	protected void searchPOSNEG(Sequence[] sequences,ArrayList<ClassedSequence> prototypes) {
		ArrayList<ClassedSequence> pn=new ArrayList<>(prototypes);
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
	
	protected void SearchPOSinUNL(Sequence[] clusteredunl, Sequence[] pos) {
		int nbpos = (int) (nbClustersinUNL * df);
		int[] posunl = new int[nbpos];
		double[] distpu = new double[nbClustersinUNL];
		for (int i = 0; i < clusteredunl.length; i++) {
			double dist = Double.POSITIVE_INFINITY;
			for (int j = 0; j < pos.length; j++) {
				double d = clusteredunl[i].distance(pos[j]);
				if (d < dist) {
					dist = d;
				}
				distpu[i] = dist;
			}
		}

		// get position of pos in unl
		int flg = 0;
		while (flg < nbpos) {
			int position = Utils.minIndex(distpu);
			posunl[flg] = position;
			distpu[position] = Double.POSITIVE_INFINITY;
			flg++;
		}

		// add pos and unl into prototypes
		Arrays.sort(posunl);
		for (int i = 0; i < clusteredunl.length; i++) {
			if (Arrays.binarySearch(posunl, i) >= 0) {
				ClassedSequence s = new ClassedSequence(clusteredunl[i], pClass);
//				prototypes.add(s);
			} else {
				ClassedSequence s = new ClassedSequence(clusteredunl[i], uClass);
				prototypes.add(s);
			}
		}
	}
	
	/**
	 * Find a reliable P from U with a positive example
	 * @param p
	 * @param u
	 * @return a reliable P
	 */
	private int findUnlabeledNN(Instance p, ArrayList<Sequence> u) {
		int unlpos = -1;
		double dist = Double.POSITIVE_INFINITY;
		for (int i = 0; i < u.size(); i++) {
			Sequence[] splitsequences = new Sequence[2];
			splitsequences[0]  = u.get(i);
			splitsequences[1] = InsToSeq(p);
			double d = splitsequences[1].distance(splitsequences[0]);
			if (d < dist) {
				dist = d;
				unlpos = i;
			}
		}
		return unlpos;
	}
	/**
	 * random select one positive example
	 * @return positive example
	 */
	protected Instance init() {
		Instance posData = new Instance(trainingData.numAttributes());
		posData.setDataset(trainingData);
		while (posData.classValue() != trainingData.classAttribute().indexOfValue(pClass)) {
			int rd = new Random().nextInt(trainingData.numInstances());
			if (trainingData.instance(rd).classValue() == trainingData.classAttribute().indexOfValue(pClass)) {
				posData = trainingData.instance(rd);
				trainingData.delete(rd);
			}
		}
		ClassedSequence s = new ClassedSequence(InsToSeq(posData), pClass);
		prototypes.add(s);
		return posData;
	}
}
