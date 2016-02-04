package classif.gmm;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Random;

import classif.ExperimentsLauncher;
import classif.PrototyperEUC;
import classif.kmeans.EUCKMeansSymbolicSequence;
import items.ClassedSequence;
import items.MonoDoubleItemSet;
import items.Sequence;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class EUCKNNClassifierGmm extends Classifier{

	private static final long serialVersionUID = 1717176683182910935L;

	protected ArrayList<ClassedSequence> prototypes;
	protected Sequence[][] centroidsPerClass = null;
	protected double[][] sigmasPerClass = null;
	protected HashMap<String, ArrayList<Sequence>> classedData;
	HashMap<String, ArrayList<Integer>> indexClassedDataInFullData;
	protected int nClustersPerClass;
	public int[] clusterMap;
	Sequence[] sequences;
	String[] classMap;
	Instances trainingData = null;
	double[][] prior = null;
	double[] nck = null;

	private static final double sqrt2Pi = Math.sqrt(2 * Math.PI);
	private static final double threshold = Math.pow(10, -11);

	public EUCKNNClassifierGmm() {
		super();
	}

	/**
	 * train from dataset
	 * 
	 * @param data
	 */
	public void buildClassifier(Instances data) {

		trainingData = data;
		Attribute classAttribute = data.classAttribute();
		prototypes = new ArrayList<>();

		classedData = new HashMap<String, ArrayList<Sequence>>();
		indexClassedDataInFullData = new HashMap<String, ArrayList<Integer>>();
		for (int c = 0; c < data.numClasses(); c++) {
			classedData.put(data.classAttribute().value(c), new ArrayList<Sequence>());
			indexClassedDataInFullData.put(data.classAttribute().value(c), new ArrayList<Integer>());
		}

		sequences = new Sequence[data.numInstances()];
		classMap = new String[sequences.length];
		for (int i = 0; i < sequences.length; i++) {
			Instance sample = data.instance(i);
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
			int shift = (sample.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < sequence.length; t++) {
				sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
			}
			sequences[i] = new Sequence(sequence);
			String clas = sample.stringValue(classAttribute);
			classMap[i] = clas;
			classedData.get(clas).add(sequences[i]);
			indexClassedDataInFullData.get(clas).add(i);
			// System.out.println("Element "+i+" of train is classed "+clas+"
			// and went to element
			// "+(indexClassedDataInFullData.get(clas).size()-1));
		}

		ArrayList<String> classes = new ArrayList<String>(classedData.keySet());
		centroidsPerClass = new Sequence[classes.size()][nClustersPerClass];
		sigmasPerClass = new double[classes.size()][nClustersPerClass];
		prior = new double[classes.size()][nClustersPerClass];
		nck = new double[nClustersPerClass];
		int dataAttributes=data.numAttributes() - 1;

		for (String clas : classes) {
			int c = trainingData.classAttribute().indexOfValue(clas);

			// System.out.println("clas "+clas+" -> "+c);
			// if the class is empty, continue
			if (classedData.get(clas).isEmpty())
				continue;
			EUCGMMSymbolicSequence gmmclusterer = new EUCGMMSymbolicSequence(nClustersPerClass, classedData.get(clas),dataAttributes);
			gmmclusterer.cluster();

			centroidsPerClass[c] = gmmclusterer.getMus();
			sigmasPerClass[c] = gmmclusterer.getSigmas();

			for (int k = 0; k < centroidsPerClass[c].length; k++) {
				ClassedSequence s = new ClassedSequence(centroidsPerClass[c][k], clas);
				prototypes.add(s);
				prior[c][k] = gmmclusterer.nck[k] / data.numInstances();
				System.out.println("priors is "+prior[c][k]+" Gaussian #"+clas+":mu="+centroidsPerClass[c][k]+"\tsigma="+sigmasPerClass[c][k]);
			}
		}
	}

		/**
		 * test on one sample
		 * 
		 * @param sample
		 * @return p(y|sample) forall y
		 * @throws Exception
		 */
		public double classifyInstance(Instance sample) throws Exception {
			// transform instance to sequence
			MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
			int shift = (sample.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < sequence.length; t++) {
				sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
			}
			Sequence seq = new Sequence(sequence);

			// for each class
			String classValue = null;
			double maxProb = 0.0;
			for (String clas : classedData.keySet()) {
				int c = trainingData.classAttribute().indexOfValue(clas);
				double prob = 1.0;
				for (int k = 0; k < centroidsPerClass[c].length; k++) {
					// compute P(Q|k_c)
					double dist = seq.distanceEuc(centroidsPerClass[c][k]);
					double p = computeProbaForQueryAndCluster(sigmasPerClass[c][k], dist);
					prob += p*prior[c][k];
				}
				if (prob > maxProb) {
					maxProb = prob;
					classValue = clas;
				}
			}
			return sample.classAttribute().indexOfValue(classValue);
		}

		private double computeProbaForQueryAndCluster(double sigma, double d) {
			double pqk;
			if ( Double.isNaN(sigma)) {
//				System.err.println("alert");
				pqk = 0.0;
			}
			else
				pqk = Math.exp(-(d * d) / (2 * sigma * sigma)) / (sigma * sqrt2Pi);

			return pqk;
		}
		
		public int getNClustersPerClass() {
			return nClustersPerClass;
		}

		public void setNClustersPerClass(int nbPrototypesPerClass) {
			this.nClustersPerClass = nbPrototypesPerClass;
		}

		public Sequence[][] getCentroidsPerClass() {
			return centroidsPerClass;
		}

		public void setCentroidsPerClass(Sequence[][] centroidsPerClass) {
			this.centroidsPerClass = centroidsPerClass;
		}

		public ArrayList<ClassedSequence> getPrototypes() {
			return prototypes;
		}

		public void setPrototypes(ArrayList<ClassedSequence> prototypes) {
			this.prototypes = prototypes;
		}
	}
