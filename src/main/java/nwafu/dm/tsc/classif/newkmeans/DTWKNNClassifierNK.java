package nwafu.dm.tsc.classif.newkmeans;

import java.util.ArrayList;
import java.util.HashMap;

import nwafu.dm.tsc.items.ClassedSequence;
import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.Sequence;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class DTWKNNClassifierNK extends AbstractClassifier{

	private static final long serialVersionUID = 3743484062993569254L;
	protected ArrayList<ClassedSequence> prototypes;
	protected Sequence[][] centroidsPerClass = null;
	protected double[][] sigmasPerClass = null;
	protected HashMap<String, ArrayList<Sequence>> classedData;
	HashMap<String, ArrayList<Integer>> indexClassedDataInFullData;
	protected int nClustersPerClass;
	Sequence[] sequences;
	String[] classMap;
	Instances trainingData = null;
	protected double[][] prior = null;
	protected double[] nck = null;
	protected HashMap<Integer, double[][]> gamma = null;
	protected HashMap<Integer, double[][]> dist = null;
	protected double[][] sumweightk = null;

	private static final double sqrt2Pi = Math.sqrt(2 * Math.PI);

	public DTWKNNClassifierNK() {
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
		prototypes = new ArrayList<ClassedSequence>();

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
		gamma=new HashMap<Integer, double[][]>(data.numClasses());
		dist=new HashMap<Integer, double[][]>(data.numClasses());
		sumweightk=new double[classes.size()][nClustersPerClass];
		
		prior = new double[classes.size()][nClustersPerClass];
		nck = new double[nClustersPerClass];
		int dataAttributes=data.numAttributes() - 1;

		for (String clas : classes) {
			int c = trainingData.classAttribute().indexOfValue(clas);

			DTWNKSymbolicSequence nkclusterer = new DTWNKSymbolicSequence(nClustersPerClass, classedData.get(clas),dataAttributes);
			nkclusterer.cluster();

			centroidsPerClass[c] = nkclusterer.getMus();
			sigmasPerClass[c] = nkclusterer.getSigmas();
			gamma.put(c, nkclusterer.getGamma());
			dist.put(c, nkclusterer.getDist());
			sumweightk[c] = nkclusterer.getSumofgammak();
			
			for (int k = 0; k < centroidsPerClass[c].length; k++) {
				if (sigmasPerClass[c][k] == Double.NaN)
					continue;
				ClassedSequence s = new ClassedSequence(centroidsPerClass[c][k], clas);
				prototypes.add(s);
				prior[c][k] = nkclusterer.getNck()[k] / data.numInstances();
//				System.out.println(sumweightk[c][k]+"\t"+nkclusterer.getNck()[k]);
//				System.out.println(nkclusterer.getNck()[k]+" objects,priors is "+prior[c][k]+" Gaussian "+clas+" #"+k+":mu="+centroidsPerClass[c][k]+"\tsigma="+sigmasPerClass[c][k]);
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
		double maxProb = 1.0;
		for (String clas : classedData.keySet()) {
			int c = trainingData.classAttribute().indexOfValue(clas);
			double prob = 0.0;
			for (int k = 0; k < centroidsPerClass[c].length; k++) {
				if (sigmasPerClass[c][k] == Double.NaN || sigmasPerClass[c][k] == 0) {
					System.err.println("sigma=NAN||sigma=0");
					continue;
				}
				double distance = seq.distance(centroidsPerClass[c][k]);
				double p = computeProbaForQueryAndCluster(gamma.get(c), dist.get(c), distance, k, clas);
				// System.out.println(p);
				prob += p / centroidsPerClass[c].length;
				// if (p <= maxProb) {
				// maxProb = p;
				// classValue = clas;
			}
			if (prob <= maxProb) {
				maxProb = prob;
				classValue = clas;
			}
		}
		return sample.classAttribute().indexOfValue(classValue);
	}
	
	private double computeProbaForQueryAndCluster(double[][] gamma, double[][] d, double dist, int k,String clas) {
		double sumweight =0;
		for (int i = 0; i <classedData.get(clas).size() ; i++) {
			if(d[i][k]<=dist){
				sumweight+=gamma[i][k];
			}
		}
		return (sumweight/sumweightk[trainingData.classAttribute().indexOfValue(clas)][k]);
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
