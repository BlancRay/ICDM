package classif.gmm;

import java.util.ArrayList;
import java.util.HashMap;

import items.ClassedSequence;
import items.MonoDoubleItemSet;
import items.Sequence;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class DTWKNNClassifierGmm extends Classifier{

	private static final long serialVersionUID = 1717176683182910935L;

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

	private static final double sqrt2Pi = Math.sqrt(2 * Math.PI);

	public DTWKNNClassifierGmm() {
		super();
	}

	/**
	 * train from dataset
	 * 
	 * @param data
	 */
	public void buildClassifier(Instances data) {

		trainingData = new Instances(data);
		Attribute classAttribute = trainingData.classAttribute();
		prototypes = new ArrayList<>();

		classedData = new HashMap<String, ArrayList<Sequence>>();
		indexClassedDataInFullData = new HashMap<String, ArrayList<Integer>>();
		for (int c = 0; c < trainingData.numClasses(); c++) {
			classedData.put(trainingData.classAttribute().value(c), new ArrayList<Sequence>());
			indexClassedDataInFullData.put(trainingData.classAttribute().value(c), new ArrayList<Integer>());
		}

		sequences = new Sequence[trainingData.numInstances()];
		classMap = new String[sequences.length];
		for (int i = 0; i < sequences.length; i++) {
			Instance sample = trainingData.instance(i);
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
		int dataAttributes=trainingData.numAttributes() - 1;

		for (String clas : classes) {
			int c = trainingData.classAttribute().indexOfValue(clas);

			DTWGMMSymbolicSequence gmmclusterer = new DTWGMMSymbolicSequence(nClustersPerClass, classedData.get(clas),dataAttributes);
			gmmclusterer.cluster();

			centroidsPerClass[c] = gmmclusterer.getMus();
			sigmasPerClass[c] = gmmclusterer.getSigmas();
			
			for (int k = 0; k < centroidsPerClass[c].length; k++) {
				if (sigmasPerClass[c][k] == Double.NaN)
					continue;
				ClassedSequence s = new ClassedSequence(centroidsPerClass[c][k], clas);
				prototypes.add(s);
				prior[c][k] = gmmclusterer.getNck()[k] / trainingData.numInstances();
				System.out.println(gmmclusterer.getNck()[k]+" objects,priors is "+prior[c][k]+" Gaussian "+clas+" #"+k+":mu="+centroidsPerClass[c][k]+"\tsigma="+sigmasPerClass[c][k]);
			}
		}
/*		for (int i = 0; i < centroidsPerClass.length; i++) {
			for (int j = 0; j < centroidsPerClass.length; j++) {
				if(i==j)
					continue;
				for (int i2 = 0; i2 < centroidsPerClass[i].length; i2++) {
					for (int j2 = 0; j2 < centroidsPerClass[j].length; j2++) {
						double d=centroidsPerClass[i][i2].distance(centroidsPerClass[j][j2]);
						System.out.println(trainingData.classAttribute().value(i)+i2+" "+sigmasPerClass[i][i2]+" "+trainingData.classAttribute().value(j)+j2+" "+sigmasPerClass[j][j2]+" "+d);
						
					}
					
				}
				
			}
		}*/
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
//		double[] pr= new double[classedData.keySet().size()];
		for (String clas : classedData.keySet()) {
			int c = trainingData.classAttribute().indexOfValue(clas);
//			double prob = 0.0;
			for (int k = 0; k < centroidsPerClass[c].length; k++) {
				// compute P(Q|k_c)
				if (sigmasPerClass[c][k] == Double.NaN||sigmasPerClass[c][k] ==0){
					System.out.println("sigma=NAN||sigma=0");
					continue;
				}
				double dist = seq.distance(centroidsPerClass[c][k]);
				System.out.println(sigmasPerClass[c][k]+"\t"+dist);
				double p = computeProbaForQueryAndCluster(sigmasPerClass[c][k], dist);
				System.out.println(p);
//				prob += p/centroidsPerClass[c].length;
//				prob += p*prior[c][k];
				if (p > maxProb) {
					maxProb = p;
					classValue = clas;
//				}
			}
//			if (prob > maxProb) {
//				maxProb = prob;
//				classValue = clas;
			}
		}
//		System.out.println(Arrays.toString(pr));
		System.out.println(classValue);
		return sample.classAttribute().indexOfValue(classValue);
	}
	
	private double computeProbaForQueryAndCluster(double sigma, double d) {
		double pqk = 0.0;
		if (sigma==0) {
			if (d == 0) {
				pqk = 1;
			} else
				pqk = 0;
		} else
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
