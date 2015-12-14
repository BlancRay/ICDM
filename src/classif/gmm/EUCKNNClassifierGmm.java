package classif.gmm;

import items.ClassedSequence;
import items.MonoDoubleItemSet;
import items.Sequence;

import java.util.ArrayList;
import java.util.HashMap;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import classif.kmeans.EUCKMeansSymbolicSequence;
import classif.kmeans.KMeansSymbolicSequence;

public class EUCKNNClassifierGmm extends Classifier{

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
	double[][] prior = null;
	double[][] nck = null;

	private static final double sqrt2Pi = Math.sqrt(2 * Math.PI);

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
		nck = new double[classedData.keySet().size()][nClustersPerClass];
		double sumpk[]=new double[classedData.keySet().size()];
		
		double sumoflog=0;
		double Lprev=-(Math.exp(308));

		for (String clas : classes) {
			int c = trainingData.classAttribute().indexOfValue(clas);
			double[][] pik = new double[classedData.get(clas).size()][nClustersPerClass];

			// System.out.println("clas "+clas+" -> "+c);
			// if the class is empty, continue
			if (classedData.get(clas).isEmpty())
				continue;
			EUCKMeansSymbolicSequence kmeans = new EUCKMeansSymbolicSequence(nClustersPerClass, classedData.get(clas));

			kmeans.cluster();
			for (int k = 0; k < kmeans.centers.length; k++) {
				if (kmeans.centers[k] != null) { // ~ if empty cluster
					// find the center
					centroidsPerClass[c][k] = kmeans.centers[k];
					int nObjectsInCluster = kmeans.affectation[k].size();

					// compute sigma
					double sumOfSquares = kmeans.centers[k].EUCsumOfSquares(kmeans.affectation[k]);
					sigmasPerClass[c][k] = Math.sqrt(sumOfSquares / (nObjectsInCluster - 1));
//					System.out.println(sigmasPerClass[c][k]);
					// compute p(k)
					// the P(K) of k
					prior[c][k] = 1.0 * nObjectsInCluster / data.numInstances();
					nck[c][k] = nObjectsInCluster;
					sumpk[c] += nObjectsInCluster;
				}
			}

			//computing initial likelihood
			for (Sequence ss : classedData.get(clas)) {
				double prob = 0.0;
				for (int k = 0; k < centroidsPerClass[c].length; k++) {
					double dist = ss.distanceEuc(centroidsPerClass[c][k]);
					double p = computeProbaForQueryAndCluster(sigmasPerClass[c][k], dist);
					prob += p * prior[c][k];// xi点由k个聚类生成的概率
				}
				sumoflog += Math.log(prob);
			}

			double[][] gamma = new double[classedData.get(clas).size()][nClustersPerClass];
			while (Math.abs(sumoflog - Lprev) > Math.exp(-6)) {
				// System.out.println("sumoflog="+sumoflog);
				// p(i,k)
				Lprev = sumoflog;
				// get p(i,k)
				ArrayList<Sequence> sequencesForClass = classedData.get(clas);
				for (int i=0; i<sequencesForClass.size();i++) {
				    Sequence s = sequencesForClass.get(i);
					double[] sumofgamma = new double[classedData.get(clas).size()];
					// for each p(k)

					for (int k = 0; k < centroidsPerClass[c].length; k++) {
						double dist = s.distanceEuc(centroidsPerClass[c][k]);
						double p = computeProbaForQueryAndCluster(sigmasPerClass[c][k], dist);
						gamma[i][k] = p * nck[c][k] / sumpk[c];
					}

					// sum of p(k)
					for (int k = 0; k < gamma[i].length; k++) {
						sumofgamma[i] += gamma[i][k];
					}
					// p(i,k)
					for (int k = 0; k < centroidsPerClass[c].length; k++) {
						pik[i][k] = gamma[i][k] / sumofgamma[i];
					}
				}

				// Nk

				for (int k = 0; k < centroidsPerClass[c].length; k++) {
					double sumofpik = 0;
					for (int i = 0; i < classedData.get(clas).size(); i++) {
						sumofpik += pik[i][k];
					}
					nck[c][k] = sumofpik;
				}

				// centroidsPerClass
				MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[data.numAttributes() - 1];

				for (int k = 0; k < centroidsPerClass[c].length; k++) {
					Sequence px = null;
					MonoDoubleItemSet[] sequencetmp = new MonoDoubleItemSet[data.numAttributes() - 1];
					MonoDoubleItemSet[] sequencepx = new MonoDoubleItemSet[data.numAttributes() - 1];
					// new MonoDoubleItemSet
					for (int t = 0; t < sequence.length; t++) {
						sequencetmp[t] = new MonoDoubleItemSet(0);
						sequencepx[t] = new MonoDoubleItemSet(0);
					}

					for (int i = 0; i < classedData.get(clas).size(); i++) {
						sequence = (MonoDoubleItemSet[]) classedData.get(clas).get(i).getSequence();
						for (int t = 0; t < sequence.length; t++) {
							sequencetmp[t] = new MonoDoubleItemSet(
									pik[i][k] * sequence[t].getValue() / nck[c][k] + sequencepx[t].getValue());
						}
						sequencepx = sequencetmp;
					}
					px = new Sequence(sequencepx);
					centroidsPerClass[c][k] = px;
				}

				// sigma
				// sigmasPerClass = new
				// double[classes.size()][nClustersPerClass];
				for (int k = 0; k < centroidsPerClass[c].length; k++) {
					sigmasPerClass[c][k] = 0;
					for (int i = 0; i < classedData.get(clas).size(); i++) {
						sigmasPerClass[c][k] += pik[i][k]
								* classedData.get(clas).get(i).distanceEuc(centroidsPerClass[c][k]) / nck[c][k];
					}
				}
				sumoflog = 0;
				for (Sequence ss : classedData.get(clas)) {
					double prob = 0.0;
					for (int k = 0; k < centroidsPerClass[c].length; k++) {
						double dist = ss.distanceEuc(centroidsPerClass[c][k]);
						double p = computeProbaForQueryAndCluster(sigmasPerClass[c][k], dist);
						prob += p * nck[c][k] / sumpk[c];
					}
					sumoflog += Math.log(prob);
				}
			}

			for (int k = 0; k < centroidsPerClass[c].length; k++) {
				ClassedSequence s = new ClassedSequence(centroidsPerClass[c][k], clas);
				prototypes.add(s);
				prior[c][k] = nck[c][k] / data.numInstances();
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
				double prob = 0.0;
				for (int k = 0; k < centroidsPerClass[c].length; k++) {
					// compute P(Q|k_c)
					double dist = seq.distanceEuc(centroidsPerClass[c][k]);
					double p = computeProbaForQueryAndCluster(sigmasPerClass[c][k], dist);
					prob += p * prior[c][k];
					// System.out.println(probabilities[c]);
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
