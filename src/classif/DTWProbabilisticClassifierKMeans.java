package classif;

import items.ClassedSequence;
import items.MonoDoubleItemSet;
import items.Sequence;
import mdsj.MDSJ;

import java.util.ArrayList;
import java.util.HashMap;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;
import java.lang.Math;

import classif.kmeans.KMeansSymbolicSequence;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class DTWProbabilisticClassifierKMeans extends Classifier {

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

	private static final double sqrt2Pi = Math.sqrt(2 * Math.PI);

	public DTWProbabilisticClassifierKMeans() {
		super();
	}

	/**
	 * train from dataset
	 * 
	 * @param data
	 */
	@Override
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

		
		
		for (String clas : classes) {
			XYSeriesCollection xyseriescollection = new XYSeriesCollection(); //再用XYSeriesCollection添加入XYSeries 对象
			int c = trainingData.classAttribute().indexOfValue(clas);
			// System.out.println("clas "+clas+" -> "+c);
			// if the class is empty, continue
			if (classedData.get(clas).isEmpty())
				continue;
			KMeansSymbolicSequence kmeans = new KMeansSymbolicSequence(nClustersPerClass, classedData.get(clas));

			kmeans.cluster();
			for (int k = 0; k < kmeans.centers.length; k++) {

				if (kmeans.centers[k] != null) { // ~ if empty cluster

					ClassedSequence s = new ClassedSequence(kmeans.centers[k], clas);
					prototypes.add(s);
					// find the center
					centroidsPerClass[c][k] = kmeans.centers[k];
					int nObjectsInCluster = kmeans.affectation[k].size();

					//画图
					XYSeries xyseries = new XYSeries(clas+" "+k);
					double[][] dis=new double[kmeans.affectation[k].size()][kmeans.affectation[k].size()];
					for(int q=0;q<kmeans.affectation[k].size();q++){
						for(int w=0;w<kmeans.affectation[k].size();w++){
							
							dis[q][w]=kmeans.affectation[k].get(q).distanceEuc(kmeans.affectation[k].get(w));
						}
					}
					int n=dis[0].length;
					double[][] output=MDSJ.classicalScaling(dis); // apply MDS
					for(int i=0; i<n; i++) {  // output all coordinates
						if(Double.isNaN(output[0][i]))
							output[0][i]=0;
						if(Double.isNaN(output[1][i]))
							output[1][i]=0;
						xyseries.add(output[0][i],output[1][i]);
					}
					
					xyseriescollection.addSeries(xyseries);
					

					// compute sigma
					double sumOfSquares = kmeans.centers[k].sumOfSquares(kmeans.affectation[k]);
					sigmasPerClass[c][k] = Math.sqrt(sumOfSquares / (nObjectsInCluster + 1));
					// compute p(k)
					// the P(K) of k
					prior[c][k] = 1.0 * nObjectsInCluster / data.numInstances();
				}
			}
			JFreeChart jfChart=ChartFactory.createScatterPlot(null, null, null, xyseriescollection);
			try {
				ChartUtilities.saveChartAsPNG(new File("C:/Users/leix/workspace/ICDM2014/save/"+clas+".png"), jfChart, 1000, 1000);
			} catch (IOException e) {
				e.printStackTrace();
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
	@Override
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
				double dist = seq.distance(centroidsPerClass[c][k]);
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

	/*
	 * public double[] distributionForInstance(Instance sample) throws Exception
	 * { // transform instance to sequence MonoDoubleItemSet[] sequence = new
	 * MonoDoubleItemSet[sample.numAttributes() - 1]; int shift =
	 * (sample.classIndex() == 0) ? 1 : 0; for (int t = 0; t < sequence.length;
	 * t++) { sequence[t] = new MonoDoubleItemSet(sample.value(t + shift)); }
	 * Sequence seq = new Sequence(sequence);
	 * 
	 * double[] probabilities = new double[sample.numClasses()];
	 * 
	 * 
	 * // for each class for (String clas : classedData.keySet()) { int c =
	 * trainingData.classAttribute().indexOfValue(clas); // for each cluster of
	 * the class for (int k = 0; k < centroidsPerClass[c].length; k++) {
	 * 
	 * // compute P(Q|k_c) double p =
	 * computeProbaForQueryAndCluster(sigmasPerClass[c][k],
	 * seq.distance(centroidsPerClass[c][k])); probabilities[c] += p *
	 * prior[c][k]; } }
	 * 
	 * Utils.normalize(probabilities);
	 * 
	 * return probabilities; }
	 */
	private double computeProbaForQueryAndCluster(double sigma, double d) {
		double pqk;
		if ( Double.isNaN(sigma)) {
			System.err.println("alert");
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
}