package classif.pukmeans;

import items.MonoDoubleItemSet;
import items.Sequence;

import java.util.Arrays;
import java.util.Random;

import classif.fastkmeans.DTWKNNClassifierKMeansCached;
import classif.gmm.DTWKNNClassifierGmm;
import weka.classifiers.AbstractClassifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class PUtoPN extends AbstractClassifier {

	private static final long serialVersionUID = 6150443653739190526L;
	private static final String pClass = "1.0";
	private static final String uClass = "-1.0";
	Instances trainingData = null;
	DTWKNNClassifierKMeansCached[] kMeansCached=new DTWKNNClassifierKMeansCached[1];

	public PUtoPN() {
		super();
	}

	/**
	 * train from dataset
	 * 
	 * @param data
	 * @throws Exception 
	 */
	public void buildClassifier(Instances data) throws Exception {
		trainingData = new Instances(CovPU2PN(data, 0.25));
		int i=5;
		for (int j = 0; j < kMeansCached.length; j++) {
			kMeansCached[j]=new DTWKNNClassifierKMeansCached();
			kMeansCached[j].setNbPrototypesPerClass(i);
			kMeansCached[j].buildClassifier(trainingData);
			System.out.println(i/5+" build finish");
			i+=5;
		}
	}
	
	public double classifyInstance(Instance sample) throws Exception {
		int[] classlabel = new int[sample.numClasses()];
		for (int i = 0; i < kMeansCached.length; i++) {
//			DTWKNNClassifierGmm classifierGmm = gmm[i];
//			classifierGmm.classifyInstance(sample);
			classlabel[(int) kMeansCached[i].classifyInstance(sample)]++;
		}
		System.out.println(Arrays.toString(classlabel));
		return Utils.maxIndex(classlabel);
  	}
	
	/**
	 * 
	 * @param data P and U
	 * @param ratio percent of P in dataset
	 * @return Instances with P and N
	 */
	private Instances CovPU2PN(Instances data, double ratio) {
		Instances posIns=new Instances(data,0);
		posIns.add(init(data));
		Instances samples = new Instances(data);
		data=null;
		for (int i = 0; i < samples.numInstances(); i++) {
			samples.instance(i).setClassValue(uClass);
		}
		
		double[]dist=new double[samples.numInstances()];
		int flg=0;
		while(samples.numInstances()!=0){
			double[] mindist = findUnlabeledNN(posIns, samples);
			int unlpos = Utils.minIndex(mindist);
			double now_dist = mindist[unlpos];
			dist[flg]=now_dist;
			samples.instance(unlpos).setClassValue(pClass);
			posIns.add(samples.instance(unlpos));
			samples.delete(unlpos);
			flg++;
		}
		double[] scc = SCC(dist, posIns.numInstances() - 1);
		int index = Utils.maxIndex(scc);
		System.out.println(Utils.maxIndex(scc)+"\t"+Utils.minIndex(scc));
		for (int i = index; i < posIns.numInstances(); i++) {
			posIns.instance(i).setClassValue(uClass);
		}
		return posIns;
	}

	/**
	 * random select one positive example
	 * 
	 * @return positive example
	 */
	protected Instance init(Instances data) {
		Instance posSample = new DenseInstance(data.numAttributes());
		posSample.setDataset(data);
		while (posSample.classValue() != data.classAttribute().indexOfValue(pClass)) {
			int rd = new Random().nextInt(data.numInstances());
			if (data.instance(rd).classValue() == data.classAttribute().indexOfValue(pClass)) {
				posSample = data.instance(rd);
				data.delete(rd);
			}
		}
		// ClassedSequence s = new ClassedSequence(InsToSeq(posSample), pClass);
		// prototypes.add(s);
		return posSample;
	}

	/**
	 * Find a reliable P from U with a positive example
	 * 
	 * @param p
	 * @param u
	 * @return a reliable P
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

	/**
	 * Convert Instance to Sequence
	 * 
	 * @param sample
	 *            Instance to be convert
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
		double[]scc=new double[mindist.length-1];
		int scci=0;
		for (int i = 1; i < mindist.length; i++) {
			double abs=Math.abs(mindist[i]-mindist[i-1]);
			double std=Math.sqrt(Utils.variance(Arrays.copyOfRange(mindist, 0, i+1)));
			scc[scci]=(abs/std*(1.0*(nbUNL-(i-1))/nbUNL));
			scci++;
		}
		return scc;
	}
}
