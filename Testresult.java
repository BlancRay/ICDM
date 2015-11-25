package classif.kmeans;

import items.MonoDoubleItemSet;
import items.Sequence;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import java.lang.Math;
import java.util.ArrayList;
import java.util.HashMap;

public class Testresult {
	public DTWKNNClassifierKMeans classifier;
	public Instances test;
	public double[][] dt;
	public int nbClasses;
	public int nbInstances;
	protected Sequence[] sequences;
	protected String[] classMap;
	protected HashMap<String, ArrayList<Sequence>> classedData;
	protected HashMap<String, ArrayList<Integer>> indexClassedDataInFullData;

	public Testresult(DTWKNNClassifierKMeans classifier, Instances data, int nbClasses, int nbInstances) {
		this.classifier = classifier;
		this.test = data;
		this.dt = classifier.dtw;
		this.nbClasses = nbClasses;
		this.nbInstances = nbInstances;
	}

	public void ini() {
		Attribute classAttribute = test.classAttribute();
		classedData = new HashMap<String, ArrayList<Sequence>>();
		indexClassedDataInFullData = new HashMap<String, ArrayList<Integer>>();
		for (int c = 0; c < test.numClasses(); c++) {
			classedData.put(test.classAttribute().value(c), new ArrayList<Sequence>());
			indexClassedDataInFullData.put(test.classAttribute().value(c), new ArrayList<Integer>());
		}
		sequences = new Sequence[test.numInstances()];
		classMap = new String[sequences.length];
		for (int i = 0; i < sequences.length; i++) {
			Instance sample = test.instance(i);
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
		}
	}

	public void PQKs() {
		ini();
		Sequence s;
		int[] classnumber = new int[sequences.length];
		double testRate = 0;
		for (int i = 0; i < sequences.length; i++) {
			s = sequences[i];
			classnumber[i] = PQK(s);
		}
		testRate = error(classnumber, classMap);
		System.out.println(testRate);
	}

	public double error(int[] a, String[] b) {
		ArrayList<String> classes = new ArrayList<String>(classedData.keySet());
		double errorRate = 0;
		int right = 0;
		test.classAttribute();

		for (int i = 0; i < a.length; i++) {
			if (classes.get(a[i]) == b[i])
				right += 1;
		}
		errorRate = 1 - 1.0 * right / a.length;
		return errorRate;
	}

	public int PQK(Sequence s) {
		double[][] pqkprecluster = new double[nbClasses][classifier.getNbPrototypesPerClass()];// P(Q|K)
		int[][] sk;// number of clusters in cluster k   聚类K的个数
		sk = classifier.sk;
		double[][] pkprecluster = new double[nbClasses][classifier.getNbPrototypesPerClass()];// the probability of k, P(K)=SK/Total  属于聚类K的概率
		double dtwsk;// DTW(S,k) the DTW between S and cluster k's centroid    S与第k个cluster中心的DTW距离
		double[][] pkqprecluster = new double[nbClasses][classifier.getNbPrototypesPerClass()];// P(K|Q)
		int clas = 0;// the label of S after classified by classifier     S属于概率最大的类
		double[] maxpreclass = new double[nbClasses];

		// for each class
		for (int flg = 0; flg < nbClasses; flg++) {
			
			// for each cluster
			for (int i = 0; i < sk[flg].length; i++) {
				// P(K|Q)
				int x = 0;
				x = sk[flg][i];
				pkprecluster[flg][i] = 1.0*x / nbInstances;// the P(K) of k in flg label    flg类的P(K)
				dtwsk = s.distance(classifier.ck[flg][i]);// s与flg类第i个聚类中心的DTW距离
				if(x==1)
				{
					pqkprecluster[flg][i]=0.0;}
				else
				{
				pqkprecluster[flg][i] = Math.pow(Math.E, (-(dtwsk) / (2 * dt[flg][i] * dt[flg][i])))
						/ (dt[flg][i] * Math.sqrt(2 * Math.PI));}
				pkqprecluster[flg][i] = pkprecluster[flg][i] * pqkprecluster[flg][i];
			}
			// for each class return the number of max probability of cluster

			// maxpreclass[flg] = Max(pkqprecluster[flg]);
			maxpreclass[flg] = sum(pkqprecluster[flg]);
		}
		// return number of max probability of class
		clas = Maxclass(maxpreclass);
		return clas;
	}

	public double sum(double[] pyq) {
		double sum = 0;
		for (int i = 0; i < pyq.length; i++) {
			sum = pyq[i] + sum;
		}
		return sum;
	}

	public double Max(double[] a) {
		double max = 0;
		for (int i = 0; i < a.length; i++) {
			if (max < a[i]) {
				max = a[i];
			}
		}
		return max;
	}

	public int Maxclass(double[] a) {
		double max = 0;
		int tmp = 0;
		for (int i = 0; i < a.length; i++) {
			if (max < a[i]) {
				max = a[i];
				tmp = i;
			}
		}
		return tmp;
	}
}
