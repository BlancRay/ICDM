package classif.Majority;

import java.util.ArrayList;
import java.util.HashMap;

import com.sun.org.apache.xalan.internal.xsltc.compiler.sym;

import items.ClassedSequence;
import items.MonoDoubleItemSet;
import items.Sequence;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class KMeans extends AbstractClassifier{
	private static final long serialVersionUID = 2198495636176008932L;
	protected ArrayList<ClassedSequence> prototypes;
	protected ArrayList<ClassedSequence> classedData;
	protected HashMap<String, ArrayList<Integer>> indexClassedDataInFullData;
	protected int nbPrototypes;
	protected Sequence[] sequences;
	protected String[] classMap;
	protected Instances trainingData = null;
	protected boolean fillPrototypes = true;

	public KMeans() {
		super();
	}

	public void buildClassifier(Instances data) throws Exception {
		trainingData = data;
		classedData= new ArrayList<ClassedSequence>();
		prototypes= new ArrayList<ClassedSequence>();
		for (int i = 0; i < trainingData.numInstances(); i++) {
			Instance sample = data.instance(i);
			MonoDoubleItemSet[] Monosequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
			int shift = (sample.classIndex() == 0) ? 1 : 0;
			for (int t = 0; t < Monosequence.length; t++) {
				Monosequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
			}
			Sequence sequence = new Sequence(Monosequence);
			ClassedSequence s = new ClassedSequence(sequence, Double.toString(sample.classValue()));
			classedData.add(s);
		}

		KMeansSymbolicSequence kmeans = new KMeansSymbolicSequence(nbPrototypes, classedData,trainingData.classAttribute());
		kmeans.cluster();
		double purity=purity(kmeans.affectation);
		System.out.println("purity:"+purity);
		for (int i = 0; i < kmeans.centers.length; i++) {
			if(kmeans.centers[i]!=null){ //~ if empty cluster
				prototypes.add(kmeans.centers[i]);
			}
		}
	}
	
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
		// System.out.println(prototypes.size());
		return sample.classAttribute().indexOfValue(classValue);
	}

	protected double purity(ArrayList<ClassedSequence>[] affectation) {
		double purity = 0.0;
		double tureaffectation = 0.0;
		for (int i = 0; i < affectation.length; i++) {
			int[] labels = new int[trainingData.classAttribute().numValues()];
			for (ClassedSequence s : affectation[i]) {
				String classValue = s.classValue;
				labels[(int) Double.parseDouble(classValue)]++;
			}
			
			int majority = 0;
			int max = 0;
			for (int j = 0; j < labels.length; j++) {
				if (labels[j] > max)
					majority = j;
			}
			tureaffectation += labels[majority];
		}
		purity = tureaffectation / trainingData.numInstances();
		return purity;
	}
	
	public int getNbPrototypes() {
		return nbPrototypes;
	}

	public void setNbPrototypes(int nbPrototypes) {
		this.nbPrototypes = nbPrototypes;
	}
}
