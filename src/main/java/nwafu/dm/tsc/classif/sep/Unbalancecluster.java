package nwafu.dm.tsc.classif.sep;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import org.apache.commons.math3.random.RandomDataGenerator;

import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.ClassedSequence;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class Unbalancecluster extends AbstractClassifier{
	private static final long serialVersionUID = -8080623340811851235L;
	protected ArrayList<ClassedSequence> prototypes;
	protected HashMap<String, ArrayList<Sequence>> classedData;
	protected HashMap<String, ArrayList<Integer>> indexClassedDataInFullData;
	protected int nbPrototypesPerClass[];
	protected Sequence[] sequences;
	protected String[] classMap;
	protected Instances trainingData = null;
	protected boolean fillPrototypes = true;

	public Unbalancecluster() {
		super();
	}
	
	
	public void buildClassifier(Instances data) throws Exception {
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
//			System.out.println("Element "+i+" of train is classed "+clas+" and went to element "+(indexClassedDataInFullData.get(clas).size()-1));
		}
		buildSpecificClassifier(data);
	}

	
	
	HashMap<String, double[][]>distancesPerClass=null;
	
	protected void initDistances() {
		distancesPerClass = new HashMap<String, double[][]>();
		ArrayList<String> classes = new ArrayList<String>(classedData.keySet());
		
		for (String clas : classes) {
			// if the class is empty, continue
			ArrayList<Sequence> objectsInClass = classedData.get(clas);
			if(objectsInClass.isEmpty()) continue;
			int nObjectsInClass = objectsInClass.size();
			
			double[][]distances = new double[nObjectsInClass][nObjectsInClass];
			for(int i=0;i<nObjectsInClass;i++){
				for(int j=i+1;j<nObjectsInClass;j++){
					distances[i][j]=objectsInClass.get(i).distance(objectsInClass.get(j));
					distances[j][i]=distances[i][j];
				}
			}
			distancesPerClass.put(clas, distances);
		}
//		System.out.println("all distances cached");
	}

	protected void buildSpecificClassifier(Instances data) {
		if(distancesPerClass==null){
			initDistances();
		}
		
		ArrayList<String> classes = new ArrayList<String>(classedData.keySet());
		
		for (String clas : classes) {
			// if the class is empty, continue
			if(classedData.get(clas).isEmpty()) 
				continue;
			KMeansCachedSymbolicSequence kmeans = new KMeansCachedSymbolicSequence(nbPrototypesPerClass[trainingData.classAttribute().indexOfValue(clas)], classedData.get(clas),distancesPerClass.get(clas));
			kmeans.cluster();
			
			for (int i = 0; i < kmeans.centers.length; i++) {
				if(kmeans.centers[i]!=null){ //~ if empty cluster
					ClassedSequence s = new ClassedSequence(kmeans.centers[i], clas);
					prototypes.add(s);
				}
			}
		}
	}
	/**
	 * Set if you want to fill the prototypes
	 * 
	 * @param fillPrototypes
	 */
	public void setFillPrototypes(boolean fillPrototypes) {
		this.fillPrototypes = fillPrototypes;
	}

	/**
	 * Predict the accuracy of the prototypes based on the learning set. It uses
	 * cross validation to draw the prediction.
	 * 
	 * @param nbFolds
	 *            the number of folds for the x-validation
	 * @return the predicted accuracy
	 */
	public double predictAccuracyXVal(int nbFolds) throws Exception {
		Evaluation eval = new Evaluation(trainingData);
		eval.crossValidateModel(this, trainingData, nbFolds, new Random(), new Object[] {});
		return eval.errorRate();
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
	

	public int[] getNbPrototypesPerClass() {
		return nbPrototypesPerClass;
	}
	public void setNbPrototypesPerClass(int[] nbPrototypesPerClass) {
		this.nbPrototypesPerClass = nbPrototypesPerClass;
	}

	public int getActualNumberOfPrototypesSelected() {
		return prototypes.size();
	}


	public void setPrototypes(ArrayList<ClassedSequence> prototypes) {
		this.prototypes = prototypes;
	}


	public ArrayList<ClassedSequence> getPrototypes() {
		return prototypes;
	}
}
