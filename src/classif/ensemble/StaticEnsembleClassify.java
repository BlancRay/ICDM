package classif.ensemble;

import classif.BIGDT.ClassifyBigDT;
import classif.fastkmeans.DTWKNNClassifierKMeansCached;
import classif.kmeans.DTWKNNClassifierKMeans;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class StaticEnsembleClassify extends AbstractClassifier{
	private static final long serialVersionUID = -2848829508586998659L;
	public StaticEnsembleClassify() {
		super();
	}
	DTWKNNClassifierKMeansCached fkm_4 = new DTWKNNClassifierKMeansCached();
	DTWKNNClassifierKMeansCached fkm_10 = new DTWKNNClassifierKMeansCached();
//	DTWKNNClassifierKMeans KMeans = new DTWKNNClassifierKMeans();
	ClassifyBigDT dt = new ClassifyBigDT();
	double Weight_fkm_4=0.0;
	double Weight_fkm_10=0.0;
	double Weight_dt=0.0;
//	double Weight_kmeans=0.0;
	public void buildClassifier(Instances data) throws Exception {
		fkm_4.setNbPrototypesPerClass(4);
		fkm_4.setFillPrototypes(true);
		fkm_4.buildClassifier(data);
		System.out.println("Fast K-Means with 4 Prototypes build finished");
		
//		KMeans.setNbPrototypesPerClass(10);
//		KMeans.setFillPrototypes(true);
//		KMeans.buildClassifier(data);
//		System.out.println("K-Means with 10 Prototypes build finished");
		
		fkm_10.setNbPrototypesPerClass(10);
		fkm_10.setFillPrototypes(true);
		fkm_10.buildClassifier(data);
		System.out.println("Fast K-Means with 10 Prototypes build finished");
		
		dt.buildClassifier(data);
		System.out.println("Decision Tree build finished");
	}
	
	public double classifyInstance(Instance sample) throws Exception {
		int[] classlabel = new int[sample.numClasses()];
		classlabel[(int) fkm_4.classifyInstance(sample)] +=(int) (Weight_fkm_4*100);
		classlabel[(int) fkm_10.classifyInstance(sample)] +=(int) (Weight_fkm_10*100);
		classlabel[(int) dt.classifyInstance(sample)] +=(int) (Weight_dt*100);
//		classlabel[(int) KMeans.classifyInstance(sample)] +=(int) (Weight_kmeans*100);
//		System.out.println(Arrays.toString(classlabel));
		return Utils.maxIndex(classlabel);
	}

	public DTWKNNClassifierKMeansCached getFkm_4() {
		return fkm_4;
	}

	public DTWKNNClassifierKMeansCached getFkm_10() {
		return fkm_10;
	}
	
	public ClassifyBigDT getDt() {
		return dt;
	}

	public void setWeight_fkm_4(double weight_fkm_4) {
		Weight_fkm_4 = 1-weight_fkm_4;
	}

	public void setWeight_fkm_10(double weight_fkm_10) {
		Weight_fkm_10 = 1-weight_fkm_10;
	}

	public void setWeight_dt(double weight_dt) {
		Weight_dt = 1-weight_dt;
	}

//	public DTWKNNClassifierKMeans getKMeans() {
//		return KMeans;
//	}
//
//	public void setWeight_kmeans(double weight_kmeans) {
//		Weight_kmeans = weight_kmeans;
//	}

}
