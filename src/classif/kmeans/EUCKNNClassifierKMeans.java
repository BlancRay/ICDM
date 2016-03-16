package classif.kmeans;

import java.util.ArrayList;

import classif.PrototyperEUC;
import items.ClassedSequence;
import weka.core.Instances;

public class EUCKNNClassifierKMeans extends PrototyperEUC {

	private static final long serialVersionUID = 1717176683182910935L;
	
	public EUCKNNClassifierKMeans() {
		super();
	}

	@Override
	protected void buildSpecificClassifier(Instances data) {
		
		ArrayList<String> classes = new ArrayList<String>(classedData.keySet());
		
		for (String clas : classes) {
			// if the class is empty, continue
			if(classedData.get(clas).isEmpty()) 
				continue;
			EUCKMeansSymbolicSequence kmeans = new EUCKMeansSymbolicSequence(nbPrototypesPerClass, classedData.get(clas));
			kmeans.cluster();
			for (int i = 0; i < kmeans.centers.length; i++) {
				if(kmeans.centers[i]!=null){ //~ if empty cluster
					ClassedSequence s = new ClassedSequence(kmeans.centers[i], clas);
					prototypes.add(s);
					int nObjectsInCluster = kmeans.affectation[i].size();
					double prior = 1.0 * nObjectsInCluster / data.numInstances();
					double sumOfSquares = s.sequence.EUCsumOfSquares(kmeans.affectation[i]);
					double sigmasPerClass = Math.sqrt(sumOfSquares / (nObjectsInCluster - 1));
//					System.out.println(nObjectsInCluster+" priors is "+prior+" Gaussian "+clas+" #"+i+":mu="+s.sequence.toString()+"\t "+sigmasPerClass);
				}
			}
		}
	}
}
