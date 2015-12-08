package classif.kmeans;

import items.ClassedSequence;

import java.util.ArrayList;

import classif.Prototyper;
import weka.core.Instances;

public class EUCKNNClassifierKMeans extends Prototyper {

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
				}
			}
		}
	}
}
