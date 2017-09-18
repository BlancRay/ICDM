package classif.NNClassifier;

import items.ClassedSequence;
import items.Sequence;

import java.util.ArrayList;

import classif.Prototyper;
import weka.core.Instances;

public class DTWKNNClassifier extends Prototyper {
	private static final long serialVersionUID = 4784722751028659039L;

	public DTWKNNClassifier() {
		super();
	}

	@Override
	protected void buildSpecificClassifier(Instances data) {
		ArrayList<String> classes = new ArrayList<String>(classedData.keySet());
		for (String clas : classes) {
			// if the class is empty, continue
			if (classedData.get(clas).isEmpty())
				continue;
			for (int i = 0; i < classedData.get(clas).size(); i++) {
				ClassedSequence s = new ClassedSequence(classedData.get(clas).get(i), clas);
				prototypes.add(s);
			}
		}
	}
}
