/**
 *    Date : 2008-1-9
 *      by : Yang ZHANG  @ UQ
 *   Email : mokuram@itee.uq.edu.au
 *           zhangyang@nwsuaf.edu.cn
 *    
 * Project : Classification of Data Streams
 *
 * Description:
 *     Building decision tree from positive and unlabeled dataset.
 *     Section 4.3
 *     of the paper: (IN PAGE 10)
 *     
 *     @inproceedings{
 *          author = {Francois Denisa, Remi Gilleronb, Fabien Letouzey},
 *    		title = {Learning from positive and unlabeled examples},
 *    		booktitle = {Theoretical Computer Science 348 },
 *    		year = {2005},
 *    		pages = {70 锟紺 83},
 *    		publisher = {ELSEVIER},
 *    	}
 *    
 *    This implement is only suitable for binary classification task.
 *    @attribute 'class' {'won','nowin'}
 *        the first class is the POS class, here, POS=won
 *        and the second class is the NEG class, NEG=nowin
 *        
 */

package classif.POSC45;

import weka.core.*;
import weka.classifiers.*;
import java.util.*;

import org.apache.commons.math3.random.RandomDataGenerator;

public class POSC45 extends AbstractClassifier{
	private static final long serialVersionUID = 2214198401582342899L;
	public int nbpos;
	private Instances Traindata;
	public double r=0.3;

	// the classifier
	public ClassifyPOSC45 claC45posunl;
	ClassifyPOSC45 c45posunl[];

	// training
	public void buildClassifier(Instances data) throws java.lang.Exception {
		Traindata=new Instances(data);
		System.out.println("r="+r);
		nbpos=(int) (nbpos*r);
		Instances posData = new Instances(Traindata, nbpos);
		Instances unlData = new Instances(Traindata, Traindata.numInstances() - nbpos);
		Enumeration enu = Traindata.enumerateInstances();
		int flg = 0;
		while (enu.hasMoreElements()) {
			Instance instance = (Instance) enu.nextElement();
			if (instance.classValue() == 0 && flg < nbpos) {
				posData.add(instance);
				flg++;
			} else {
				instance.setClassValue("-1.0");
				unlData.add(instance);
			}
		}

		// split the POS dataset
			Instances two[] = splitdata(posData);
			Instances posTrainData = two[0];
			Instances posTestData = two[1];
			// split the UN dataset
			two = splitdata(unlData);
			Instances unlTrainData = two[0];
			Instances unlTestData = two[1];

		c45posunl = new ClassifyPOSC45[9];
		for (int i = 0; i < 9; i++) {
			c45posunl[i] = new ClassifyPOSC45((i + 1) / 10.0);
			c45posunl[i].setDataset(posTrainData, unlTrainData);
			c45posunl[i].buildClassifier(null);
			System.out.println(c45posunl[i].toString());
		}

		// select best DF
		double dEstimate[] = new double[9];
		for (int i = 0; i < 9; i++) {
			dEstimate[i] = evaluateBaseEstimate(c45posunl[i], posTestData, unlTestData);
			System.out.println(dEstimate[i]);
		}

		int nBestIndex = Utils.minIndex(dEstimate);
		// train the final classifier
		claC45posunl = new ClassifyPOSC45((nBestIndex + 1) / 10.0);
		claC45posunl.setDataset(posData, unlData);
		claC45posunl.buildClassifier(null);
	}

	// estimate the performance of base classifier
	double evaluateBaseEstimate(ClassifyPOSC45 c45posunl, Instances posTestData, Instances unTestData) throws Exception {
		double dPre;
		int nPosError = 0;
		int nUn = 0;

		// evaluate on POS dataset
		for (int i = 0; i < posTestData.numInstances(); i++) {
			Instance sample = posTestData.instance(i);
			// Out.print(sample.toString() + "\t");
			dPre = c45posunl.classifyInstance(sample);
			// Out.println(dPre);
			if (Utils.eq(dPre, 1))
				nPosError++;
		}

		// evaluate on UN dataset
		for (int i = 0; i < unTestData.numInstances(); i++) {
			Instance sample = unTestData.instance(i);
			// Out.print(sample.toString() + "\t");
			dPre = c45posunl.classifyInstance(sample);
			// Out.println(dPre);
			if (Utils.eq(dPre, 0))
				nUn++;

		}


		System.out.println(nPosError + ";" + posTestData.numInstances() + ";" + nUn + ";" + unTestData.numInstances());

		dPre = 2 * (double) nPosError / posTestData.numInstances() + (double) nUn / unTestData.numInstances();
		return dPre;
	}

	// classify
	public double classifyInstance(Instance instance) throws Exception {
		return claC45posunl.classifyInstance(instance);
	}

	public void printTree() {
		System.out.println(claC45posunl.toString());
	}
	private Instances[] splitdata(Instances data) {
		Instances[] subsets = new Instances[2];
		subsets[0] = new Instances(data, 0);
		subsets[1] = new Instances(data, 0);
		RandomDataGenerator randGen = new RandomDataGenerator();
		int[] classselected = randGen.nextPermutation(data.numInstances(), data.numInstances() * 2 / 3);
		Arrays.sort(classselected);
		for (int i = 0; i < data.numInstances(); i++) {
			if (Arrays.binarySearch(classselected, i) >= 0)
				subsets[0].add(data.instance(i));
			else
				subsets[1].add(data.instance(i));
		}
		return subsets;
	}

}
