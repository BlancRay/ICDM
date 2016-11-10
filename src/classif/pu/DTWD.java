package classif.pu;

import weka.core.*;

import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

import org.apache.commons.math3.random.RandomDataGenerator;

import com.mysql.jdbc.Util;

import items.MonoDoubleItemSet;
import items.Sequence;
import weka.classifiers.*;

public class DTWD extends Classifier {
	private static final long serialVersionUID = -8055689900166489949L;

	// the classifier
	private ClassifyPOSC45 claC45posunl;
	private ClassifyPOSC45 c45posunl[];
	private RandomDataGenerator randGen;
	protected double ratio=0.5;

	// training
	public void buildClassifier(Instances data) throws Exception{
		Instances traindata=new Instances(data);
		// split data to P and U sets
		Instances posData = new Instances(traindata, 0);
		Instances unlData = new Instances(traindata, 0);
		Instances negData = new Instances(traindata, 0);

		while (posData.numInstances() < 1) {
			int rd = new Random().nextInt(traindata.numInstances());
			if (traindata.instance(rd).classValue() == 0) {
				posData.add(traindata.instance(rd));
				data.delete(rd);
				traindata.delete(rd);
			}
		}
//		for (int i = 0; i < traindata.numInstances(); i++) {
//			traindata.instance(i).setClassValue(1.0);
//		}
		
		/**
		 * test of leaf node with 50% positive samples
		 */
//		double pre_dist = 0;
		PrintWriter out1 = new PrintWriter(new FileOutputStream("./pos"), true);
		PrintWriter out2 = new PrintWriter(new FileOutputStream("./posdist"), true);
		double[]dist=new double[traindata.numInstances()];
		int flg=0;
		while(traindata.numInstances()!=0){
			double[] mindist = findUnlabeledNN(posData, traindata);
			int unlpos = Utils.minIndex(mindist);
			double now_dist = mindist[unlpos];
			dist[flg]=now_dist;
			System.out.println(traindata.instance(unlpos));
			out1.println(traindata.instance(unlpos));
			out2.println(now_dist);
			System.out.println(now_dist);
//			traindata.instance(unlpos).setClassValue(0.0);
			posData.add(traindata.instance(unlpos));
//			data.delete(unlpos);
			traindata.delete(unlpos);
			flg++;
			
			/*if (i == 1)
				pre_dist = now_dist;
			if (Math.abs(now_dist - pre_dist) < 1) {
				traindata.instance(unlpos).setClassValue(0.0);
				posData.add(traindata.instance(unlpos));
				data.delete(unlpos);
				traindata.delete(unlpos);
				pre_dist = now_dist;
			} else {
				Instance tmp = posData.lastInstance();
				tmp.setClassValue(1.0);
				data.add(tmp);
				traindata.add(tmp);
				posData.delete(posData.numInstances() - 1);
				break;
			}*/
		}
//		System.out.println(posData.numInstances());
//		
//		for (int i = 0; i < data.numInstances(); i++) {
//			System.out.println(data.instance(i));
//		}
		out1.close();
		out2.close();
		double[] scc=SCC(dist, posData.numInstances()-1);
		int index=Utils.maxIndex(scc);
		int negindex=Utils.minIndex(scc);
		System.out.println("pos:\t"+Utils.maxIndex(scc)+"\t"+Utils.minIndex(scc));
		for (int i = posData.numInstances() - 1; i > index; i--) {
			if(i==negindex||i==negindex-1||i==index+1){
//				posData.instance(i).setClassValue("-1.0");
				negData.add(posData.instance(i));
			}
			else{
//			posData.instance(i).setClassValue(1.0);
			unlData.add(posData.instance(i));
			}
			posData.delete(i);
		}
		flg=0;
		PrintWriter out3 = new PrintWriter(new FileOutputStream("./neg"), true);
		PrintWriter out4 = new PrintWriter(new FileOutputStream("./negdist"), true);
		dist=new double[unlData.numInstances()];
		while(unlData.numInstances()!=0){
			double[] mindist = findUnlabeledNN(negData, unlData);
			int unlpos = Utils.minIndex(mindist);
			double now_dist = mindist[unlpos];
			dist[flg]=now_dist;
			out3.println(unlData.instance(unlpos));
			out4.println(now_dist);
//			unlData.instance(unlpos).setClassValue(0.0);
			negData.add(unlData.instance(unlpos));
			unlData.delete(unlpos);
			flg++;
		}
		out3.close();
		out4.close();
		double[] scc1=SCC(dist, negData.numInstances()-1);
		index=Utils.maxIndex(scc1);
		negindex=Utils.minIndex(scc1);
		System.out.println(index+"\t"+negindex);
		
//		for (int i = negData.numInstances() - 1; i > index; i--) {
//			negData.instance(i).setClassValue(1.0);
//			unlData.add(negData.instance(i));
//			negData.delete(i);
//		}
		System.out.println("finish");
		System.exit(0);
//		unlData=new Instances(traindata);
		
		
		// split the POS dataset
		Instances two[] = splitdata(posData);
		Instances posTrainData = two[0];
		Instances posTestData = two[1];
		// split the UN dataset
		two = splitdata(unlData);
		Instances unlTrainData = two[0];
		Instances unlTestData = two[1];
		
		System.out.println("Train:\tPOS:"+posTrainData.numInstances()+"\tUNL:"+unlTrainData.numInstances());
		
		c45posunl = new ClassifyPOSC45[9];
		for (int i = 0; i < c45posunl.length; i++) {
			c45posunl[i] = new ClassifyPOSC45((i + 1) / 40.0);
			c45posunl[i].setDataset(posTrainData, unlTrainData);
			c45posunl[i].buildClassifier(null);
			System.out.println("buildfinished");
		}

		System.out.println("Test:\tPOS:"+posTestData.numInstances()+"\tUNL:"+unlTestData.numInstances());
		// select best DF
		double dEstimate[] = new double[c45posunl.length];
		for (int i = 0; i < dEstimate.length; i++) {
			dEstimate[i] = evaluateBaseEstimate(c45posunl[i], posTestData, unlTestData);
			System.out.println("error rate of "+i+" classifier is: "+dEstimate[i]);
		}

		int nBestIndex = Utils.minIndex(dEstimate);
		System.out.println(nBestIndex);
		System.out.println("Final Classifier:\tPOS:"+posData.numInstances()+"\tUNL:"+unlData.numInstances());
		// train the final classifier
		claC45posunl = new ClassifyPOSC45((nBestIndex + 1) / 40.0);
		claC45posunl.setDataset(posData, unlData);
		claC45posunl.buildClassifier(null);
	}

	// estimate the performance of base classifier
	double evaluateBaseEstimate(ClassifyPOSC45 c45posunl, Instances posTestData, Instances unTestData) throws Exception {
		int nPosError = 0;
		int nUnlError = 0;
		double error=0.0;

		// evaluate on POS dataset
		for (int i = 0; i < posTestData.numInstances(); i++) {
			double classlabel=-1;
			Instance sample = posTestData.instance(i);
			classlabel = c45posunl.classifyInstance(sample);
			if (!Utils.eq(classlabel, sample.classValue()))
				nPosError++;
		}

		// evaluate on UNL dataset
		for (int i = 0; i < unTestData.numInstances(); i++) {
			double classlabel=-1;
			Instance sample = unTestData.instance(i);
			classlabel = c45posunl.classifyInstance(sample);
			if (!Utils.eq(classlabel, sample.classValue()))
				nUnlError++;
		}
		error = 2.0 * (double) nPosError / posTestData.numInstances() + (double) nUnlError / unTestData.numInstances();
		return error;
	}

	// classify
	public double classifyInstance(Instance instance) throws Exception {
//		System.out.println(claC45posunl.classifyInstance(instance));
		return claC45posunl.classifyInstance(instance);
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
	
	/**
	 * Minimum Distances between each U and P sets
	 * @param p Labeled Positive example
	 * @param u Unlabeled data
	 * @return Every u to P mindistance
	 */
	private double[] findUnlabeledNN(Instances p, Instances u) {
		double[][]updist=new double[u.numInstances()][p.numInstances()];
		double[]mindist=new double[u.numInstances()];
		for (int i = 0; i < u.numInstances(); i++) {
			for (int j = 0; j < p.numInstances(); j++) {
				Sequence[] splitsequences = new Sequence[2];
				splitsequences[0] = InsToSeq(u.instance(i));
				splitsequences[1] = InsToSeq(p.instance(j));
				updist[i][j] = splitsequences[0].distance(splitsequences[1]);
			}
			// min distance from u to p
			mindist[i]=updist[i][Utils.minIndex(updist[i])];
		}
		return mindist;
	}

	public void setRatio(double ratio) {
		this.ratio = ratio;
	}
	
	/**
	 * Convert Instance to Sequence
	 * @param sample Instance to be convert
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
