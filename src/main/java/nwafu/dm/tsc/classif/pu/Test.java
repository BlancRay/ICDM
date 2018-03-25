package nwafu.dm.tsc.classif.pu;

import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.Arrays;

import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.Sequences;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class Test extends AbstractClassifier {
	private static final long serialVersionUID = 4578982546614560972L;
	public void buildClassifier(Instances data) throws Exception {
		Instances traindata=new Instances(data);
		// split data to P and U sets
		Instances posData = new Instances(traindata, 0);
		Instances unlData = new Instances(traindata, 0);
		Instances negData = new Instances(traindata, 0);
		posData.add(traindata.instance(0));
		traindata.delete(0);
		
		PrintWriter out1 = new PrintWriter(new FileOutputStream("./pos"), true);
//		PrintWriter out2 = new PrintWriter(new FileOutputStream("./neg"), true);
		double[]dist=new double[traindata.numInstances()];
		int flg=0;
		while(traindata.numInstances()!=0){
			double[] mindist = findUnlabeledNN(posData, traindata);
			int unlpos = Utils.minIndex(mindist);
			double now_dist = mindist[unlpos];
			dist[flg]=now_dist;
			System.out.println(traindata.instance(unlpos));
			out1.println(traindata.instance(unlpos));
			posData.add(traindata.instance(unlpos));
//			traindata.instance(unlpos).setClassValue(0.0);
//			data.delete(unlpos);
			traindata.delete(unlpos);
			flg++;
		}
//		System.out.println(posData.numInstances());
//		
//		for (int i = 0; i < data.numInstances(); i++) {
//			System.out.println(data.instance(i));
//		}
		out1.close();
//		out2.close();
		double[] scc=SCC(dist, posData.numInstances()-1);
		int index=Utils.maxIndex(scc);
		int negindex=Utils.minIndex(scc);
		System.out.println("pos:\t"+Utils.maxIndex(scc)+"\t"+Utils.minIndex(scc));
		for (int i = posData.numInstances() - 1; i > index; i--) {
//			posData.instance(i).setClassValue(1.0);
			unlData.add(posData.instance(i));
			posData.delete(i);
		}
		int[] d1=new int[unlData.numInstances()];
		for (int i = 0; i < unlData.numInstances()-1; i++) {
			double[] d2=new double[unlData.numInstances()-i-1];
			for (int j = 0; j < unlData.numInstances(); j++) {
				if (j<=i) {
					d2[j]=Double.MAX_VALUE;
				}else
				d2[j]=InsToSeq(unlData.instance(i)).distance(InsToSeq(unlData.instance(j)));
			}
			d1[i]=Utils.minIndex(d2);
		}
		int[] p={Utils.minIndex(d1),d1[Utils.minIndex(d1)]};
		negData.add(unlData.instance(p[0]));
		negData.add(unlData.instance(p[1]));
		unlData.delete(p[1]);
		unlData.delete(p[0]);
		PrintWriter out3 = new PrintWriter(new FileOutputStream("./neg"), true);
//		PrintWriter out4 = new PrintWriter(new FileOutputStream("./negdist"), true);
		flg=0;
		dist=new double[unlData.numInstances()];
		while(unlData.numInstances()!=0){
			double[] mindist = findUnlabeledNN(negData, unlData);
			int unlpos = Utils.minIndex(mindist);
			double now_dist = mindist[unlpos];
			dist[flg]=now_dist;
			out3.println(unlData.instance(unlpos));
//			out4.println(now_dist);
//			unlData.instance(unlpos).setClassValue(0.0);
			negData.add(unlData.instance(unlpos));
			unlData.delete(unlpos);
			flg++;
		}
		out3.close();
//		out4.close();
		double[] scc1=SCC(dist, negData.numInstances()-1);
		index=Utils.maxIndex(scc1);
		negindex=Utils.minIndex(scc1);
		System.out.println(index+"\t"+negindex);
		
//		for (int i = negData.numInstances() - 1; i > index; i--) {
//			negData.instance(i).setClassValue(1.0);
//			unlData.add(negData.instance(i));
//			negData.delete(i);
//		}
	}
	private double[] findUnlabeledNN(Instances p, Instances u) {
		Sequence[] sequences=new Sequence[p.numInstances()];
		for (int i = 0; i < p.numInstances(); i++) {
			sequences[i]=InsToSeq(p.instance(i));
		}
		Sequence mean=Sequences.mean(sequences);
		double[]mindist=new double[u.numInstances()];
		for (int i = 0; i < u.numInstances(); i++) {
			mindist[i]=InsToSeq(u.instance(i)).distance(mean);
		}
		return mindist;
	}
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
