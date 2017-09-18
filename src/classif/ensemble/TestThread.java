package classif.ensemble;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.commons.math3.random.RandomDataGenerator;
import classif.BIGDT.ClassifyBigDT;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class TestThread extends AbstractClassifier {
	private static final long serialVersionUID = 1478342548949228865L;

	public TestThread() {
		super();
	}

	private Instances Traindata;
	private RandomDataGenerator randGen;
	private int nbclassifiers = 1;
	private int K = 5;
	private int nbbestclassifiers;
	private ClassifyBigDT[] classifyBigDTs;

	private void trainThread(Instances traindata) throws InterruptedException {
		ArrayList<Future<ClassifyBigDT>> al = new ArrayList<Future<ClassifyBigDT>>();
		CountDownLatch threadSignal = new CountDownLatch(nbclassifiers);
		ExecutorService executor = Executors.newFixedThreadPool(nbclassifiers);
		for (int i = 0; i < nbclassifiers; i++) {
			Instances resample = new Instances(traindata, traindata.numInstances() / 2);
			randGen = new RandomDataGenerator();
			int[] selected = randGen.nextPermutation(traindata.numInstances(), traindata.numInstances() / 2);
			for (int j = 0; j < selected.length; j++) {
				resample.add(traindata.instance(selected[j]));
			}
			CallThreadTrain callThread = new CallThreadTrain(threadSignal);
			callThread.setTrain(resample);
			al.add(executor.submit(callThread));
		}
		try {
			for (int i = 0; i < al.size(); i++) {
				classifyBigDTs[i] = new ClassifyBigDT();
				classifyBigDTs[i] = (ClassifyBigDT) AbstractClassifier.makeCopy(al.get(i).get());
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		executor.shutdown();
	}

	private class CallThreadTrain implements Callable<ClassifyBigDT> {
		private Instances train;
		private CountDownLatch threadsSignal;

		public CallThreadTrain(CountDownLatch threadsSignal) {
			this.threadsSignal = threadsSignal;
		}

		public void setTrain(Instances train) {
			this.train = train;
		}

		@Override
		public ClassifyBigDT call() throws Exception {
//			System.out.println(Thread.currentThread().getName() + "开始");
//			System.out.println("未完成：" + threadsSignal.getCount());
			ClassifyBigDT dt = new ClassifyBigDT();
			dt.buildClassifier(train);
			threadsSignal.countDown();
//			System.out.println(Thread.currentThread().getName() + "结束. 还有" + threadsSignal.getCount() + " 个线程");
			return dt;
		}
	}
	
	private double[] testThread(ClassifyBigDT[] classifiers,Instances testdata) throws Exception {
		ArrayList<Future<Double>> al = new ArrayList<Future<Double>>();
		CountDownLatch threadSignal = new CountDownLatch(nbclassifiers);
		ExecutorService executor = Executors.newFixedThreadPool(nbclassifiers);
		for (int i = 0; i < nbclassifiers; i++) {
			CallThreadTest callThread = new CallThreadTest(threadSignal);
			callThread.setTest(testdata);
			callThread.setClassifier(classifiers[i]);
			al.add(executor.submit(callThread));
		}
		double[] error=new double[nbclassifiers];
		try {
			for (int i = 0; i < al.size(); i++) {
				error[i] = al.get(i).get();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		executor.shutdown();
		return error;
	}
	private class CallThreadTest implements Callable<Double> {
		private Instances test;
		private ClassifyBigDT classifier;
		private CountDownLatch threadsSignal;

		public void setTest(Instances test) {
			this.test = test;
		}

		public void setClassifier(ClassifyBigDT classifier) throws Exception {
			this.classifier = (ClassifyBigDT) AbstractClassifier.makeCopy(classifier);
		}

		public CallThreadTest(CountDownLatch threadsSignal) {
			this.threadsSignal = threadsSignal;
		}

		@Override
		public Double call() throws Exception {
//			System.out.println(Thread.currentThread().getName() + "开始");
//			System.out.println("未完成：" + threadsSignal.getCount());
			Evaluation evalKNNtest = new Evaluation(Traindata);
			evalKNNtest.evaluateModel(classifier, test);
			double errorRate = evalKNNtest.errorRate();
			threadsSignal.countDown();
//			System.out.println(Thread.currentThread().getName() + "结束. 还有" + threadsSignal.getCount() + " 个线程");
			return errorRate;
		}
	}
	
	private double[] classifyThread(ClassifyBigDT[] classifiers,int[] nbbestclassifiers,Instance testsample,double[] besterrorRate) throws Exception{
		ArrayList<Future<Integer>> al = new ArrayList<Future<Integer>>();
		CountDownLatch threadSignal = new CountDownLatch(nbclassifiers);
		ExecutorService executor = Executors.newFixedThreadPool(nbclassifiers);
		for (int i = 0; i < nbbestclassifiers.length; i++) {
			CallThreadclassify callThread = new CallThreadclassify(threadSignal);
			callThread.setTestSample(testsample);
			callThread.setClassifier(classifiers[nbbestclassifiers[i]]);
			al.add(executor.submit(callThread));
		}
		double[] error=new double[nbclassifiers];
		try {
			for (int i = 0; i < al.size(); i++) {
				error[al.get(i).get()]+= besterrorRate[i];
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		executor.shutdown();
		return error;
	}
	private class CallThreadclassify implements Callable<Integer> {
		private Instance testSample;
		private ClassifyBigDT classifier;
		private CountDownLatch threadsSignal;

		public void setTestSample(Instance testSample) {
			this.testSample = testSample;
		}

		public void setClassifier(ClassifyBigDT classifier) throws Exception {
			this.classifier = (ClassifyBigDT) AbstractClassifier.makeCopy(classifier);
		}

		public CallThreadclassify(CountDownLatch threadsSignal) {
			this.threadsSignal = threadsSignal;
		}

		@Override
		public Integer call() throws Exception {
//			System.out.println(Thread.currentThread().getName() + "开始");
//			System.out.println("未完成：" + threadsSignal.getCount());
			int classlable=(int) classifier.classifyInstance(testSample);
			threadsSignal.countDown();
//			System.out.println(Thread.currentThread().getName() + "结束. 还有" + threadsSignal.getCount() + " 个线程");
			return classlable;
		}
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		Traindata = new Instances(data);
		// K=Traindata.numInstances()/2;
		nbclassifiers = Math.min(Math.max((Traindata.numInstances() / Traindata.numClasses()) / 2, 30), 100);
		// nbbestclassifiers=nbclassifiers/10;
		classifyBigDTs = new ClassifyBigDT[nbclassifiers];
		trainThread(Traindata);
	}

	public double classifyInstance(Instance sample) throws Exception {
		double start,end,total;
		/*
		 * Find KNN Test each kMeansCached classifier select best 10 classify
		 * with weight classify query
		 */
		FindKNN findKNN = new FindKNN(sample, Traindata, K);
		Instances KNNInstances = findKNN.KNN();
		double[] errorRate = new double[nbclassifiers];

		start=System.currentTimeMillis();
		errorRate=testThread(classifyBigDTs,KNNInstances);
		end=System.currentTimeMillis();
		total=end-start;
		System.out.println("testthread"+total);
		
		start=System.currentTimeMillis();
		for (int i = 0; i < classifyBigDTs.length; i++) {
			ClassifyBigDT bigDT = classifyBigDTs[i];
			Evaluation evalKNNtest = new Evaluation(Traindata);
			evalKNNtest.evaluateModel(bigDT, KNNInstances);
			errorRate[i]=evalKNNtest.errorRate();
//			if(errorRate[i]>0.0)
//			System.out.println(errorRate[i]);
		}
		end=System.currentTimeMillis();
		total=end-start;
		System.out.println("test"+total);
		
		nbbestclassifiers = 0;
		double avgerror = Utils.mean(errorRate);
		for (int i = 0; i < errorRate.length; i++) {
			if (Utils.smOrEq(errorRate[i], avgerror))
				nbbestclassifiers++;
		}

		int[] classifiers = Utils.sort(errorRate);
		double[] besterrorRate = new double[nbbestclassifiers];
		double[] errorRateCopy = errorRate.clone();
		Arrays.sort(errorRateCopy);
		besterrorRate = Arrays.copyOf(errorRateCopy, nbbestclassifiers);
		for (int i = 0; i < besterrorRate.length; i++) {
			besterrorRate[i] = (1 - besterrorRate[i]);
		}
		Utils.normalize(besterrorRate);
		// System.out.println(Arrays.toString(besterrorRate));
		double[] classlabel = new double[sample.numClasses()];
		start = System.currentTimeMillis();
		classlabel=classifyThread(classifyBigDTs, Arrays.copyOf(classifiers, nbbestclassifiers),sample,besterrorRate);
		end = System.currentTimeMillis();
		total = end - start;
		System.out.println("classifythread" + total);
		start = System.currentTimeMillis();
		for (int i = 0; i < nbbestclassifiers; i++) {
			classlabel[(int) classifyBigDTs[classifiers[i]].classifyInstance(sample)] += besterrorRate[i];
		}
		end = System.currentTimeMillis();
		total = end - start;
		System.out.println("classify" + total);
		// System.out.println(Arrays.toString(classlabel));
		return Utils.maxIndex(classlabel);
	}
}