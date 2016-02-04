package classif.gmm;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import classif.ExperimentsLauncher;
import items.MonoDoubleItemSet;
import weka.experiment.Experiment;

public class GMMin2D {

	public static void main(String... args) throws Exception {
		File repSave = new File("C:\\Users\\leix\\Downloads\\UCR_TS_Archive_2015\\GMM");
		File[] repSavelist;
		if (!repSave.exists()) {
			repSave.mkdirs();
		} else {
			repSavelist = repSave.listFiles();
			for (int i = 0; i < repSavelist.length; i++) {
				repSavelist[i].delete();
			}
		}
		PrintStream outtrain,outtest, outGaussian;
		outtrain = new PrintStream(
				new FileOutputStream("C:\\Users\\leix\\Downloads\\UCR_TS_Archive_2015\\GMM\\GMM_TRAIN", true));
		outtest = new PrintStream(
				new FileOutputStream("C:\\Users\\leix\\Downloads\\UCR_TS_Archive_2015\\GMM\\GMM_TEST", true));
		outGaussian = new PrintStream(
				new FileOutputStream("C:\\Users\\leix\\Downloads\\UCR_TS_Archive_2015\\GMM\\Gaussian", true));
		// testing GMM with a mimxture of normal
		int nDataPoints = 100;
		int nGaussians = 1;
		int nDims = 3;
		RandomGenerator rg = new MersenneTwister();
        RandomDataGenerator r = new RandomDataGenerator(rg);
        Random p = new Random();

		double[][] mus = new double[nGaussians][nDims];
		double[][] sigmas = new double[nGaussians][nDims];

		double[] pMixtures = new double[nGaussians];
		double sum = 0.0;
		for (int i = 0; i < pMixtures.length; i++) {
			pMixtures[i] = p.nextDouble();
			sum += pMixtures[i];
		}
		for (int i = 0; i < pMixtures.length; i++) {// normalize
			pMixtures[i] /= sum;
		}
		outGaussian.println("priors for mixtures=" + Arrays.toString(pMixtures));

		// generate some randome mixture parameters
		for (int gaussian = 0; gaussian < nGaussians; gaussian++) {
			for (int dim = 0; dim < nDims; dim++) {
				mus[gaussian][dim] = p.nextDouble()+10.0; //generating 'dim'-coordinate of the 'gaussian' center
				sigmas[gaussian][dim] = p.nextDouble();
			}
			 outGaussian.println("Gaussian #"+gaussian+":mu="+Arrays.toString(mus[gaussian])+"\tsigma="+Arrays.toString(sigmas[gaussian]));
			 System.out.println("Gaussian #"+gaussian+":mu="+Arrays.toString(mus[gaussian])+"\tsigma="+Arrays.toString(sigmas[gaussian]));
		}

		MonoDoubleItemSet[] sampleCoordinates_train = new MonoDoubleItemSet[nDims];
		for (int instance = 0; instance < nDataPoints; instance++) {

			// choosing which mixture it's coming from
			int chosenGaussian = 0;
			double sumProba = pMixtures[chosenGaussian];
			double rand = p.nextDouble();
			while (rand > sumProba) {
				chosenGaussian++;
				sumProba += pMixtures[chosenGaussian];
			}
			
			 //now I know I want to sample from gaussian number 'chosenGaussian' 
			for (int dim = 0; dim < nDims; dim++) {
				sampleCoordinates_train[dim] = new MonoDoubleItemSet(Double.parseDouble(String.format("%.5f",
						(r.nextGaussian(mus[chosenGaussian][dim],sigmas[chosenGaussian][0])))));
			}
			outtrain.print("1");
			for (int i = 0; i < sampleCoordinates_train.length; i++) {
				outtrain.print("," + sampleCoordinates_train[i]);
			}
			outtrain.println();
			
			outtest.print("1");
			for (int i = 0; i < sampleCoordinates_train.length; i++) {
				outtest.print("," + sampleCoordinates_train[i]);
			}
			outtest.println();
		}

		outtrain.close();
		outtest.close();
		outGaussian.close();
		// here to launch GMM
//		ExperimentsLauncher.main(null);
	}
}
