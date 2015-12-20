package classif.gmm;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;

import classif.ExperimentsLauncher;
import items.MonoDoubleItemSet;
import items.Sequence;
import weka.core.Instances;

public class GMMin2D {
	
	public static void main(String...args) throws Exception{
		SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd-HH-mm");
		String time =df.format(new Date());
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
		PrintStream outtrain,outGaussian,outtest;
		outtrain = new PrintStream(new FileOutputStream("C:\\Users\\leix\\Downloads\\UCR_TS_Archive_2015\\GMM\\GMM_TRAIN", true));
		outtest = new PrintStream(new FileOutputStream("C:\\Users\\leix\\Downloads\\UCR_TS_Archive_2015\\GMM\\GMM_TEST", true));
		outGaussian = new PrintStream(new FileOutputStream("C:\\Users\\leix\\Downloads\\UCR_TS_Archive_2015\\GMM\\Gaussian", true));
	    //testing GMM with a mimxture of normal
	    int nDataPoints = 10000;
	    int nGaussians = 4;
	    int nDims = 2;
	    Random r = new Random(3071980);
	    
	    double[][]mus = new double[nGaussians][nDims];
	    double[][]sigmas = new double[nGaussians][nDims];
	    
	    double []pMixtures = new double[nGaussians];
	    double sum = 0.0;
	    for (int i = 0; i < pMixtures.length; i++) {
		pMixtures[i]=r.nextDouble();
		sum+=pMixtures[i];
	    }
	    for (int i = 0; i < pMixtures.length; i++) {//normalize
		pMixtures[i]/=sum;
	    }
	    outGaussian.println("priors for mixtures="+Arrays.toString(pMixtures));
	    
	    //generate some randome mixture parameters
	    for (int gaussian = 0; gaussian < nGaussians; gaussian++) {
		for(int dim=0; dim<nDims;dim++){
		    mus[gaussian][dim] = r.nextDouble()*10.0; //generating 'dim'-coordinate of the 'gaussian' center
		    sigmas[gaussian][dim] = r.nextDouble();
		}
		
		outGaussian.println("Gaussian #"+gaussian+":mu="+Arrays.toString(mus[gaussian])+"\tsigma="+Arrays.toString(sigmas[gaussian]));
	    }
	    
	    MonoDoubleItemSet[]sampleCoordinates_train = new MonoDoubleItemSet[nDims];
	    MonoDoubleItemSet[]sampleCoordinates_test = new MonoDoubleItemSet[nDims];
//	    Sequence[] train = new Sequence[nDataPoints];
//	    Sequence[] test = new Sequence[nDataPoints];
	    for (int instance = 0; instance < nDataPoints; instance++) {
		
		//choosing which mixture it's coming from
		int chosenGaussian = 0;
		double sumProba = pMixtures[chosenGaussian];
		double rand = r.nextDouble();
		while (rand > sumProba) {
			chosenGaussian++;
			sumProba += pMixtures[chosenGaussian];
		}
		
		//now I know I want to sample from gaussian number 'chosenGaussian'
		for (int dim = 0; dim < nDims; dim++) {
		    sampleCoordinates_train[dim]=new MonoDoubleItemSet(Double.parseDouble(String.format("%.5f",(r.nextGaussian()*sigmas[chosenGaussian][dim]+mus[chosenGaussian][dim]))));
		}
//		train[instance]=new Sequence(sampleCoordinates_train);
		
		outtrain.println(chosenGaussian+","+sampleCoordinates_train[0]+","+sampleCoordinates_train[1]);
//		System.out.println(Arrays.toString(sampleCoordinates));
		
		for (int dim = 0; dim < nDims; dim++) {
			sampleCoordinates_test[dim]=new MonoDoubleItemSet(Double.parseDouble(String.format("%.5f",(r.nextGaussian()*sigmas[chosenGaussian][dim]+mus[chosenGaussian][dim]))));
		}
//		test[instance]=new Sequence(sampleCoordinates_train);
		
		outtest.println(chosenGaussian+","+sampleCoordinates_test[0]+","+sampleCoordinates_test[1]);
	    }
	    
	    //here to launch GMM
	    
	    
	    outtrain.close();
	    outtest.close();
	    outGaussian.close();
	}
}
