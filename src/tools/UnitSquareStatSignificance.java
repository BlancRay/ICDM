package tools;

import java.security.SecureRandom;

public class UnitSquareStatSignificance {

	public static void main(String... args) throws Exception {

		int nRuns = 500000;
		// int nTestSamples = 100;
		int nExamplesPerClass = 100;
		for (int n = 2; n <= nExamplesPerClass; n++) {

			SecureRandom r = SecureRandom.getInstance("SHA1PRNG");

			double[] xFalseClass = new double[nExamplesPerClass];
			double[] yFalseClass = new double[nExamplesPerClass];
			double[] xTrueClass = new double[nExamplesPerClass];
			double[] yTrueClass = new double[nExamplesPerClass];

			int nErrorsNC = 0, nErrorsNN = 0;

			double averageRankNC = 0.0, averageRankNN = 0.0;
			

			for (int i = 0; i < nRuns; i++) {
				boolean correctNC,correctNN;
				double xCentroidFalse = 0.0;
				double yCentroidFalse = 0.0;
				double xCentroidTrue = 0.0;
				double yCentroidTrue = 0.0;

				// create training data
				for (int j = 0; j < nExamplesPerClass; j++) {
					xFalseClass[j] = r.nextDouble() / 2.0;
					yFalseClass[j] = r.nextDouble();
					xTrueClass[j] = r.nextDouble() / 2.0 + .5;
					yTrueClass[j] = r.nextDouble();

					xCentroidFalse += xFalseClass[j];
					yCentroidFalse += yFalseClass[j];
					xCentroidTrue += xTrueClass[j];
					yCentroidTrue += yTrueClass[j];
				}

				// NC
				xCentroidFalse /= nExamplesPerClass;
				yCentroidFalse /= nExamplesPerClass;
				xCentroidTrue /= nExamplesPerClass;
				yCentroidTrue /= nExamplesPerClass;

				double xTestSample = r.nextDouble();
				double yTestSample = r.nextDouble();
				boolean is = (0.5 < xTestSample);
				boolean assigned;

				// System.out.println(xCentroidFalse);

				double distanceToFalseC = 0.0;
				distanceToFalseC += (xTestSample - xCentroidFalse) * (xTestSample - xCentroidFalse);
				distanceToFalseC += (yTestSample - yCentroidFalse) * (yTestSample - yCentroidFalse);
				distanceToFalseC = Math.sqrt(distanceToFalseC);

				double distanceToTrueC = 0.0;
				distanceToTrueC += (xTestSample - xCentroidTrue) * (xTestSample - xCentroidTrue);
				distanceToTrueC += (yTestSample - yCentroidTrue) * (yTestSample - yCentroidTrue);
				distanceToTrueC = Math.sqrt(distanceToTrueC);

				assigned = (distanceToTrueC < distanceToFalseC);
				if (is != assigned) {
					nErrorsNC++;
					correctNC=false;
				}else{
					correctNC = true;
				}

				// NN
				double distance = Double.MAX_VALUE;
				for (int j = 0; j < nExamplesPerClass; j++) {
					double distanceToN = 0.0;
					distanceToN += (xTestSample - xFalseClass[j]) * (xTestSample - xFalseClass[j]);
					distanceToN += (yTestSample - yFalseClass[j]) * (yTestSample - yFalseClass[j]);
					distanceToN = Math.sqrt(distanceToN);
					if (distanceToN < distance) {
						distance = distanceToN;
						assigned = false;
					}
				}
				for (int j = 0; j < nExamplesPerClass; j++) {
					double distanceToN = 0.0;
					distanceToN += (xTestSample - xTrueClass[j]) * (xTestSample - xTrueClass[j]);
					distanceToN += (yTestSample - yTrueClass[j]) * (yTestSample - yTrueClass[j]);
					distanceToN = Math.sqrt(distanceToN);
					if (distanceToN < distance) {
						distance = distanceToN;
						assigned = true;
					}
				}
				if (is != assigned) {
					nErrorsNN++;
					correctNN = false;
				}else{
					correctNN = true;
				}
				
				if(correctNN){
					if(correctNC){
						averageRankNC+=1.5;
						averageRankNN+=1.5;
					}else{
						averageRankNN+=1.0;
						averageRankNC+=2.0;
					}
				}else{
					if(correctNC){
						averageRankNC+=1.0;
						averageRankNN+=2.0;
					}else{
						averageRankNC+=1.5;
						averageRankNN+=1.5;
					}
				}
				
			}
			averageRankNC/=nRuns;
			averageRankNN/=nRuns;
			double diff = Math.abs(averageRankNC-averageRankNN);
			double CD = 1.96*Math.sqrt(1.0/nRuns);
			if(CD<diff){
				if(averageRankNC<averageRankNN){
					System.out.println("With "+n+" examples, NC is significantly better than NN (diff rank="+diff+" > CD="+CD+")");
				}else{
					System.out.println("With "+n+" examples, NN is significantly better than NC (diff rank="+diff+" > CD="+CD+")");
				}
			}else{
				System.out.println("With "+n+" examples, the results are not statistically significant (diff rank="+diff+" - CD="+CD+").");
			}

//			System.out.println("Error NN = " + (1.0 * nErrorsNN / nRuns));
//			System.out.println("Error NC = " + (1.0 * nErrorsNC / nRuns));
		}

	}

}
