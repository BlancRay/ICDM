package tools;

public class Normalization {

	
	
	/**
	 * Normalize the P(y|Q)
	 * @param arrary of the P(y|Q)
	 * @return the class of the most probable
	 */
	
	public static int run(double array[]) {
		// double[] array= {0.1,0.2,0.1};
		double sum = 0;
		double flg = 0.0;
		int big = 0;
		
		//Normalize the array
		for (double arr : array) {
			sum += arr;
		}
		for (int i = 0; i < array.length; i++) {
			array[i] = array[i] / sum;
		}
		
		//get the class		
		for (int i = 0; i < array.length; i++) {
			double c = array[i];
			if (flg < c) {
				flg = c;
				big = i;
			}
		}
		return big;
	}
}
