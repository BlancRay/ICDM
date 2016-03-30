/*******************************************************************************
 * Copyright (C) 2014 Anonymized
 * Contributors:
 * 	Anonymized
 * 
 * This file is part of ICDM2014SUBMISSION. 
 * This is a program related to the paper "Dynamic Time Warping Averaging of 
 * Time Series allows more Accurate and Faster Classification" submitted to the
 * 2014 Int. Conf. on Data Mining.
 * 
 * ICDM2014SUBMISSION is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * ICDM2014SUBMISSION is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with ICDM2014SUBMISSION.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package items;

import java.util.ArrayList;

public class Sequences {

	public static Sequence medoid(final Sequence... tabSequence) {
		Sequence medoid = null;
		double lowestInertia = Double.MAX_VALUE;
		for (Sequence possibleMedoid : tabSequence) {
			double tmpInertia = possibleMedoid.sumOfSquares(tabSequence);
			if (tmpInertia < lowestInertia) {
				medoid = possibleMedoid;
				lowestInertia = tmpInertia;
			}
		}
		return medoid;
	}

	public static int medoidIndex(final ArrayList<Sequence> tabSequence) {
		int indexMedoid = -1;
		double lowestInertia = Double.MAX_VALUE;

		for (int i = 0; i < tabSequence.size(); i++) {
			Sequence possibleMedoid = tabSequence.get(i);
			double tmpInertia = possibleMedoid.sumOfSquares(tabSequence);
			if (tmpInertia < lowestInertia) {
				indexMedoid = i;
				lowestInertia = tmpInertia;
			}
		}
		return indexMedoid;
	}

	public static int medoidIndex(final ArrayList<Integer> setOfIndexes, double[][] distances) {
		int indexMedoid = -1;
		double lowestInertia = Double.MAX_VALUE;

		for (Integer indexPossibleMedoid : setOfIndexes) {
			double wgss = 0.0;
			for (Integer dataIndex : setOfIndexes) {
				double tmpDistance = distances[indexPossibleMedoid][dataIndex];
				wgss += tmpDistance * tmpDistance;
			}
			if (wgss < lowestInertia) {
				indexMedoid = indexPossibleMedoid;
				lowestInertia = wgss;
			}
		}
		return indexMedoid;
	}

	public static final Sequence meanWithMedoid(final Sequence medoid, final Sequence[] tabSequence) {
		// ~ look for the medoid to initialise the averaging process

		Sequence res = DBAMean(medoid, tabSequence);
		for (int i = 0; i < Sequence.NB_ITERATIONS; i++) {
			res = DBAMean(res, tabSequence);
		}
		return res;

	}

	public static final Sequence mean(final Sequence... tabSequence) {
		// ~ look for the medoid to initialise the averaging process
		Sequence medoid = new Sequence(medoid(tabSequence).getSequence());
		Sequence res = DBAMean(medoid, tabSequence);
		for (int i = 0; i < Sequence.NB_ITERATIONS; i++) {
			res = DBAMean(res, tabSequence);
		}
		return res;

	}
	
	public static final Sequence weightMean(final Sequence[] tabSequence,final double[][] gamma,final int k,final double nck) {
		// ~ look for the medoid to initialise the averaging process
		Sequence medoid = new Sequence(medoid(tabSequence).getSequence());
		Sequence res = weightDBAMean(medoid, tabSequence,gamma,k,nck);
		for (int i = 0; i < Sequence.NB_ITERATIONS; i++) {
			res = weightDBAMean(res, tabSequence,gamma,k,nck);
		}
		return res;

	}
	
	public static final Sequence meanEUC(final Sequence... tabSequence) {
	    Sequence oneSample = tabSequence[0];
	    int lengthMean = Integer.MAX_VALUE;
	    for(Sequence sample: tabSequence){
		lengthMean = Math.min(lengthMean, sample.getNbTuples());
	    }
	    final Itemset[] mean = new Itemset[lengthMean];
	    Itemset[] elements = new Itemset[tabSequence.length];
	    for (int i = 0; i < mean.length; i++) {
		for (int s = 0; s < tabSequence.length; s++) {
		    elements[s]=tabSequence[s].getItem(i);
		}
		mean[i] = oneSample.sequence[0].mean(elements);
	    }
	    return new Sequence(mean);
	}

	private synchronized static final Sequence DBAMean(final Sequence oldCenter, final Sequence[] tabSequence) {
		@SuppressWarnings("unchecked")
		final ArrayList<Itemset>[] tupleAssociation = new ArrayList[oldCenter.getNbTuples()];
		for (int i = 0; i < tupleAssociation.length; i++) {
			tupleAssociation[i] = new ArrayList<Itemset>(tabSequence.length);
		}
		int nbTuplesAverageSeq, i, j, indiceRes;
		double res = 0.0;
		final int tailleCenter = oldCenter.getNbTuples();
		int tailleT;

		for (final Sequence S : tabSequence) {

			tailleT = S.getNbTuples();

			Sequence.matriceW[0][0] = oldCenter.sequence[0].squaredDistance(S.sequence[0]);
			Sequence.matriceChoix[0][0] = Sequence.NOTHING;
			Sequence.optimalPathLength[0][0] = 0;

			for (i = 1; i < tailleCenter; i++) {
				Sequence.matriceW[i][0] = Sequence.matriceW[i - 1][0] + oldCenter.sequence[i].squaredDistance(S.sequence[0]);
				Sequence.matriceChoix[i][0] = Sequence.TOP;
				Sequence.optimalPathLength[i][0] = i;
			}
			for (j = 1; j < tailleT; j++) {
				Sequence.matriceW[0][j] = Sequence.matriceW[0][j - 1] + S.sequence[j].squaredDistance(oldCenter.sequence[0]);
				Sequence.matriceChoix[0][j] = Sequence.LEFT;
				Sequence.optimalPathLength[0][j] = j;
			}

			for (i = 1; i < tailleCenter; i++) {
				for (j = 1; j < tailleT; j++) {
					indiceRes = Tools.ArgMin3(Sequence.matriceW[i - 1][j - 1], Sequence.matriceW[i][j - 1], Sequence.matriceW[i - 1][j]);
					Sequence.matriceChoix[i][j] = indiceRes;
					switch (indiceRes) {
					case Sequence.DIAGONAL:
						res = Sequence.matriceW[i - 1][j - 1];
						Sequence.optimalPathLength[i][j] = Sequence.optimalPathLength[i - 1][j - 1] + 1;
						break;
					case Sequence.LEFT:
						res = Sequence.matriceW[i][j - 1];
						Sequence.optimalPathLength[i][j] = Sequence.optimalPathLength[i][j - 1] + 1;
						break;
					case Sequence.TOP:
						res = Sequence.matriceW[i - 1][j];
						Sequence.optimalPathLength[i][j] = Sequence.optimalPathLength[i - 1][j] + 1;
						break;
					}
					Sequence.matriceW[i][j] = res + oldCenter.sequence[i].squaredDistance(S.sequence[j]);

				}
			}

			nbTuplesAverageSeq = Sequence.optimalPathLength[tailleCenter - 1][tailleT - 1] + 1;

			i = tailleCenter - 1;
			j = tailleT - 1;

			for (int t = nbTuplesAverageSeq - 1; t >= 0; t--) {
				tupleAssociation[i].add(S.sequence[j]);
				switch (Sequence.matriceChoix[i][j]) {
				case Sequence.DIAGONAL:
					i = i - 1;
					j = j - 1;
					break;
				case Sequence.LEFT:
					j = j - 1;
					break;
				case Sequence.TOP:
					i = i - 1;
					break;
				}

			}

		}
		final Itemset[] tuplesAverageSeq = new Itemset[tailleCenter];

		for (int t = 0; t < tailleCenter; t++) {
			tuplesAverageSeq[t] = oldCenter.sequence[0].mean(tupleAssociation[t].toArray(new Itemset[0]));
		}
		final Sequence newCenter = new Sequence(tuplesAverageSeq);
		return newCenter;

	}
	
	private synchronized static final Sequence weightDBAMean(final Sequence oldCenter, final Sequence[] tabSequence, final double[][] gamma,final int k,final double nck) {
		@SuppressWarnings("unchecked")
		final ArrayList<Itemset>[] tupleAssociation = new ArrayList[oldCenter.getNbTuples()];
//		final ArrayList<Double>[] sumgamma = new ArrayList[oldCenter.getNbTuples()];
		final double[] sumgamma = new double[oldCenter.getNbTuples()];
		for (int i = 0; i < tupleAssociation.length; i++) {
			tupleAssociation[i] = new ArrayList<Itemset>(tabSequence.length);
//			sumgamma[i] = new ArrayList<Double>(tabSequence.length);
		}
		int nbTuplesAverageSeq, i, j, indiceRes;
		double res = 0.0;
		final int tailleCenter = oldCenter.getNbTuples();
		int tailleT;
		int sequencenb=0;
		for (Sequence S : tabSequence) {

			tailleT = S.getNbTuples();

			Sequence.matriceW[0][0] = oldCenter.sequence[0].squaredDistance(S.sequence[0]);
			Sequence.matriceChoix[0][0] = Sequence.NOTHING;
			Sequence.optimalPathLength[0][0] = 0;

			for (i = 1; i < tailleCenter; i++) {
				Sequence.matriceW[i][0] = Sequence.matriceW[i - 1][0] + oldCenter.sequence[i].squaredDistance(S.sequence[0]);
				Sequence.matriceChoix[i][0] = Sequence.TOP;
				Sequence.optimalPathLength[i][0] = i;
			}
			for (j = 1; j < tailleT; j++) {
				Sequence.matriceW[0][j] = Sequence.matriceW[0][j - 1] + S.sequence[j].squaredDistance(oldCenter.sequence[0]);
				Sequence.matriceChoix[0][j] = Sequence.LEFT;
				Sequence.optimalPathLength[0][j] = j;
			}

			for (i = 1; i < tailleCenter; i++) {
				for (j = 1; j < tailleT; j++) {
					indiceRes = Tools.ArgMin3(Sequence.matriceW[i - 1][j - 1], Sequence.matriceW[i][j - 1], Sequence.matriceW[i - 1][j]);
					Sequence.matriceChoix[i][j] = indiceRes;
					switch (indiceRes) {
					case Sequence.DIAGONAL:
						res = Sequence.matriceW[i - 1][j - 1];
						Sequence.optimalPathLength[i][j] = Sequence.optimalPathLength[i - 1][j - 1] + 1;
						break;
					case Sequence.LEFT:
						res = Sequence.matriceW[i][j - 1];
						Sequence.optimalPathLength[i][j] = Sequence.optimalPathLength[i][j - 1] + 1;
						break;
					case Sequence.TOP:
						res = Sequence.matriceW[i - 1][j];
						Sequence.optimalPathLength[i][j] = Sequence.optimalPathLength[i - 1][j] + 1;
						break;
					}
					Sequence.matriceW[i][j] = res + oldCenter.sequence[i].squaredDistance(S.sequence[j]);

				}
			}

			nbTuplesAverageSeq = Sequence.optimalPathLength[tailleCenter - 1][tailleT - 1] + 1;

			i = tailleCenter - 1;
			j = tailleT - 1;
			for (int t = nbTuplesAverageSeq - 1; t >= 0; t--) {
				MonoDoubleItemSet m = (MonoDoubleItemSet) S.sequence[j].clone();
				m = new MonoDoubleItemSet(m.getValue() * gamma[sequencenb][k]);
				tupleAssociation[i].add(m);
				sumgamma[i] += gamma[sequencenb][k];
				switch (Sequence.matriceChoix[i][j]) {
				case Sequence.DIAGONAL:
					i = i - 1;
					j = j - 1;
					break;
				case Sequence.LEFT:
					j = j - 1;
					break;
				case Sequence.TOP:
					i = i - 1;
					break;
				}
			}
			sequencenb++;
		}
		final Itemset[] tuplesAverageSeq = new Itemset[tailleCenter];

		for (int t = 0; t < tailleCenter; t++) {
			//should be in weight mean.
//			tuplesAverageSeq[t] = oldCenter.sequence[0].mean(tupleAssociation[t].toArray(new Itemset[0]));
			tuplesAverageSeq[t] = oldCenter.sequence[0].weightmean(tupleAssociation[t].toArray(new Itemset[0]),sumgamma[t]);
		}
		final Sequence newCenter = new Sequence(tuplesAverageSeq);
		return newCenter;

	}
}
