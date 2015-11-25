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

import static java.lang.Math.sqrt;

import java.util.ArrayList;
import java.util.Arrays;

public class Sequence implements java.io.Serializable {
	private static final long serialVersionUID = -8340081464719919763L;

	protected final static int NB_ITERATIONS = 100;

	protected final static int NOTHING = -1;
	protected final static int DIAGONAL = 0;
	protected final static int LEFT = 1;
	protected final static int TOP = 2;

	public Itemset[] sequence;

	private final static int MAX_SEQ_LENGTH = 4000;
	protected static double[][] matriceW = new double[Sequence.MAX_SEQ_LENGTH][Sequence.MAX_SEQ_LENGTH];
	protected static int[][] matriceChoix = new int[Sequence.MAX_SEQ_LENGTH][Sequence.MAX_SEQ_LENGTH];
	protected static int[][] optimalPathLength = new int[Sequence.MAX_SEQ_LENGTH][Sequence.MAX_SEQ_LENGTH];

	public Sequence(final Itemset[] sequence) {
		if (sequence == null || sequence.length == 0) {
			throw new RuntimeException("sequence vide");
		}
		this.sequence = sequence;
	}

	public Sequence(Sequence o) {
		if (o.sequence == null || o.sequence.length == 0) {
			throw new RuntimeException("sequence vide");
		}
		this.sequence = o.sequence;
	}

	public Object clone() {
		final Itemset[] newSequence = Arrays.copyOf(sequence, sequence.length);
		for (int i = 0; i < newSequence.length; i++) {
			newSequence[i] = sequence[i].clone();
		}

		return new Sequence(newSequence);
	}

	public Itemset getItem(final int n) {
		return sequence[n];
	}

	/**
	 * @return length
	 */
	public final int getNbTuples() {
		return this.sequence.length;
	}

	public double distanceEuc(Sequence a) {
		final int length = this.getNbTuples();

		double res = 0;
		for (int i = 1; i < length; i++) {
			res += this.sequence[i].squaredDistance(a.sequence[i]);
		}
		return sqrt(res);
	}
		
	public synchronized double distance(Sequence a) {
		Sequence S1 = this;
		Sequence S2 = a;

		final int tailleS = S1.getNbTuples();
		final int tailleT = S2.getNbTuples();

		int i, j;
		matriceW[0][0] = S1.sequence[0].squaredDistance(S2.sequence[0]);
		for (i = 1; i < tailleS; i++) {
			matriceW[i][0] = matriceW[i - 1][0] + S1.sequence[i].squaredDistance(S2.sequence[0]);
		}
		for (j = 1; j < tailleT; j++) {
			matriceW[0][j] = matriceW[0][j - 1] + S1.sequence[0].squaredDistance(S2.sequence[j]);
		}

		for (i = 1; i < tailleS; i++) {
			for (j = 1; j < tailleT; j++) {
				matriceW[i][j] = Tools.Min3(matriceW[i - 1][j - 1], matriceW[i][j - 1], matriceW[i - 1][j]) + S1.sequence[i].squaredDistance(S2.sequence[j]);
			}
		}
		return sqrt(matriceW[tailleS - 1][tailleT - 1]);
	}

	@Override
	public String toString() {
		String str = "[";
		for (final Itemset t : sequence) {
			str += "{";
			str += t.toString();
			str += "}";
		}
		str += "]";
		return str;
	}

	/**
	 * @return the sequence
	 */
	public Itemset[] getSequence() {
		return this.sequence;
	}

	/**
	 * Compute the inertia of the sequence with regard to a set of sequences
	 * @param tabSequence the set of sequences to consider
	 * @return the inertia
	 */
	public final double sumOfSquares(final Sequence... tabSequence) {
		double inertia = 0.0;
		for (Sequence seq : tabSequence) {
			double dist = this.distance(seq);
			// the distance in Itemset should be squared
			inertia += dist * dist; // inertia = sum of the squared distances
		}

		return inertia;

	}
	/**
	 * Compute the inertia of the sequence with regard to a set of sequences
	 * @param tabSequence the set of sequences to consider
	 * @return the inertia
	 */
	public final double sumOfSquares(ArrayList<Sequence> tabSequence) {
		double inertia = 0.0;
		for (Sequence seq : tabSequence) {
			double dist = this.distance(seq);
			// the distance in Itemset should be squared
			inertia += dist * dist; // inertia = sum of the squared distances
		}

		return inertia;

	}

}
