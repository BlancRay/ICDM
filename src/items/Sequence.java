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

import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import com.sun.xml.internal.bind.v2.runtime.unmarshaller.XsiNilLoader.Array;

import weka.core.Utils;

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
		for (int i = 0; i < length; i++) {
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
	
	public synchronized double distanceDDTW(Sequence a) {
		Sequence S1 = this;
		Sequence S2 = a;

		final int tailleS = S1.getNbTuples();
		final int tailleT = S2.getNbTuples();

		int i, j;
		matriceW[0][0] = S1.sequence[0].squaredDistance(S2.sequence[0]);
		for (i = 1; i < tailleS; i++) {
			matriceW[i][0] = matriceW[i - 1][0] + DDTW(S1,i).squaredDistance(S2.sequence[0]);
		}
		for (j = 1; j < tailleT; j++) {
			matriceW[0][j] = matriceW[0][j - 1] + S1.sequence[0].squaredDistance(DDTW(S2, j));
		}

		for (i = 1; i < tailleS; i++) {
			for (j = 1; j < tailleT; j++) {
				matriceW[i][j] = Tools.Min3(matriceW[i - 1][j - 1], matriceW[i][j - 1], matriceW[i - 1][j]) + DDTW(S1, i).squaredDistance(DDTW(S2, j));
			}
		}
		return sqrt(matriceW[tailleS - 1][tailleT - 1]);
	}
	public synchronized MonoDoubleItemSet DDTW(Sequence S,int index){
		MonoDoubleItemSet x=new MonoDoubleItemSet(0);
		x.value=(S.sequence[index].getValue()-S.sequence[index-1].getValue()+((S.sequence[index+1].getValue()-S.sequence[index-1].getValue())/2))/2;
		return x;
	}
	
	public synchronized double distanceDTWD(Sequence a) {
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
		double ed=S1.distanceEuc(S2)+Double.MIN_VALUE;
		return (sqrt(matriceW[tailleS - 1][tailleT - 1])/ed);
	}
	
	public synchronized double LB_distance(Sequence a,double longestdist) {
		double best_so_far = longestdist;
		double LB_dist = 0.0;
		double true_dist = 0.0;
		Sequence S1 = this;
		Sequence S2 = a;
		final int tailleS = S1.getNbTuples();
		final int tailleT = S2.getNbTuples();

		double[] U = new double[tailleS];
		double[] L = new double[tailleS];
		for (int k = 0; k < L.length; k++) {
			int r_i[] = new int[5];
			int r_j[] = new int[5];
			for (int n = 0; n < r_j.length; n++) {
				r_i[n] = k - n;
				r_j[n] = k + n;
				if (r_i[n] < 0)
					r_i[n] = 0;
				if (r_j[n] > tailleS-1)
					r_j[n] = tailleS-1;
			}
			double value_i[] = new double[5];
			double value_j[] = new double[5];
			for (int m = 0; m < r_j.length; m++) {
				value_i[m] = S1.sequence[r_i[m]].getValue();
				value_j[m] = S1.sequence[r_j[m]].getValue();
			}
			U[k] = value_i[Utils.maxIndex(value_i)];
			L[k] = value_j[Utils.minIndex(value_j)];
		}
		double[] dist = new double[tailleS];
		for (int i = 0; i < tailleT; i++) {
			double[] C = new double[tailleT];
			for (int j = 0; j < C.length; j++) {
				C[j] = S2.sequence[j].getValue();
			}
			if (C[i] > U[i])
				dist[i] = Math.pow((C[i] - U[i]), 2);
			else if (C[i] < L[i])
				dist[i] = Math.pow((C[i] - L[i]), 2);
			else
				dist[i] = 0.0;
		}
		LB_dist = sqrt(Utils.sum(dist));
		if (LB_dist < best_so_far) {
			true_dist = S1.distance(S2);
			if (true_dist < best_so_far)
				best_so_far = true_dist;
		}
		return best_so_far;
	}
	
	public synchronized double DTWD_distance(Sequence a) {
		Sequence S1 = this;
		Sequence S2 = a;
		return S1.distance(S2) / (S1.distanceEuc(S2) + Double.MIN_NORMAL);
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
	
	public final double EUCsumOfSquares(ArrayList<Sequence> tabSequence) {
		double inertia = 0.0;
		for (Sequence seq : tabSequence) {
			double dist = this.distanceEuc(seq);
			// the distance in Itemset should be squared
			inertia += dist * dist; // inertia = sum of the squared distances
		}

		return inertia;

	}

	public final double EUCvar(ArrayList<Sequence> tabSequence) {
		final int length = this.getNbTuples();
		double dist = 0.0;
		Sequence[] pow2seq = new Sequence[tabSequence.size()];
		Sequence s = (Sequence) this.clone();
		Sequence eofx2 = null;
		int j = 0;
		for (Sequence seq : tabSequence) {
			Itemset[] item = new Itemset[length];
			Sequence tmp =new Sequence(item);
			for (int i = 0; i < length; i++) {
				tmp.sequence[i] = seq.sequence[i].pow2();
			}
			pow2seq[j] = tmp;
			j++;
		}
		eofx2 = Sequences.meanEUC(pow2seq);
		for (int i = 0; i < this.getNbTuples(); i++) {
			s.sequence[i] = s.sequence[i].pow2();
		}
		dist = s.distanceEuc(eofx2);
		return sqrt(dist);
	}

}
