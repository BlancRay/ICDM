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
package nwafu.dm.tsc.classif.ahc;

import nwafu.dm.tsc.items.Sequence;
import nwafu.dm.tsc.items.Sequences;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;

import org.apache.commons.math3.random.RandomDataGenerator;

public class AHCSymbolicSequence {
	
	public ArrayList<Sequence> data;
	public RandomDataGenerator randGen;
	public double[][] distances;
	
	ArrayList<Sequence>[] centroidsForNumberOfClusters;

	/**
	 * used for priority queue for efficient retrieval of pair of clusters to
	 * merge
	 **/
	class Tuple {
		public Tuple(double d, int i, int j, int nSize1, int nSize2) {
			m_fDist = d;
			m_iCluster1 = i;
			m_iCluster2 = j;
			m_nClusterSize1 = nSize1;
			m_nClusterSize2 = nSize2;
		}

		double m_fDist;
		int m_iCluster1;
		int m_iCluster2;
		int m_nClusterSize1;
		int m_nClusterSize2;
	}

	/** comparator used by priority queue **/
	class TupleComparator implements Comparator<Tuple> {
		public int compare(Tuple o1, Tuple o2) {
			if (o1.m_fDist < o2.m_fDist) {
				return -1;
			} else if (o1.m_fDist == o2.m_fDist) {
				return 0;
			}
			return 1;
		}
	}

	class Node implements Serializable {
		Node m_left;
		Node m_right;
		Node m_parent;
		int m_iLeftInstance;
		int m_iRightInstance;
		double m_fLeftLength = 0;
		double m_fRightLength = 0;
		double m_fHeight = 0;

		void setHeight(double fHeight1, double fHeight2) {
			m_fHeight = fHeight1;
			if (m_left == null) {
				m_fLeftLength = fHeight1;
			} else {
				m_fLeftLength = fHeight1 - m_left.m_fHeight;
			}
			if (m_right == null) {
				m_fRightLength = fHeight2;
			} else {
				m_fRightLength = fHeight2 - m_right.m_fHeight;
			}
		}

		void setLength(double fLength1, double fLength2) {
			m_fLeftLength = fLength1;
			m_fRightLength = fLength2;
			m_fHeight = fLength1;
			if (m_left != null) {
				m_fHeight += m_left.m_fHeight;
			}
		}
	}

	public AHCSymbolicSequence(ArrayList<Sequence> data) {
		this.data = data;
		this.randGen = new RandomDataGenerator();

	}

	public void cluster() {

		// cache all distances
		distances = new double[data.size()][data.size()];
		for (int i = 0; i < data.size(); i++) {
			for (int j = i + 1; j < data.size(); j++) {
				distances[i][j] = data.get(i).distance(data.get(j));
				distances[j][i] = distances[i][j];
			}
		}
		System.out.println("distances cached");

		ArrayList<Integer>[] nClusterID = new ArrayList[data.size()];
		for (int i = 0; i < data.size(); i++) {
			nClusterID[i] = new ArrayList<Integer>();
			nClusterID[i].add(i);
		}
		int nClusters = data.size();

		int nInstances = data.size();
		Node[] clusterNodes = new Node[data.size()];

		PriorityQueue<Tuple> queue = new PriorityQueue<Tuple>(nClusters, new TupleComparator());
		double[][] fDistance0 = new double[nClusters][nClusters];
		for (int i = 0; i < nClusters; i++) {
			fDistance0[i][i] = 0;
			for (int j = i + 1; j < nClusters; j++) {
				fDistance0[i][j] = getDistanceClusters(nClusterID[i], nClusterID[j]);
				fDistance0[j][i] = fDistance0[i][j];
				queue.add(new Tuple(fDistance0[i][j], i, j, 1, 1));
			}
		}
		
		centroidsForNumberOfClusters = new ArrayList[data.size()+1];
		centroidsForNumberOfClusters[data.size()] = new ArrayList<Sequence>();
		for (int i = 0; i < data.size(); i++) {
			centroidsForNumberOfClusters[data.size()].add(data.get(i));
		}
		
		
		while (nClusters > 1) {
			System.out.println("nClusters left = "+nClusters);
			int iMin1 = -1;
			int iMin2 = -1;
			Tuple t;
			do {
				t = queue.poll();
			} while (t != null && (nClusterID[t.m_iCluster1].size() != t.m_nClusterSize1 || nClusterID[t.m_iCluster2].size() != t.m_nClusterSize2));
			iMin1 = t.m_iCluster1;
			iMin2 = t.m_iCluster2;
			
			centroidsForNumberOfClusters[nClusters-1] = (ArrayList<Sequence>) centroidsForNumberOfClusters[nClusters].clone();
			
			
			merge(iMin1, iMin2, t.m_fDist, t.m_fDist, nClusterID, centroidsForNumberOfClusters[nClusters-1],clusterNodes,distances);
			for (int i = 0; i < nInstances; i++) {
				if (i != iMin1 && nClusterID[i].size() != 0) {
					int i1 = Math.min(iMin1, i);
					int i2 = Math.max(iMin1, i);
					double fDistance = getDistanceClusters(nClusterID[i1], nClusterID[i2]);
					queue.add(new Tuple(fDistance, i1, i2, nClusterID[i1].size(), nClusterID[i2].size()));
				}
			}

			nClusters--;
			
		}
		System.out.println("Clustering done for all possible cuts");

	}


	double getDistanceClusters(ArrayList<Integer> cluster1, ArrayList<Integer> cluster2) {
		double ESS1 = calcESS(cluster1);
		double ESS2 = calcESS(cluster2);
		ArrayList<Integer> merged = new ArrayList<Integer>();
		merged.addAll(cluster1);
		merged.addAll(cluster2);
		double ESS = calcESS(merged);
		return ESS * merged.size() - ESS1 * cluster1.size() - ESS2 * cluster2.size();
	}

	double calcESS(ArrayList<Integer> cluster) {
		double distance = 0.0;

		for (int i = 0; i < cluster.size(); i++) {
			int indexI = cluster.get(i);
			for (int j = i + 1; j < cluster.size(); j++) {
				int indexJ = cluster.get(j);
				double tmpDistance = distances[indexI][indexJ];
				distance += tmpDistance * tmpDistance;
			}
		}
		distance/=cluster.size();
		return distance;
	}

	void merge(int iMin1, int iMin2, double fDist1, double fDist2, ArrayList<Integer>[] nClusterID, ArrayList<Sequence> centroidsForNumberOfClusters, Node[] clusterNodes, double[][] distances2) {
		if (iMin1 > iMin2) {
			int h = iMin1;
			iMin1 = iMin2;
			iMin2 = h;
			double f = fDist1;
			fDist1 = fDist2;
			fDist2 = f;
		}
		nClusterID[iMin1].addAll(nClusterID[iMin2]);
		
		int medoidIndex = Sequences.medoidIndex(nClusterID[iMin1], distances);
		Sequence medoid = data.get(medoidIndex);
		Sequence [] setOfSequences = new Sequence[nClusterID[iMin1].size()];
		for(int i=0;i<setOfSequences.length;i++){
			setOfSequences[i]=data.get(nClusterID[iMin1].get(i));
		}
		centroidsForNumberOfClusters.set(iMin1, Sequences.meanWithMedoid(medoid, setOfSequences));
		nClusterID[iMin2].clear();
		centroidsForNumberOfClusters.set(iMin2, null);

		Node node = new Node();
		if (clusterNodes[iMin1] == null) {
			node.m_iLeftInstance = iMin1;
		} else {
			node.m_left = clusterNodes[iMin1];
			clusterNodes[iMin1].m_parent = node;
		}
		if (clusterNodes[iMin2] == null) {
			node.m_iRightInstance = iMin2;
		} else {
			node.m_right = clusterNodes[iMin2];
			clusterNodes[iMin2].m_parent = node;
		}
		node.setHeight(fDist1, fDist2);
		clusterNodes[iMin1] = node;
	} 
}
