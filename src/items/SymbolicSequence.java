package items;

public class SymbolicSequence {
	public int nbClusters;
	public Itemset[] sequence;
	
	public SymbolicSequence(int nbClusters,final Itemset[] sequence) {
		if(sequence.length<nbClusters){
			this.nbClusters = sequence.length;
		}else{
			this.nbClusters = nbClusters;
		}
		if (sequence == null || sequence.length == 0) {
			throw new RuntimeException("sequence vide");
		}
		this.sequence = sequence;
	}
	public SymbolicSequence(final Itemset[] sequence) {
		if (sequence == null || sequence.length == 0) {
			throw new RuntimeException("sequence vide");
		}
		this.sequence = sequence;
	}

	public SymbolicSequence(Sequence o) {
		if (o.sequence == null || o.sequence.length == 0) {
			throw new RuntimeException("sequence vide");
		}
		this.sequence = o.sequence;
	}
	public Itemset getItem(final int n) {
		return sequence[n];
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
}
