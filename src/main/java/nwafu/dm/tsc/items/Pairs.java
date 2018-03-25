package nwafu.dm.tsc.items;

public class Pairs implements java.io.Serializable {

	private static final long serialVersionUID = 8912556009204963797L;
	protected double distance;
	protected Sequence[] pair=new Sequence[2];
	protected String[] classlable=new String[2];
	
	public double Distance() {
		return pair[0].distance(pair[1]);
	}
	
	
	public Sequence[] getPair() {
		return pair;
	}


	public void setPair(Sequence[] pair) {
		this.pair = pair;
	}

	public double getDistance() {
		return distance;
	}


	public void setDistance(double distance) {
		this.distance = distance;
	}


	public String[] getClasslable() {
		return classlable;
	}


	public void setClasslable(String[] classlable) {
		this.classlable = classlable;
	}
	
	
}
