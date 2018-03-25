package nwafu.dm.tsc.tools;

import nwafu.dm.tsc.items.Itemset;
import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.SymbolicSequence;

import java.io.File;
import java.util.ArrayList;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class DrawAllSequences {
	File rep;
	SymbolicSequence[] tab;
	int width;

	public DrawAllSequences(File rep, SymbolicSequence[] tab, int width) {
		this.rep = rep;
		if(!rep.exists())rep.mkdirs();
		this.tab = tab;
		this.width = width;
	}

//	public DrawAllSequences(File rep, DataObject[] data, int width) {
//		this.rep = rep;
//		this.width = width;
//		tab = new AttributeSequence[data.length];
//		try {
//			for (int i = 0; i < tab.length; i++) {
//				tab[i] = (AttributeSequence) data[0].getAttribute(0);
//			}
//		} catch (ClassCastException e) {
//			System.err.println("An AttributeSequence is needed");
//			System.exit(1);
//		}
//	}

	public void plot(String name) {
		XYSeries series;
		XYSeriesCollection dataSet = new XYSeriesCollection();
		// on exporte la courbe des séquences
		for (int i = 0; i < tab.length; i++) {
//			if(Math.random()<.5){
				Itemset[] sequence = tab[i].sequence;
				
				series = new XYSeries(i);
	
				for (int j = 0; j < sequence.length; j++) {
					series.add(j, ((MonoDoubleItemSet)sequence[j]).getValue());
				}
				dataSet.addSeries(series);
//			}
			
		}
		JFreeChart chart = ChartFactory.createXYLineChart(name, "Attributes", "",dataSet, PlotOrientation.VERTICAL, false, false, false);
		ChartTools.saveChartAsPDF(chart, new File(rep.getAbsolutePath() + "/"+name+".pdf"));

	}

	public static void main(String[] args) {
//		ImageSequenceData dataDTW;

//		RawImage[] imageTab;

		File repSave = new File("./save/");
//		File rep;
//
//		rep = new File("/home/petitjean/Images/Biganos-corrected-tif-1band/");
//		File[] listImages = rep.listFiles(new FilenameFilter() {
//			public boolean accept(File dir, String name) {
//				return name.endsWith("tif");
//			}
//		});
//
//		Arrays.sort(listImages);
//
//		imageTab = new RawImage[listImages.length];
//
//		for (int i = 0; i < listImages.length; i++) {
//			imageTab[i] = new GEOTiffImage(listImages[i].getAbsolutePath());
//		}
//
//		dataDTW = new ImageSequenceData(imageTab, AttributeSequence.DYNAMIC_TIME_WARPING, imageTab.length);
		String name="flowers";
//		RunSyntheticData l=new RunSyntheticData(new File("/home/forestier/Dropbox/share-Germain/DATASET/"+name+"/"+name+"_norm2"));
		RunSyntheticData l=new RunSyntheticData(new File("./UCR_TS_Archive_2015/Gun_Point/Gun_Point_TEST"));
		SymbolicSequence[]tab=l.getLoadedSequences();
		int[]attribution=l.getAttribution();
		boolean containsMinus1=false,contains0=false;
		for(int i:attribution){
			if(i==-1)containsMinus1=true;
			if(i==0)contains0=true;
		}
		
		ArrayList<SymbolicSequence>[] mTemp=new ArrayList[l.getNbClasses()];
		for(int i=0;i<mTemp.length;i++){
			mTemp[i]=new ArrayList<SymbolicSequence>();
		}
		if(containsMinus1){
			for(int i=0;i<tab.length;i++){
				if(attribution[i]==-1)mTemp[0].add(tab[i]);
				else mTemp[1].add(tab[i]);
			}
		}else if(contains0){
			for(int i=0;i<tab.length;i++){
				mTemp[attribution[i]].add(tab[i]);
			}
		}else{
			for(int i=0;i<tab.length;i++){
				mTemp[attribution[i]-1].add(tab[i]);
			}
		}
		
		SymbolicSequence[][] m=new SymbolicSequence[mTemp.length][];
		for(int i=0;i<m.length;i++){
			m[i]=new SymbolicSequence[mTemp[i].size()];
			for(int j=0;j<m[i].length;j++){
				m[i][j]=(SymbolicSequence)mTemp[i].get(j);
			}
		}
		System.out.println(m.length);
		
		for (int i = 0; i < m.length; i++) {
			new DrawAllSequences(repSave,m[i], 0).plot(attribution[i]+"-"+name+"-"+i);
		}
		
//		for (int j = 0; j < m.length; j++) {
//			for (int i = 0; i < m[j].length; i++) {
//				new DrawAllSequences(repSave, new SymbolicSequence[]{m[j][i]}, 0).plot(j+"-"+name+"-"+i);
//			}
//		}

//		new DrawAllSequences(repSave, m[1], 0).plot(name+"1");
	}

}
