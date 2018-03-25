package nwafu.dm.tsc.tools;

import nwafu.dm.tsc.items.MonoDoubleItemSet;
import nwafu.dm.tsc.items.SymbolicSequence;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.TreeSet;
import java.util.Vector;

public class RunSyntheticData {
	private SymbolicSequence []tab;
	private int[]attribution;
	private int nbClasses;
	public RunSyntheticData(File f){
		ArrayList<SymbolicSequence> t = new ArrayList<SymbolicSequence>();
		Vector<Integer>a=new Vector<Integer>();
		TreeSet<Integer>classes=new TreeSet<Integer>();
		BufferedReader in=null;
		String line;
		String [] temp;
		try {
			in=new BufferedReader(new FileReader(f));
		} catch (FileNotFoundException e1) {
			System.err.println("Fichier non trouve");
		}
		
		try {
			while((line=in.readLine())!=null){
				if(!line.isEmpty()){
					int k=0;
					while(line.charAt(k)==' ')k++;
					line=line.substring(k);
					temp=line.split(",");
//					temp=line.split("\\s+");
					//System.out.println(line);
//					System.out.println(Arrays.toString(temp));
//					System.out.println(temp[0]);
//					System.out.println(Double.valueOf(temp[0]));
					a.add( (int)Math.round(Double.valueOf(temp[0])));
					classes.add(a.lastElement());
					MonoDoubleItemSet[] tabTuple2 = new MonoDoubleItemSet[temp.length-1];
					for(int j=1;j<temp.length;j++){
						tabTuple2[j-1] = new MonoDoubleItemSet(Double.valueOf(temp[j]));
					}
					// build the sequence
					t.add(new SymbolicSequence(tabTuple2));
					
				}
			}
			
			tab = new SymbolicSequence[t.size()];
			attribution=new int[t.size()];
			for(int j=0;j<t.size();j++){
				tab[j]=t.get(j);
				attribution[j]=a.get(j);
			}
			nbClasses=classes.size();
			
		} catch (IOException e) {
			System.err.println("PB d'I/O");
		}
	}
	
	public SymbolicSequence[]getLoadedSequences(){
		return tab;
	}
	
	public int [] getAttribution(){
		return attribution;
	}
	
	public int getNbClasses(){
		return nbClasses;
	}

//	public static void main(String[] args) {
//		File repData=new File("/home/petitjean/SYNTHETIC/DATASET/");
//		String racine="/home/petitjean/SYNTHETIC/ReferenceDBA-temp/";
//		File[]listData=repData.listFiles();
//		Arrays.sort(listData);
//		long startTime,stopTime,distanceLostTime,startTimeTemp,stopTimeTemp;
//		
//		for(File dataset:listData){
//			String name=dataset.getName();
//			
//			File rep=new File(racine);
//			if(!rep.exists()){
//				rep.mkdir();
//			}
//			
//			RunSyntheticData l=new RunSyntheticData(new File("/home/petitjean/SYNTHETIC/DATASET/"+name+"/"+name+"_norm2"));
//			SymbolicSequence[]tab=l.getLoadedSequences();
//			int[]attribution=l.getAttribution();
//			boolean containsMinus1=false,contains0=false;
//			for(int i:attribution){
//				if(i==-1)containsMinus1=true;
//				if(i==0)contains0=true;
//			}
//			
//			AttributeSequence.setMode(AttributeSequence.DTW_BARYCENTRE);
//			int[] simplTab={-1/*,0*/};
//			
//			try{
//				FileWriter f=new FileWriter(new File(racine+"result"+name+".txt"));
//				f.write("*********** Dataset "+name+" ***********"+"\n");
//				for(int simpl:simplTab){
//					if(simpl==-1){
//						f.write("\n** Sans boost **"+"\n");
//					}else{
//						f.write("\n** Avec boost **"+"\n");
//					}
//					AttributeSequence.setSimplifyFrom(simpl);
//					Vector<AttributeSequence>[] mTemp=new Vector[l.getNbClasses()];
//					for(int i=0;i<mTemp.length;i++){
//						mTemp[i]=new Vector<AttributeSequence>();
//					}
//					if(containsMinus1){
//						for(int i=0;i<tab.length;i++){
//							if(attribution[i]==-1)mTemp[0].add(tab[i]);
//							else mTemp[1].add(tab[i]);
//						}
//					}else if(contains0){
//						for(int i=0;i<tab.length;i++){
//							mTemp[attribution[i]].add(tab[i]);
//						}
//					}else{
//						for(int i=0;i<tab.length;i++){
//							mTemp[attribution[i]-1].add(tab[i]);
//						}
//					}
//					
//					AttributeSequence[][] m=new AttributeSequence[mTemp.length][];
//					for(int i=0;i<m.length;i++){
//						m[i]=new AttributeSequence[mTemp[i].size()];
//						for(int j=0;j<m[i].length;j++){
//							m[i][j]=mTemp[i].get(j);
//						}
//					}
//					
//					AttributeSequence[] moyennes=new AttributeSequence[m.length];
//					
//					double [][]distances=new double[m.length][10];
//					for( int i=0;i<distances.length;i++){
//						Arrays.fill(distances[i],0.0);
//					}
//					
//					startTime=System.currentTimeMillis();
//					distanceLostTime=0;
//					for(int i=0;i<m.length;i++){
//						f.write("\nCalcul moyenne du cluster "+i+"\n");
//						moyennes[i]=(AttributeSequence) AttributeSequence.mean(m[i],m[i][0]);
//						
//						PlotSequenceTab p=new PlotSequenceTab(m[i]);
//						p.plotPDF(new File(racine+name+"_"+i+"_seq.pdf"));
//						
//						p=new PlotSequenceTab(moyennes[i]);
//						p.plotPDF(new File(racine+name+"_"+i+"_moy.pdf"));
//						
//						startTimeTemp=System.currentTimeMillis();
//						for(AttributeSequence s:m[i]){
//							distances[i][0]+=moyennes[i].distance(s);
//						}
//						stopTimeTemp=System.currentTimeMillis();
//						distanceLostTime+=(stopTimeTemp-startTimeTemp);
//						f.write("\tInertie au tour 0 : "+distances[i][0]+"\n");
//						
//						/*for(int j=0;j<9;j++){
//							moyennes[i]=(AttributeSequence) AttributeSequence.mean(m[i],moyennes[i]);
//							startTimeTemp=System.currentTimeMillis();
//							for(AttributeSequence s:m[i]){
//								distances[i][j+1]+=moyennes[i].distance(s);
//							}
//							stopTimeTemp=System.currentTimeMillis();
//							distanceLostTime+=(stopTimeTemp-startTimeTemp);
//							f.write("\tInertie au tour "+(j+1)+" : "+distances[i][j+1]+"\n");
//						}*/
//						
//					}
//					stopTime=System.currentTimeMillis();
//					double distanceTotal=0.0;
//					for(int i=0;i<distances.length;i++){
//						double d=distances[i][0]/(1.0*m[i].length);
//						distanceTotal+=d;
//					}
//					
//					f.write("== Inertie moyenne : "+(distanceTotal/(1.0*m.length))+" ==\n");
//					f.write("== Temps moyen par moyenne : "+((stopTime-startTime-distanceLostTime)/(1.0*m.length))+" ms ==\n");
//					
//				}
//				f.close();
//			}catch(IOException e){}
//			
//	
//		}
//	}

}
