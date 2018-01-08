package classif;

import java.io.File;
import java.io.FileFilter;
import java.util.ArrayList;
import java.util.Arrays;
import org.apache.commons.math3.random.RandomDataGenerator;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import classif.ExperimentsLauncher;
import classif.pu.CombinetoPN;
import tools.UCR2CSV;
public class PULauncher {
	public static String username = "xulei";
	long startTime;
	long endTime;
	long duration;
	private static String datasetsDir = "./PUDATA/";
//	private static String datasetsDir = "./UCR_TS_Archive_2015/";
	private static String saveoutputDir = "./save/";

	public static void main(String[] args) {
		String dataname="CBF";
		datasetsDir = "./PUDATA/";
//		String dataname=args[0];
//		datasetsDir=args[1];
		File repSave = new File(saveoutputDir);

		// datasets folder
		File rep = new File(datasetsDir);
		File[] listData = rep.listFiles(new FileFilter() {
			@Override
			public boolean accept(File pathname) {
				return pathname.isDirectory();
			}
		});
		Arrays.sort(listData);

		for (File dataRep : listData) {
			// only process GunPoint dataset to illustrates
//			if (dataRep.getName().equals("50words")||dataRep.getName().equals("Phoneme")||dataRep.getName().equals("DiatomSizeReduction"))
			// continue;
			if (!dataRep.getName().equals(dataname) || dataRep.getName().equals("ElectricDevices"))
				continue;
			System.out.println("processing: " + dataRep.getName());
			Instances[] data = readTrainAndTest(dataRep.getName());
			Instances train=new Instances(data[0],0);
			Instances test=new Instances(data[0],0);
			int nbpos=data[0].numInstances()/2;
			for (int i = 0; i < data.length; i++) {
				RandomDataGenerator randGen=new RandomDataGenerator();
				int[] selected = randGen.nextPermutation(data[i].numInstances(), data[i].numInstances()/2);
				for (int j = 0; j < selected.length; j++) {
					train.add(data[i].instance(j));
					data[i].delete(j);
				}
				data[i]=new Instances(data[i]);
			}
			for (int i = 0; i < data.length; i++) {
				for (int j = 0; j < data[i].numInstances(); j++) {
					test.add(data[i].instance(j));
				}
			}
			/**
			 * 调用launchPU
			 */
			new ExperimentsLauncher(repSave, train, test, dataRep.getName(), 10, nbpos).launchPOSC45();
			
			
		}
	}

	public static Instances[] readTrainAndTest(String name) {
		File pFile = new File(datasetsDir + name + "/" + name + "_P");
		File nFile = new File(datasetsDir + name + "/" + name + "_N");
		File trainFile = new File(datasetsDir + name + "/" + name + "_TRAIN");
		File testFile = new File(datasetsDir + name + "/" + name + "_TEST");
		if (!new File(pFile.getAbsolutePath() + ".csv").exists()||!new File(nFile.getAbsolutePath() + ".csv").exists()) {
		CombinetoPN.run(trainFile, new File(pFile.getAbsolutePath()), new File(nFile.getAbsolutePath()));
		CombinetoPN.run(testFile, new File(pFile.getAbsolutePath()), new File(nFile.getAbsolutePath()));
		}

		if (!new File(pFile.getAbsolutePath() + ".csv").exists()) {
			UCR2CSV.run(pFile, new File(pFile.getAbsolutePath() + ".csv"));
		}
		pFile = new File(pFile.getAbsolutePath() + ".csv");
		if (!new File(nFile.getAbsolutePath() + ".csv").exists()) {
			UCR2CSV.run(nFile, new File(nFile.getAbsolutePath() + ".csv"));
		}
		nFile = new File(nFile.getAbsolutePath() + ".csv");

		
		CSVLoader loader = new CSVLoader();
		Instances pDataset = null;
		Instances nDataset = null;

		try {
			loader.setFile(pFile);
			loader.setNominalAttributes("first");
			pDataset = loader.getDataSet();
	        ArrayList<String> values =new ArrayList<String>();
	        values.add("1.0");
	        values.add("-1.0");
			pDataset.insertAttributeAt(new Attribute("clas",values), 0);
			for (int i = 0; i < pDataset.numInstances(); i++) {
//				System.out.println(pDataset.instance(i).toString(1));
				pDataset.instance(i).setValue(0, pDataset.instance(i).toString(1));
			}
			pDataset.deleteAttributeAt(1);
			pDataset.setClassIndex(0);

			loader.setFile(nFile);
			loader.setNominalAttributes("first");
			nDataset = loader.getDataSet();
			nDataset.insertAttributeAt(new Attribute("clas",values), 0);
			for (int i = 0; i < nDataset.numInstances(); i++) {
//				System.out.println(nDataset.instance(i).toString(1));
				nDataset.instance(i).setValue(0, nDataset.instance(i).toString(1));
			}
			nDataset.deleteAttributeAt(1);
			nDataset.setClassIndex(0);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return new Instances[] { pDataset, nDataset };
	}

}
