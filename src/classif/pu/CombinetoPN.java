package classif.pu;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

public class CombinetoPN {
	public static void run(File f, File p,File n) {
		BufferedReader in = null;
		PrintWriter out_p = null;
		PrintWriter out_n = null;
		String line;
		String[] temp;
		try {
			in = new BufferedReader(new FileReader(f));
			out_p = new PrintWriter(new FileOutputStream(p,true), true);
			out_n = new PrintWriter(new FileOutputStream(n,true), true);

			while ((line = in.readLine()) != null) {
				if (!line.isEmpty()) {
					/*if(firstLine){
						int k = 0;
						while (line.charAt(k) == ' ')
							k++;
						line = line.substring(k);
//						temp = line.split("\\s+");
						temp = line.split(",");
						out.print("class");
						for (int j = 1; j < temp.length; j++) {
							out.print(",t"+(j-1));
						}
						out.println();
						firstLine=false;
					}*/
					int k = 0;
					while (line.charAt(k) == ' ')
						k++;
					line = line.substring(k);
//					temp = line.split("\\s+");
					temp = line.split(",");
					/**
					 * for PU
					 */
					if(temp[0].equals("1")){
						out_p.print(((int) Math.round(Double.valueOf(temp[0]))));
						for (int j = 1; j < temp.length; j++) {
							out_p.print("," + temp[j]);
						}
						out_p.println();
					} else {
						temp[0]="-1";
						out_n.print(((int) Math.round(Double.valueOf(temp[0]))));
						for (int j = 1; j < temp.length; j++) {
							out_n.print("," + temp[j]);
						}
						out_n.println();

					}
				}
			}

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				in.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			out_p.close();
			out_n.close();
		}
	}

}