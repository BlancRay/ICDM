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
package tools;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

public class UCR2CSV {
	public static void run(File f, File fout) {
		BufferedReader in = null;
		PrintWriter out = null;
		String line;
		String[] temp;
		boolean firstLine = true;
		try {
			in = new BufferedReader(new FileReader(f));
			out = new PrintWriter(new FileOutputStream(fout), true);

			while ((line = in.readLine()) != null) {
				if (!line.isEmpty()) {
					if(firstLine){
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
					}
					int k = 0;
					while (line.charAt(k) == ' ')
						k++;
					line = line.substring(k);
//					temp = line.split("\\s+");
					temp = line.split(",");
					if(temp[0].isEmpty()||temp[0].equals("0"))
						temp[0]="-1";
					out.print("'"+((int)Math.round(Double.valueOf(temp[0])))+"'");
					for (int j = 1; j < temp.length; j++) {
						out.print(","+temp[j] );
					}
					out.println();
					
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
			out.close();
		}
	}

}
