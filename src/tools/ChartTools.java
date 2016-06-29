package tools;

import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import org.jfree.chart.JFreeChart;

import com.itextpdf.awt.DefaultFontMapper;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Rectangle;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfTemplate;
import com.itextpdf.text.pdf.PdfWriter;

public class ChartTools {
	public static void saveChartAsPDF(JFreeChart chart, File f) {
//		int width = 840, height = 480;
		int width = 640, height = 480;
		OutputStream out;
		try {
			out = new BufferedOutputStream(new FileOutputStream(f));

			Rectangle pagesize = new Rectangle(width, height);
			com.itextpdf.text.Document document = new com.itextpdf.text.Document(pagesize);
			PdfWriter writer = PdfWriter.getInstance(document, out);
			document.addAuthor("Francois Petitjean");
//			document.addSubject("Plot sequences");
			document.open();
			PdfContentByte cb = writer.getDirectContent();
			PdfTemplate tp = cb.createTemplate(width, height);
			Graphics2D g2 = tp.createGraphics(width, height, new DefaultFontMapper());
			Rectangle2D r2D = new Rectangle2D.Double(0, 0, width, height);
			chart.draw(g2, r2D, null);
			g2.dispose();
			cb.addTemplate(tp, 0, 0);
			document.close();

			out.close();

		} catch (DocumentException de) {
			System.err.println(de.getMessage());
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	
	public static void saveChartAsPDF(JFreeChart chart, File f,int width,int height) {
		OutputStream out;
		try {
			out = new BufferedOutputStream(new FileOutputStream(f));

			Rectangle pagesize = new Rectangle(width, height);
			com.itextpdf.text.Document document = new com.itextpdf.text.Document(pagesize);
			PdfWriter writer = PdfWriter.getInstance(document, out);
			document.addAuthor("Francois Petitjean");
//			document.addSubject("Plot sequences");
			document.open();
			PdfContentByte cb = writer.getDirectContent();
			PdfTemplate tp = cb.createTemplate(width, height);
			Graphics2D g2 = tp.createGraphics(width, height, new DefaultFontMapper());
			Rectangle2D r2D = new Rectangle2D.Double(0, 0, width, height);
			chart.draw(g2, r2D, null);
			g2.dispose();
			cb.addTemplate(tp, 0, 0);
			document.close();

			out.close();

		} catch (DocumentException de) {
			System.err.println(de.getMessage());
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
}
