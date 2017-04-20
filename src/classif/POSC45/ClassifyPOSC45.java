/**
 *    Date : 2008-1-5
 *      by : Yang ZHANG  @ UQ
 *   Email : mokuram@itee.uq.edu.au
 *           zhangyang@nwsuaf.edu.cn
 *    
 * Project : Classification of Data Streams
 *
 * Description:
 *     Building decision tree from positive and unlabeled dataset.
 *     Section 4.2
 *     of the paper: (IN PAGE 9)
 *     
 *     @inproceedings{
 *          author = {Francois Denisa, Remi Gilleronb, Fabien Letouzey},
 *    		title = {Learning from positive and unlabeled examples},
 *    		booktitle = {Theoretical Computer Science 348 },
 *    		year = {2005},
 *    		pages = {70 �C 83},
 *    		publisher = {ELSEVIER},
 *    	}
 *    
 *    This implement is only suitable for binary classification task.
 *    @attribute 'class' {'won','nowin'}
 *        the first class is the POS class, here, POS=won
 *        and the second class is the NEG class, NEG=nowin
 *        
 *    ����ζ���ڱ�ʾ������ѵ��������ͷ����Ʒ��������������б��еĵ�һ�����
 *    
 *    �������ڣ�
 *        ��Ʒ�а�������ֵȱʧ������� ��δ��������
 *        ��Ʒ�а����������ԣ�         ��δ��������
 */

package classif.POSC45;

import weka.core.*;

public class ClassifyPOSC45 extends J48
{
	// the POS dataset
	String strPosFileName;
	Instances posDataset;

	// the UNLABELED dataset
	String strUnFileName;
	Instances unDataset;

	// This is the parameter need to calculate Entropy as described in formula
	// (9) in the paper
	// the static variable only makes the programming easier
	// however, to run multi-instance, or in parallel, is prohibited because of
	// this
	public static int nPosSize;
	public static int nUnSize;
	public static double dDF = 0.5;

	public ClassifyPOSC45(double d)
	{
		dDF = d;
	}

	public ClassifyPOSC45()
	{
		dDF = 0.5; // default value, this is OK when POS:NEG is expected to be
					// 1:1
	}

	public void setDataset(Instances posData, Instances unData)
	{
		posDataset = new Instances(posData);
		unDataset = new Instances(unData);
	}
	public Instances createTrainingData()
	{
		nPosSize = posDataset.numInstances();
		nUnSize = unDataset.numInstances();

		for (int i = 0; i < unDataset.numInstances(); i++)
		{
			Instance sample = unDataset.instance(i);
			sample.setClassValue(1);
		}

		Instances dataset = new Instances(posDataset);
		for (int i = 0; i < nUnSize; i++) {
			dataset.add(unDataset.instance(i));
		}
		return dataset;
	}
	

	/**
	 * Generates the classifier.
	 * 
	 * @param instances
	 *            the data to train the classifier with
	 * @throws Exception
	 *             if classifier can't be built successfully
	 */
	public void buildClassifier(Instances instances) throws Exception
	{
		Instances dataset = createTrainingData();
		super.buildClassifier(dataset);

	}
}
