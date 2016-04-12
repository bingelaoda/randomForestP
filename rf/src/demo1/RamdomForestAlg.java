package demo1;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * This example class trains a Random Forest classifier on a dataset and outputs for 
 * a second dataset.	
 */
public class RamdomForestAlg {
	
  /**
   * Expects two parameters: training file and test file.
   * 
   * @throws Exception	if something goes wrong
   */
	
  public static void main(String[] args) throws Exception {
    // load data
	  String path = FileNameUtil.getPrjPath();
//    BufferedReader br = null;
//    int numFolds = 10;
//    br = new BufferedReader(new FileReader(path+"doc\\maxUtilSvr.csv"));
//    Instances trainData =DataSource.read(path+"doc\\maxUtilSvrTrain.csv");
//    Instances trainData =DataSource.read(path+"doc\\respTimeTrain.csv");
//	  Instances trainData = new Instances(br);
      Instances trainData =DataSource.read(path+"doc\\reltTrain.csv");
      trainData.setClassIndex(trainData.numAttributes() - 1);
      
//      br.close();
//    Instances train = DataSource.read(args[0]);
//    train.setClassIndex(train.numAttributes() - 1);
//  	BufferedReader br = null;
//    int numFolds = 10;k
//    br = new BufferedReader(new FileReader(path+"doc\\maxUtilSvr.csv"));
//    Instances test = new Instances(br);
//    Instances test = DataSource.read(path+"doc\\maxUtilSvrTest.csv");
//    Instances test = DataSource.read(path+"doc\\respTimeTest.csv");
      
    Instances test = DataSource.read(path+"doc\\reltTest.csv");
    test.setClassIndex(test.numAttributes() - 1);
    
    if (!trainData.equalHeaders(test))
      throw new IllegalArgumentException(
	  "Train and test set are not compatible: " + trainData.equalHeadersMsg(test));
    
    // train classifier
    RandomForest rf = new RandomForest();
    rf.setNumTrees(500);
    rf.setBatchSize("100");
    rf.setNumDecimalPlaces(10);
    rf.buildClassifier(trainData);
    
    // output predictions
    System.out.println("# - actual - predicted - error - distribution");
//  int num=0;
    List<Double> realv = new ArrayList<>();
    List<Double> predv = new ArrayList<>();
    for (int i = 0; i < test.numInstances(); i++) {
    	double real = Double.parseDouble(test.instance(i).toString(test.classIndex()));
    	double pred = rf.classifyInstance(test.instance(i));
    	realv.add(real);
    	predv.add(pred);
      
      System.out.println("pred"+pred);
//      double[] dist = rf.distributionForInstance(test.instance(i));

      System.out.print((i+1));
//      System.out.print(" - ");
//      System.out.println(pred);
//      int real = Integer.parseInt(test.instance(i).toString(test.classIndex()));
//      System.out.print(Integer.parseInt(test.instance(i).toString(test.classIndex())));
//      int pred1 = (int) Math.round(pred);
//      System.out.println("    "+real+"   "+pred1);
      
//      if(real!=pred1){
////    	  throw new Exception("在这一次出现错误"+i);
//    	 num++;
//      }
//      System.out.print(" - ");
//      System.out.print(test.classAttribute().value((int) pred));
//      System.out.println(pred);
//      System.out.print(" - ");
//      if (pred != test.instance(i).classValue())
//	System.out.print("yes");
//      else
//	System.out.print("no");
//      System.out.print(" - ");
//      System.out.print(Utils.arrayToString(dist));
      System.out.println();
    }
    computeAccuracy2(realv, predv);
    
  }
  
  
}
