package foundation.SurModel;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.encog.ConsoleStatusReportable;
import org.encog.Encog;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.encog.util.simple.EncogUtility;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import control.JssBRSCtrlEngine;
import foundation.KNN.TrainKNN;
import foundation.fileUtil.FileNameUtil;

public class MaxUtilSvr {
	public static  NormalizationHelper helper = new NormalizationHelper();
//	public static File file = new File(FileNameUtil.getPrjPath()+"demo\\jssbrs\\runFiles\\resultFile\\test.csv"); 
	public static File file  = new File("E:\\brs.csv");
	public static MLRegression bestMethod = null;
	
	public void run() {
//			File irisFile = new File(FileNameUtil.getPrjPath()+"demo\\jssbrs\\runFiles\\resultFile", "brs1.csv");
			File irisFile = new File("E:\\","train.csv");
			// Define the format of the data file.
			// This area will change, depending on the columns and 
			// format of the file that you are trying to model.
			VersatileDataSource source = new CSVDataSource(irisFile, false,
					CSVFormat.DECIMAL_POINT);
			VersatileMLDataSet data = new VersatileMLDataSet(source);
			data.defineSourceColumn("1", 0, ColumnType.continuous);
			data.defineSourceColumn("2", 1, ColumnType.continuous);
			data.defineSourceColumn("3", 2, ColumnType.continuous);
			data.defineSourceColumn("4", 3, ColumnType.continuous);
			data.defineSourceColumn("5", 4, ColumnType.continuous);
			data.defineSourceColumn("6", 5, ColumnType.continuous);
			data.defineSourceColumn("7", 6, ColumnType.continuous);
			data.defineSourceColumn("8", 7, ColumnType.continuous);
			data.defineSourceColumn("9", 8, ColumnType.continuous);
			data.defineSourceColumn("10", 9, ColumnType.continuous);
			data.defineSourceColumn("11", 10, ColumnType.continuous);
			data.defineSourceColumn("12", 11, ColumnType.continuous);
			data.defineSourceColumn("13", 12, ColumnType.continuous);
			data.defineSourceColumn("14", 13, ColumnType.continuous);
			// Define the column that we are trying to predict.
			ColumnDefinition outputColumn = data.defineSourceColumn("species", 14,
					ColumnType.continuous);
			
			// Analyze the data, determine the min/max/mean/sd of every column.
			data.analyze();
			
			// Map the prediction column to the output of the model, and all
			// other columns to the input.
			data.defineSingleOutputOthersInput(outputColumn);
			
			// Create feedforward neural network as the model type. MLMethodFactory.TYPE_FEEDFORWARD.
			// You could also other model types, such as:
			// MLMethodFactory.SVM:  Support Vector Machine (SVM)
			// MLMethodFactory.TYPE_RBFNETWORK: RBF Neural Network
			// MLMethodFactor.TYPE_NEAT: NEAT Neural Network
			// MLMethodFactor.TYPE_PNN: Probabilistic Neural Network
			EncogModel model = new EncogModel(data);
			model.selectMethod(data, MLMethodFactory.TYPE_SVM);
			
			// Send any output to the console.
			model.setReport(new ConsoleStatusReportable());
			
			// Now normalize the data.  Encog will automatically determine the correct normalization
			// type based on the model you chose in the last step.
			data.normalize();
			
			// Hold back some data for a final validation.
			// Shuffle the data into a random ordering.
			// Use a seed of 1001 so that we always use the same holdback and will get more consistent results.
			model.holdBackValidation(0.3, true, 1001);
			
			// Choose whatever is the default training type for this model.
			model.selectTrainingType(data);
			
			// Use a 5-fold cross-validated train.  Return the best method found.
			bestMethod= (MLRegression)model.crossvalidate(5, true);

			// Display the training and validation errors.
//			System.out.println( "Training error: " + EncogUtility.calculateRegressionError(bestMethod, model.getTrainingDataset()));
//			System.out.println( "Validation error: " + EncogUtility.calculateRegressionError(bestMethod, model.getValidationDataset()));
			
			// Display our normalization parameters.
			helper = data.getNormHelper();
//			System.out.println(helper.toString());
			
			// Display the final model.
//			System.out.println("Final model: " + bestMethod);
		} 
	
	public String[] solveMaxUtilSrv(ReadCSV csv,int size){
		String[] line = new String[14];
		String[] st = new String[size];
		MLData input = MaxUtilSvr.helper.allocateInputVector();
		String irisChosen = "";
		int m=0;
		while (csv.next()) {
//			StringBuilder result = new StringBuilder();
			for (int i = 0; i < 14; i++) {
				line[i] = csv.get(i);
			}

			MaxUtilSvr.helper.normalizeInputVector(line, input.getData(), false);
			MLData output = MaxUtilSvr.bestMethod.compute(input);
			irisChosen = MaxUtilSvr.helper.denormalizeOutputVectorToString(output)[0];
			st[m] = irisChosen;
			m++;
//			result.append(Arrays.toString(line));
//			result.append(" -> predicted: ");
//			result.append(irisChosen);
//			result.append(")");

//			System.out.println(result.toString());
		}

		Encog.getInstance().shutdown();
	
		return st ;
	}

	public  void createCSV(List<StringBuffer> code){
		
		String[] strs = new String[code.size()];
		for(int i=0;i<code.size();i++){
			strs[i] = code.get(i).toString();
		}
		 
        Writer writer = null;
		try {
			writer = new FileWriter(MaxUtilSvr.file);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}  
        CSVWriter csvWriter = new CSVWriter(writer, ',');  
        csvWriter.writeNext(strs);
        try {
			csvWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		} 
        
	}
	
	public static void main(String[] args) {
/*		List<StringBuffer> strb = new ArrayList<>();
		StringBuffer st = new StringBuffer();
		st.append(11.5035);
		strb.add(st);
		StringBuffer st1 = new StringBuffer();
		st1.append(11.86025);
		strb.add(st1);
		StringBuffer st2 = new StringBuffer();
		st2.append(7.8185);
		strb.add(st2);
		StringBuffer st3 = new StringBuffer();
		st3.append(7.9362497);
		strb.add(st3);
		StringBuffer st4 = new StringBuffer();
		st4.append(1);
		strb.add(st4);
		StringBuffer st5 = new StringBuffer();
		st5.append(2);
		strb.add(st5);
		StringBuffer st6 = new StringBuffer();
		st6.append(2);
		strb.add(st6);
		StringBuffer st7 = new StringBuffer();
		st7.append(3);
		strb.add(st7);
		StringBuffer st8 = new StringBuffer();
		st8.append(1);
		strb.add(st8);
		StringBuffer st9 = new StringBuffer();
		st9.append(1);
		strb.add(st9);
		StringBuffer st10 = new StringBuffer();
		st10.append(2);
		strb.add(st10);
		StringBuffer st11 = new StringBuffer();
		st11.append(3);
		strb.add(st11);
		StringBuffer st12 = new StringBuffer();
		st12.append(3);
		strb.add(st12);
		StringBuffer st13 = new StringBuffer();
		st13.append(4);
		strb.add(st13);
		
		new MaxUtilSvr().createCSV(strb);*/
		
		try {
			File file = new File("e:\\predRelt.csv");
			List<Double> pred = readCSV(file);
			File file1 = new File("e:\\realRelt.csv");
			List<Double> real = readCSV1(file1);
			
			for(int i=0;i<real.size();i++){
				System.out.print(real.get(i)+"  ");
				System.out.println(pred.get(i));
			}
			computeAccuracy2(real,pred);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		/*MaxUtilSvr prg = new MaxUtilSvr();
		prg.run();
		File delete = new File("E:\\predRelt.csv");
		if(delete.exists()){
			delete.delete();
		}
		File irisFile1 = new File("E:", "test1.csv");
		ReadCSV csv = new ReadCSV(irisFile1, false, CSVFormat.DECIMAL_POINT);
		String[] pred = prg.solveMaxUtilSrv(csv,200);
		List<List<StringBuffer>> code = new ArrayList<List<StringBuffer>>();
		for(int i=0;i<pred.length;i++){
			List<StringBuffer> stf = new ArrayList<>();
			StringBuffer sf = new StringBuffer();
			sf.append(pred[i]);
			stf.add(sf);
			code.add(stf);
		}
		TrainKNN.creaCSV1(code);
		
		System.out.println(pred);*/
	}
	
	public static void computeAccuracy2(List<Double> realValue,List<Double> predValue){
		double mse = 0;
		double variance = 0;
		double mean = 0;
		for (int i = 0; i < realValue.size(); i++) {
			mse += (realValue.get(i) - predValue.get(i)) * (realValue.get(i) - predValue.get(i));
			mean += realValue.get(i);
		}
		mean = mean / realValue.size();
		for (int i = 0; i < realValue.size(); i++) {
			variance += (realValue.get(i) - mean) * (realValue.get(i) - mean);
		}
		double accuracy = 1 - mse / variance;
		
		System.out.println("Accuracy"+accuracy);
		
		
		/*HashMap<Double,Double> hashMap = new HashMap<Double, Double>();
		for(int i=0;i<realValue.size();i++){			
				hashMap.put(realValue.get(i), predValue.get(i));
		}
		Collections.sort(realValue);*/
		
		/*for(int i=0;i<realValue.size();i++){
			rankPredValue.add(hashMap.get(realValue.get(i)));
		}*/
//		for(double b:rankPredValue){
//			System.out.println(b);
//		}
		int num1=0;
		int num2=0;
		for(int i=0;i<realValue.size();i++){
			for(int j=i+1;j<realValue.size();j++){
			if((predValue.get(i)<predValue.get(j)&&realValue.get(i)<realValue.get(j))
					||(predValue.get(i)>predValue.get(j)&&realValue.get(i)>realValue.get(j))
					||(predValue.get(i)==predValue.get(j)&&realValue.get(i)==realValue.get(j))){
				num1++;
			}
			num2++;
			}
		}
		
		System.out.println("RP"+1.0*num1/num2);
	}
	
	public static void computeError(List<Double> realValue,List<Double> predValue){
		
		List<Double> rankPredValue = new ArrayList<>();
		double molecule = 0;
		double denominator = 0;
		double mean = 0;
		for (int i = 0; i < realValue.size(); i++) {
			molecule += (realValue.get(i) - predValue.get(i)) * (realValue.get(i) - predValue.get(i));
			mean += realValue.get(i);
		}
		mean = mean / realValue.size();
		for (int i = 0; i < realValue.size(); i++) {
			denominator += (realValue.get(i) - mean) * (realValue.get(i) - mean);
		}
		double errorRate = 1 - molecule / denominator;
		
		System.out.println("误差"+errorRate);
		
		
		HashMap<Double,Double> hashMap = new HashMap<Double, Double>();
		for(int i=0;i<realValue.size();i++){			
				hashMap.put(realValue.get(i), predValue.get(i));
		}
		Collections.sort(realValue);
		System.out.println(realValue.size());
		System.out.println(predValue.size());
		System.out.println("+++"+hashMap.size());
		for(int i=0;i<realValue.size();i++){
//			System.out.println("  "+predValue.indexOf(hashMap.get(realValue.get(i))));
			rankPredValue.add(hashMap.get(realValue.get(i)));
		}
		System.out.println(rankPredValue.size());
		for(double b:rankPredValue){
			System.out.println(b);
		}
		int num1=0;
		for(int i=0;i<rankPredValue.size()-1;i++){
			if(rankPredValue.get(i)-rankPredValue.get(i+1)>0.000001){
				num1++;
			}
		}
		
		System.out.println("num"+num1);

	}
	
	 public static List<Double> readCSV(File file) throws Exception {  
		 List<Double> value = new ArrayList<>();
//		 File file = new File("e:\\respTime.csv");  
	        FileReader fReader = new FileReader(file);  
	        CSVReader csvReader = new CSVReader(fReader);  
	        String[] strs = csvReader.readNext();  
	        if(strs != null && strs.length > 0){  
	            for(String str : strs)  
	                if(null != str && !str.equals("")){  
	                    System.out.println(str + " , "); 
	                    value.add(Double.parseDouble(str));
	                }
//	            System.out.println("\n---------------");  
	        }  
	        List<String[]> list = csvReader.readAll();  
	        for(String[] ss : list){  
	            for(String s : ss)  
	                if(null != s && !s.equals("")){  
	                	value.add(Double.parseDouble(s));
	                    System.out.print(s + " , ");
	                }
	            System.out.println();  
	        }  
	        csvReader.close();
	        return value;
	    } 
	 public static List<Double> readCSV1(File file) throws Exception {  
		 List<Double> value = new ArrayList<>();
//		 File file = new File("e:\\respTime.csv");  
	        FileReader fReader = new FileReader(file);  
	        CSVReader csvReader = new CSVReader(fReader);  
	        String[] strs = csvReader.readNext();  
	        if(strs != null && strs.length > 0){  
	            for(String str : strs)  
	                if(null != str && !str.equals("")){  
	                    System.out.println(str + " , ");
	                    value.add(Double.parseDouble(str));
	                }
//	            System.out.println("\n---------------");  
	        }  
	        List<String[]> list = csvReader.readAll();  
	        for(String[] ss : list){  
	            for(String s : ss)  
	                if(null != s && !s.equals("")){  
	                	value.add(Double.parseDouble(s));
	                    System.out.print(s + " , ");
	                }
	            System.out.println();  
	        }  
	        csvReader.close();
	        return value;
	    }  
	  

}
