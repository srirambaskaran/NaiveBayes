package org.main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;

import org.classifiers.bayes.NaiveBayes;
import org.classifiers.models.UnivariateDataPoint;

/**
 * Input to the Univariate Naive Bayes Classifier is a Comma separated list of values, where each 
 * row contains a X (double) and its class. This works for Binary classification.
 * @author Sriram
 *
 */
public class MainClass {
	
	public static void main(String[] args) throws Exception {
		String inputFile = "test-files/height-smurf-troll.csv";
		
		BufferedReader reader = new BufferedReader(new FileReader(inputFile));
		String line = "";
		ArrayList<UnivariateDataPoint> points = new ArrayList<>();
		while((line = reader.readLine())!=null){
			String[] tokens = line.split(",");
			points.add(new UnivariateDataPoint(Double.parseDouble(tokens[0]),tokens[1]));
		}
		reader.close();
		Collections.shuffle(points);
		int k = 15;
		ArrayList<UnivariateDataPoint> trainingSet = new ArrayList<>();
		while(k!=0){
			trainingSet.add(points.get(k-1));
			k--;
		}
		ArrayList<UnivariateDataPoint> validationSet = new ArrayList<>();
		k=15;
		while(k<points.size()){
			validationSet.add(points.get(k++));
		}
		
		NaiveBayes bayes = new NaiveBayes(points);
		Set<String> classes = bayes.getClasses();
		for(String clazz: classes){
			System.out.println(clazz+"\tMean: "+bayes.getMean(clazz)+"\tSD: "+bayes.getSD(clazz));
		}
		System.out.println("Training Size: "+trainingSet.size()+"\tValidation Set: "+validationSet.size());
		System.out.println("====================================================");
		System.out.println("Point\t|\tActual\t|\tPredicted");
		int tp = 0;
		int fp = 0;
		for(UnivariateDataPoint point: validationSet){
			String prediction = bayes.predict(point.getX());
			System.out.println(point.getX()+"\t|\t"+point.getY()+"\t|\t"+prediction);
			if(prediction.contentEquals(point.getY()))
				tp++;
			else
				fp++;
		}
		System.out.println("====================================================");
		System.out.println("Precision: "+((double)tp)/(tp+fp));
		
	}
}
