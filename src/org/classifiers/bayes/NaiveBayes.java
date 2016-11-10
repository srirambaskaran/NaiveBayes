package org.classifiers.bayes;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import org.classifiers.models.UnivariateDataPoint;

public class NaiveBayes {
	private ArrayList<UnivariateDataPoint> univariateDataPoints; //list of input data points
	private HashMap<String, Double> priorMean; //created for each class
	private HashMap<String, Double> priorSD; //created for each class
	private HashMap<String, Double> likelihoods;
	private HashMap<String, ArrayList<UnivariateDataPoint>> classWise;
	
	
	public NaiveBayes(ArrayList<UnivariateDataPoint> univariateDataPoints) {
		super();
		this.univariateDataPoints = univariateDataPoints;
		this.priorMean = new HashMap<>();
		this.priorSD = new HashMap<>();
		this.likelihoods = new HashMap<>();
		this.classWise = new HashMap<>();
		loadData();
		train();
		
	}
	
	/**
	 * Divides data into different classes.
	 */
	private void loadData() {
		for(UnivariateDataPoint univariateDataPoint:univariateDataPoints){
			if(!classWise.containsKey(univariateDataPoint.getY())){
				classWise.put(univariateDataPoint.getY(), new ArrayList<>());
			}
			classWise.get(univariateDataPoint.getY()).add(univariateDataPoint);
		}
	}

	/**
	 * Trains the classifier by calculating prior and likelihood.
	 */
	public void train(){
		for(String clazz: classWise.keySet()){
			//Calculating Prior Mean and SD to calculate p(x|C).
			double mean = getMean(classWise.get(clazz));
			double sd = getSD(classWise.get(clazz), mean);
			priorMean.put(clazz, mean);
			priorSD.put(clazz, sd);
			
			//Calculating Likelihood p(C)
			int numClassItems = classWise.get(clazz).size();
			int totalItems = univariateDataPoints.size();
			Double likelihood = ((double)numClassItems)/totalItems;
			likelihoods.put(clazz, likelihood);
		}		
	}
	
	
	public String predict(Double x){
		Double maxProb = Double.NEGATIVE_INFINITY;
		String maxProbClass = "";
		double evidence = calculateEvidence(x);
		for(String clazz:classWise.keySet()){
			double prior = gaussian(x, priorMean.get(clazz), priorSD.get(clazz));
			double likelihood = likelihoods.get(clazz);
			double prob =  prior * likelihood/evidence;
			if(prob > maxProb){
				maxProb = prob;
				maxProbClass = clazz;
			}
		}
		
		return maxProbClass;
	}
	
	/**
	 * Calculates evidence probability. p(x)
	 * @param x
	 * @return
	 */
	private double calculateEvidence(Double x){
		double evidence = 0.0;
		for(String clazz: classWise.keySet()){
			double prob = gaussian(x, priorMean.get(clazz), priorSD.get(clazz));
			evidence += prob * likelihoods.get(clazz);
		}
		return evidence;
	}
	
	/**
	 * Calculates Population Mean
	 * @param points
	 * @return
	 */
	private double getMean(ArrayList<UnivariateDataPoint> points){
		double mean = 0.0;
		for(UnivariateDataPoint point: points){
			mean += point.getX()/points.size();
		}
		return mean;
	}
	
	/**
	 * Calculates Population variance/SD
	 * @param points
	 * @param mean
	 * @return
	 */
	private double getSD(ArrayList<UnivariateDataPoint> points, double mean){
		double sd = 0.0;
		for(UnivariateDataPoint point: points){
			sd += (Math.pow(point.getX(),2)/points.size());
		}
		sd = Math.sqrt(sd - Math.pow(mean, 2));
		return sd;
	}
	
	private double gaussian(double x, double mean, double sd){
		double gaussian = (1/Math.sqrt(2*Math.PI*Math.pow(sd, 2))) * Math.pow(Math.E, -(Math.pow(x-mean, 2)/(2*Math.pow(sd,2))));
		return gaussian;
	}
	
	public double getMean(String clazz){
		return this.priorMean.get(clazz);
	}
	public double getSD(String clazz){
		return this.priorSD.get(clazz);
	}
	
	public Set<String> getClasses(){
		return this.classWise.keySet();
	}
}
