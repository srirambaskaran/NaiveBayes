package org.classifiers.models;

public class UnivariateDataPoint {
	private Double x;
	private String y;
	public UnivariateDataPoint(Double x, String y){
		this.x = x;
		this.y = y;
	}
	public Double getX() {
		return x;
	}
	public void setX(Double x) {
		this.x = x;
	}
	public String getY() {
		return y;
	}
	public void setY(String y) {
		this.y = y;
	}
}
