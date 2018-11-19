package es.unizar.ml4bigdata.dl4j.tests;

import org.apache.commons.math3.util.Pair;
//import org.apache.commons.lang3.tuple.Pair;

//import javafx.util.Pair; ORACLE (No está en OpenJDK)

public class ParseCsvPreprocessor {
	private String delimiter;
	
	public ParseCsvPreprocessor(String delimiter){
		this.delimiter = delimiter;
	}
	
	public Pair<String[], double[]> preProcess(String s) {
        String[] sentenceAndLabel = s.split(delimiter);
        String[] words = sentenceAndLabel[0].toLowerCase().split(" ");
        
        double label = Double.parseDouble(sentenceAndLabel[1]); //Label is 0 (negative) or 1 (positive)
        double[] labels = {label, 1 - label}; //Right now it uses just positive[1 0] and negative[0 1]	
        									  //TODO Take into account more classes (??)
        
        return new Pair<>(words, labels);
	}
}
