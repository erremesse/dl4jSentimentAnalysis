package es.unizar.ml4bigdata.dl4j.tests;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.logging.Logger;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.apache.commons.math3.util.Pair;
import org.deeplearning4j.text.sentenceiterator.BaseSentenceIterator;

//import javafx.util.Pair;

public class CsvIterator extends BaseSentenceIterator{
	private static Logger log = Logger.getLogger(CsvIterator.class.getName());
	
	private String fileInput;
	private LineIterator iter;
	private ParseCsvPreprocessor parseCsvPreprocessor;
	
	public CsvIterator(String fileInput, ParseCsvPreprocessor parseCsvProc){
		this.setFileInput(fileInput);
		this.setParseCsvPreprocessor(parseCsvProc);
		readFile();
	}
	
	private void readFile() {
		try{
			BufferedInputStream file = new BufferedInputStream(new FileInputStream(new File(fileInput)));
			this.iter = IOUtils.lineIterator(file, "UTF-8");
		}catch(IOException ioex){
			log.info("File not read properly");
		}
		
	}

    public Pair<String[], double[]> nextSentenceParsedCsv() {
        String line = this.iter.nextLine();
        if (this.parseCsvPreprocessor != null) {
            return this.parseCsvPreprocessor.preProcess(line);
        } else {
            throw new IllegalStateException("ParseCsvPreprocessor not defined.");
        }
}
    
	@Override
	public String nextSentence() {
		String line = this.iter.nextLine();
        if(this.preProcessor != null) {
            line = this.preProcessor.preProcess(line);
        }
        return line;
	}

	@Override
	public boolean hasNext() {
		return this.iter.hasNext();
	}

	@Override
	public void reset() {
		// TODO Auto-generated method stub
	}

	public String getFileInput() {
		return fileInput;
	}

	public void setFileInput(String fileInput) {
		this.fileInput = fileInput;
	}

	public ParseCsvPreprocessor getParseCsvPreprocessor() {
		return parseCsvPreprocessor;
	}

	public void setParseCsvPreprocessor(ParseCsvPreprocessor parseCsvPreprocessor) {
		this.parseCsvPreprocessor = parseCsvPreprocessor;
	}

}
