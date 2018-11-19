package es.unizar.sentiment.analysis;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.opencsv.CSVReader;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;

public class SentimentIterator implements DataSetIterator {
	private String dataPath;
	private Word2Vec word2Vec;
	private int vectorSize;
	private int batchSize;
	private int truncateLength;
	private TokenizerFactory tokenizerFactory;
	private BiMap<Integer,String> labels = HashBiMap.create(); //https://www.baeldung.com/guava-bimap
	private int nLabels;
	private boolean train; //TODO Implementar el caso en el que sólo tenemos un conjunto de datos y hay que partirlo en TRAIN vs TEST --> CrossValidation

	private CSVReader reader;
	
    private int cursor = 0;
    private int totalDocs = 0;

    private static Logger log = Logger.getLogger(SentimentIterator.class.getName());
    
    //TODO Implement it with BUILDER pattern
    public SentimentIterator(String dataPath,
            					Word2Vec word2Vec,
            					int batchSize,
            					int truncateLength,
            					TokenizerFactory tokenizerFactory,
            					String labelsPath) throws IOException{
    	this.dataPath = dataPath;
    	this.word2Vec = word2Vec;
    	this.vectorSize = word2Vec.getLayerSize();
    	this.batchSize = batchSize;
    	this.truncateLength = truncateLength;
    	this.tokenizerFactory = tokenizerFactory;
    	try (BufferedReader brCategories = new BufferedReader(new FileReader(new File(labelsPath)))) {
    		String line = "";
    		while ((line = brCategories.readLine()) != null) {
    			String[] content = line.split(",");
    			labels.put(Integer.parseInt(content[0]), content[1]);
    		}
    		brCategories.close();
    	}catch(IOException e) {
    		log.severe("Exception while reading labels file :" + e.getMessage());
    		throw e;
    	}
    	this.nLabels = labels.size();
    	
    	try {
    		this.reader = new CSVReader(new FileReader(dataPath));
    		CSVReader readerAux = new CSVReader(new FileReader(dataPath));	
    		this.totalDocs = readerAux.readAll().size();
    		readerAux.close();
    	}catch(IOException e) {
    		System.out.println("Exception while reading data file :" + e.getMessage());
    		throw e;
    	}
    }
    
	@Override
	public boolean hasNext() {
		return this.cursor < this.totalDocs;
	}

	@Override
	public DataSet next() {
		return next(batchSize);
	}

	@Override
	public DataSet next(int num) {
		if (cursor >= this.totalDocs) throw new NoSuchElementException();
        try {
            return nextBatch(num);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
	}
	
	private DataSet nextBatch(int num) throws IOException {
        List<String> docs = new ArrayList<>(num);
        int[] category = new int[num];
        
        //Load batch
        String[] line;
        for(int i=0; i<num && this.cursor<this.totalDocs; i++) {
        	//reader.skip(this.cursor+1);
        	if((line = reader.readNext()) != null ) {
        		if(labels.inverse().get(line[1]) != null) {
        			//System.out.println("Record " + i + " -> Text = " + line[0] + " && Label = " + line[1]);
                    docs.add(line[0]);
                    category[i] = labels.inverse().get(line[1]); //Integer.parseInt(line[1]);	
        		} else if(labels.inverse().get(line[0]) != null){
        			//System.out.println("Record " + i + " -> Text = " + line[1] + " && Label = " + line[0]);
                    docs.add(line[1]);
                    category[i] = labels.inverse().get(line[0]); 
        		} else {
                	log.info("[WARN]Didn't get a correct label in cursor point = " + cursor + "\n...Skipping line...");
                	i--;
                }
        		cursor++;
        	}
        }				
				
        
        //Tokenize batch and filter docs (get rid of words that are not present in word2Vec)
        List<List<String>> docsFilter = new ArrayList<>(docs.size());
        int maxLength = 0;
        for(String doc:docs) {
        	List<String> original = tokenizerFactory.create(doc).getTokens();
        	//TODO What if we wanna use UNK words?? Is UNK vectorized?
        	//EQUAL TO: for(String t:original){...}
        	List<String> docFilter = original.stream()
        			.filter(t -> word2Vec.getWordVector(t) != null)
        			.collect(Collectors.toList()); //Collectors.toCollection(ArrayList:new)
        	docsFilter.add(docFilter);
        	maxLength = Math.max(maxLength, docFilter.size());
        } 
        if(maxLength > truncateLength) maxLength = truncateLength;	
         
        
        //Create data for training
        INDArray features = Nd4j.create(docs.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(docs.size(), nLabels, maxLength);
        INDArray featuresMask = Nd4j.create(docs.size(), maxLength);
        INDArray labelsMask = Nd4j.create(docs.size(), maxLength);
        for(int i=0; i<docs.size(); i++){
        	//Set doc features
        	List<String> docTokens = docsFilter.get(i); //Vectorized tokens of the document i
        	for(int j=0; j<docTokens.size() && j<maxLength; j++){
        		String token = docTokens.get(j);
        		INDArray tokenVector = word2Vec.getWordVectorMatrix(token);
        		INDArrayIndex[] indexFeat = new INDArrayIndex[]{point(i), all(), point(j)}; //Posición de la palabra en el documento (todo el vector que representa la palabra)
        		features.put(indexFeat, tokenVector); //features.put(index, value)
        		featuresMask.putScalar(new int[]{i,j},1.0);
        	} 
        	
        	//Set doc label
        	int label = category[i];
        	int lastIdx = Math.min(docTokens.size(), maxLength); 
        	int[] indexLabel = new int[]{i, label, lastIdx-1}; //Posición de la última palabra del documento
        	labels.putScalar(indexLabel, 1.0); //labels.put(index,value)
        	labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);
        }
        
		DataSet ds = new DataSet(features, labels, featuresMask, labelsMask);
		return ds;
	}//end .nextBatch()

	@Override
	public int totalExamples() {
		return this.totalDocs;
	}

	@Override
	public int inputColumns() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int totalOutcomes() {
		return this.nLabels;
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void reset() {
		try {
			this.reader.close();
			this.reader = new CSVReader(new FileReader(dataPath));
		}catch(IOException ioex) {
			System.out.println("IOException while trying to restart CSV reader.");
			ioex.printStackTrace();
		}
		this.cursor = 0;
	}

	@Override
	public int batch() {
		return this.batchSize;
	}

	@Override
	public int cursor() {
		return this.cursor;
	}

	@Override
	public int numExamples() {
		return this.totalDocs;
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		// TODO Auto-generated method stub
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public List<String> getLabels() {
		Set<String> values = this.labels.values();
		return new ArrayList<>(values);
	}

}
