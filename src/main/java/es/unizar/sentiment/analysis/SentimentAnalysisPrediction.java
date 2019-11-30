package es.unizar.sentiment.analysis;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class SentimentAnalysisPrediction {

	private static final String WORD2VEC_MODEL_PATH = "./src/main/resources/word2vec/ES/SBW-vectors-300-min5.bin.gz"; //"./src/main/resources/word2vec/EN/raw_eng_model.zip";
	private static final String LABELS_DESCRIPTION_PATH = "./src/main/resources/data/ES/train/labels5"; //"./src/main/resources/data/EN/labelsEN";
	private static final String SENTIMENT_MODEL_PATH = "./src/main/resources/models/ES/ITA/model"; //"./src/main/resources/models/EN/model";
	private static final String PREDICTIONSET_PATH = "./src/main/resources/data/ES/predictions";
	    
    private static final int truncateLength = 50; //Same as in training
    
    private static Word2Vec word2Vec;
    private static TokenizerFactory tokenizerFactory;
    private static int vectorSize;
	
	private static Logger log = Logger.getLogger(SentimentAnalysisTrain.class.getName());
	
	public static void main(String[] args) {
		String textP = "Este es un texto aleatorio muy muy bonito :)";
		String textN = "Sin embargo, este otro texto no es nada bonito y todo lo contrario de positivo...";
		String text0 = "Y este yo diría que es un texto normal y corriente, no?";
		String text = text0;
		
        //Load Sentiment model
		MultiLayerNetwork model = null;
		try {
			model = ModelSerializer.restoreMultiLayerNetwork(SENTIMENT_MODEL_PATH, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		word2Vec = WordVectorSerializer.readWord2VecModel(WORD2VEC_MODEL_PATH);  
		vectorSize = word2Vec.getLayerSize();
        //TODO Improve tokenizer/preprocessor
        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
		
        System.out.println("\n\n-------------------------------");
        System.out.println("Text: \n" + text);
        String pred1 = getPredictionLastWord(text, model);
        String pred2 = getPredictionSequence(text, model);
        System.out.println("Prediction from last word => " + pred1 + " vs. Prediction from the whole sequence => " + pred2);
    
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        List<String> texts = Arrays.asList(textP, textN, text0);
        Map<String, String> predictions = getPrediction(texts, model);
        System.out.println("\n\n-------------------------------");
        for(Entry<String, String> p:predictions.entrySet()) {
        	System.out.println(p.getKey() + " ---> " + p.getValue());
        }
	}
	
    /**
     * Used in prediction to convert a String to a features INDArray that can be passed to the network output method
     *
     * @param text Text to vectorize
     * @return Features array for the given input String
     */
    public static INDArray loadFeaturesFromString(String text){
    	List<String> tokens = tokenizerFactory.create(text).getTokens();
    	//TODO What if we wanna use UNK words?? Is UNK vectorized?
    	//EQUAL TO: for(String t:original){...}
    	List<String> docFilter = tokens.stream()
    			.filter(t -> word2Vec.getWordVector(t) != null)
    			.collect(Collectors.toList()); //Collectors.toCollection(ArrayList:new)
        int maxLength = docFilter.size();
        if(docFilter.size() > truncateLength) maxLength = truncateLength;
        
        //Create features array
    	INDArray features = Nd4j.create(1, vectorSize, maxLength);
    	//INDArray featuresMask = Nd4j.create(1, maxLength);
    	for(int j=0; j<docFilter.size() && j<maxLength; j++){
    		String token = docFilter.get(j);
    		INDArray tokenVector = word2Vec.getWordVectorMatrix(token);
    		INDArrayIndex[] indexFeat = new INDArrayIndex[]{point(0), all(), point(j)}; //Posición de la palabra en el documento (todo el vector que representa la palabra)
    		features.put(indexFeat, tokenVector); //features.put(index, value)
    		//featuresMask.putScalar(new int[]{0,j},1.0);
    	} 
		return features;
    }
    
    public static String getPredictionLastWord(String text, MultiLayerNetwork model) {
        INDArray features = loadFeaturesFromString(text);
        INDArray networkOutput = model.output(features);
        long timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        System.out.println("\n\nProbabilities at last time step:");
        System.out.println(probabilitiesAtLastWord); //TODO Check which probability fits with each class in TRAINING!!!

        Map<Integer, String> labels = getLabelsMap();
    	String prediction = labels.get(probabilitiesAtLastWord.argMax(0).toIntVector()[0]);
    	return prediction;
    }
    
    public static String getPredictionSequence(String text, MultiLayerNetwork model) {
        INDArray features = loadFeaturesFromString(text);
        INDArray networkOutput = model.output(features);
        
        long[] arrsiz = networkOutput.shape();
        double max = 0;
        int pos = 0;
        for (int i = 0; i < arrsiz[1]; i++) { //arrsiz[1])=numClasses
            if (max < (double) networkOutput.getColumn(i).sumNumber()) {  //sum of probabilities for each word of the sequence (for a given class)
                max = (double) networkOutput.getColumn(i).sumNumber();
                pos = i;	//It takes as a result the index of the highest probability
            }
        }
        Map<Integer, String> labels = getLabelsMap();
    	String prediction = labels.get(pos);
    	return prediction;
    }
    
    public static Map<String, String> getPrediction(List<String> docs, MultiLayerNetwork model) {
    	Map<String, String> predictions = new HashMap<>();
    	
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
        	//List<String> docFilter = filterDoc(doc);
        	docsFilter.add(docFilter); 
        	maxLength = Math.max(maxLength, docFilter.size());
        	
        } 
        if(maxLength > truncateLength) maxLength = truncateLength;	
		
    	INDArray features = Nd4j.create(docs.size(), vectorSize, maxLength);
        for(int i=0; i<docs.size(); i++){
        	//Set doc features
        	List<String> docTokens = docsFilter.get(i); //Vectorized tokens of the document i
        	for(int j=0; j<docTokens.size() && j<maxLength; j++){
        		String token = docTokens.get(j);
        		INDArray tokenVector = word2Vec.getWordVectorMatrix(token);
        		INDArrayIndex[] indexFeat = new INDArrayIndex[]{point(i), all(), point(j)}; //Posición de la palabra en el documento (todo el vector que representa la palabra)
        		features.put(indexFeat, tokenVector); //features.put(index, value)
        	} 
        }
    	
        Map<Integer, String> labels = getLabelsMap();
        
    	INDArray networkOutput = model.output(features);
    	long[] arrsiz = networkOutput.shape();
    	for(int i = 0; i < arrsiz[0]; i++) { //arrsiz[0]-->numDocs
    		INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(arrsiz[2] - 1));
    		System.out.println(probabilitiesAtLastWord);
    		String pred = labels.get(probabilitiesAtLastWord.argMax(0).toIntVector()[0]);
    		predictions.put(docs.get(i), pred);
    	}
		return predictions;
    }
    
    public static Map<Integer, String> getLabelsMap(){
        Map<Integer,String> labels = new HashMap<Integer, String>(); 
    	try (BufferedReader brCategories = new BufferedReader(new FileReader(new File(LABELS_DESCRIPTION_PATH)))) {
    		String line = "";
    		while ((line = brCategories.readLine()) != null) {
    			String[] content = line.split(",");
    			labels.put(Integer.parseInt(content[0]), content[1]);
    		}
    		brCategories.close();
    	}catch(IOException e) {
    		log.severe("Exception while reading labels file :" + e.getMessage());
    	}
    	return labels;
    }

}
