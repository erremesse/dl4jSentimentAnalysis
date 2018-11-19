package es.unizar.sentiment.analysis.word2vec;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
//import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.io.ClassPathResource;


public class Word2VecTrain {

	//private static final String FILE_NAME = "raw_sentences.txt";
	private static final String FILE_PATH = "./src/main/resources/word2vec/EN/raw_sentences.txt";
	private static final String WORD2VEC_MODEL_PATH = "./src/main/resources/word2vec/EN/raw_eng_model.zip";
	private static final String WORD_VECTORS_PATH = "./src/main/resources/word2vec/EN/raw_eng_vectors.txt";
	
	private static final List<String> stopList_EN = Arrays.asList("a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
			"be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during",
			"each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's",
			"i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not",
			"of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such",
			"than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up",
			"very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't",
			"you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"); 
	private static final List<String> stopList_ES = Arrays.asList("");//TODO
	
	private static Logger log = Logger.getLogger(Word2VecTrain.class.getName());
	
	public static void main(String[] args) {
		log.info("Loading files and configuring processors...");
		
		//String fileName = new ClassPathResource(FILE_NAME).getFile().getAbsolutePath();
		// Strip white space before and after for each line
        //SentenceIterator iter = new BasicLineIterator(fileName); //It uses Classpath
		
		SentenceIterator iter = new LineSentenceIterator(new File(FILE_PATH)); //It needs a path
		iter.setPreProcessor(new SentencePreProcessor() {
            public String preProcess(String sentence) {
            	//TODO Improve text preprocessing for each line ?
                return sentence.toLowerCase();
            }
        });
		
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor()); //TODO Improve text preprocessing for each word ?
        
        
        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
        		//.batchSize(0)
                .epochs(10)
                .minWordFrequency(5)
                //.useAdaGrad(true)
                .layerSize(100)
                //.learningRate(0)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .stopWords(stopList_EN) 
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing model to a file....");
        try {
        	//WordVectorSerializer.writeFullModel(vec, WORD2VEC_MODEL_PATH); --> Consider using writeWord2VecModel() method
        	WordVectorSerializer.writeWord2VecModel(vec, WORD2VEC_MODEL_PATH);
			WordVectorSerializer.writeWordVectors(vec.lookupTable(), WORD_VECTORS_PATH); //Writes word vectors to text file
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        log.info("word2vec training finished!");
	}
}
