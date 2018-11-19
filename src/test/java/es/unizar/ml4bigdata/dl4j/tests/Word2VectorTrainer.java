package es.unizar.ml4bigdata.dl4j.tests;

import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

public class Word2VectorTrainer {

	private static final String RAW_SENTENCES = "src/main/resources/raw_sentences/raw_sentences_full.txt";//"src/main/resources/raw_sentences.txt";
	private static final String MODEL_PATH = "src/main/resources/word2Vec/full_eng_model.txt";
	private static final String WORD_VECTORS_PATH = "src/main/resources/word2Vec/full_eng_vectors.txt";
	
	private static final int MIN_WORD_FREQ = 5;
	private static final int ITERATIONS = 2;
	private static final double LEARNINGRATE = 0.025;
	private static final int VECTOR_DIM = 100;
	private static final long SEED = 42L;
	private static final int WINDOW_SIZE = 5;
	
	private static final List<String> stopList = Arrays.asList("a","am","an","are","at","be","been","being","do","did","for","get","had","has","have",
			"i","is","in","of","on","that","the","these","this","to","with", "b", "c", "d", "e", "f", "g", "h", "l", "m", "n", "o", "p", "q", "r", "s", "t",
			"u", "v", "w");
	
	private static Logger log = Logger.getLogger(Word2VectorTrainer.class.getName());
	
	public static void main(String[] args) {
		try{
			// Strip white space before and after for each line
			SentenceIterator iterator = new BasicLineIterator(RAW_SENTENCES); //FileSentenceIterator(file)
			// Split on white spaces in the line to get words
			TokenizerFactory t = new DefaultTokenizerFactory();
			t.setTokenPreProcessor(new CommonPreprocessor());

			log.info("Building model....");
			Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(MIN_WORD_FREQ)
                .iterations(ITERATIONS)
                .learningRate(LEARNINGRATE)
                .layerSize(VECTOR_DIM)
                .seed(SEED)
                .windowSize(WINDOW_SIZE)
                .iterate(iterator)
                .tokenizerFactory(t)
                .stopWords(stopList) //Just for english
                .build();

			log.info("Fitting Word2Vec model....");
			vec.fit();

			log.info("Writing model to a file....");
			WordVectorSerializer.writeFullModel(vec, MODEL_PATH);
        
			/** In case we want to save serialized model.
			 * 
			 * Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
			 * SerializationUtils.saveObject(vec,new File("w2v_model.ser"));
			 */
        
			log.info("Writing word vectors to text file....");
			WordVectorSerializer.writeWordVectors(vec, WORD_VECTORS_PATH);
		}catch(Exception ex){
			ex.printStackTrace();
		}

	}

}
