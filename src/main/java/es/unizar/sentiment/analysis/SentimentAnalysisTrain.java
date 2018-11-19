package es.unizar.sentiment.analysis;

import java.io.IOException;
import java.util.Collection;
import java.util.logging.Logger;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class SentimentAnalysisTrain {
	private static final String WORD2VEC_MODEL_PATH = "./src/main/resources/word2vec/ES/SBW-vectors-300-min5.bin.gz"; //"./src/main/resources/word2vec/EN/raw_eng_model.zip";
	//private static final String WORD_VECTORS_PATH = "./src/main/resources/word2vec/EN/raw_eng_vectors.txt";
	//private static final String DATASET_PATH = "./src/main/resources/data/EN";
	private static final String TRAINSET_PATH = "./src/main/resources/data/ES/intertass-ES-development-tagged.csv"; //"./src/main/resources/data/EN/tweetsEN_32.csv";
	private static final String TESTSET_PATH = "./src/main/resources/data/ES/intertass-ES-development-tagged.csv"; //"./src/main/resources/data/EN/tweetsEN_32.csv";
	private static final String LABELS_DESCRIPTION_PATH = "./src/main/resources/data/ES/labels"; //"./src/main/resources/data/EN/labelsEN";
	private static final String SENTIMENT_MODEL_PATH = "./src/main/resources/models/ES/model"; //"./src/main/resources/models/EN/model";
	
    private static final int batchSize = 8;	//Number of examples in each minibatch
    private static final int nEpochs = 2;	//Number of epochs (full passes of training data) to train on
    
    private static final int truncateLength = 50; //Number of words per Tweet
	
	private static Logger log = Logger.getLogger(SentimentAnalysisTrain.class.getName());
	
	static {
		Nd4j.getMemoryManager().setAutoGcWindow(10000); //https://deeplearning4j.org/workspaces
	}

	public static void main(String[] args) {
		//Load Word2Vec model
		//WordVectors wordVectors= WordVectorSerializer.loadStaticModel(new File(WORD2VEC_MODEL_PATH)); //StaticWord2Vec
		Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(WORD2VEC_MODEL_PATH);
        Collection<String> lst = word2Vec.wordsNearestSum("day", 10);
        log.info("10 Words closest to 'day': {}" + lst);
        
        //Create dataset iterators (train & test)
        //TODO Imrprove tokenizer/preprocessor
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        
        DataSetIterator train = null;
        DataSetIterator test = null;
        try {
            train = new SentimentIterator(TRAINSET_PATH, word2Vec, batchSize, truncateLength, tokenizerFactory, LABELS_DESCRIPTION_PATH);   
            test = new SentimentIterator(TESTSET_PATH, word2Vec, batchSize, truncateLength, tokenizerFactory, LABELS_DESCRIPTION_PATH); 
        }catch(Exception ex) {
        	log.severe("Exception while constructing SentimentIterator.");
        	ex.printStackTrace();
        }

        
        int inputNeurons = word2Vec.getLayerSize(); //wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        int outputs = train.getLabels().size(); //Number of classes

        //Configure neural network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) 
		        .updater(new RmsProp())
		        .l2(1e-1)
		        .weightInit(WeightInit.XAVIER)
		        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
		        .list()
		        .layer(0, new LSTM.Builder().nIn(inputNeurons).nOut(50)
		                .activation(Activation.SOFTSIGN).build())
		        .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
		                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(50).nOut(outputs).build())
		        .pretrain(false).backprop(true)
		        .build();
        		
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        //net.setListeners(new HistogramIterationListener(1));
        
        
        //Train neural network
        log.info("Starting training... ");
        for (int i = 0; i < nEpochs; i++) {
        	net.fit(train);
            train.reset();
            log.info("Epoch " + i + " complete. Starting evaluation:");
            
            Evaluation evaluation = net.evaluate(test);
            System.out.println(evaluation.stats());
        }
        log.info("Training finished!");
        
        try {
        	ModelSerializer.writeModel(net, SENTIMENT_MODEL_PATH, true);
    		log.info("Model saved. Execution completed successfully.");
        }catch(IOException ioex) {
        	log.info("Model couldn't be saved due to an error: \n" + ioex);
        }
        
	}

}
