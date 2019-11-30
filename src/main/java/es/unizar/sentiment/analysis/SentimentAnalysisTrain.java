package es.unizar.sentiment.analysis;

import java.io.IOException;
import java.util.Collection;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import es.unizar.sentiment.analysis.data.SentimentIterator;

public class SentimentAnalysisTrain {
	private static final String WORD2VEC_MODEL_PATH = "./src/main/resources/word2vec/ES/SBW-vectors-300-min5.bin.gz";
													//"./src/main/resources/word2vec/ES/TCH_W2V_100.zip"; //TODO
													//"./src/main/resources/word2vec/ES/SBW-vectors-300-min5.bin.gz"; 
													//"./src/main/resources/word2vec/EN/raw_eng_model.zip";
	//private static final String WORD_VECTORS_PATH = ";
	//private static final String DATASET_PATH = "./src/main/resources/data/";
	private static final String TRAINSET_PATH = "./src/main/resources/data/ES/train/Turismo_General_Comunicacion_Hackathon_5l-TAG_sinduplicados.csv"; //"./src/main/resources/data/ES/train/InterTASS_ES_TRAIN.csv"; 
												//"./src/main/resources/data/ES/train/Turismo_General_Comunicacion_Hackathon_5l-TAG_sinduplicados.csv"; 
												//"./src/main/resources/data/ES/train/Turismo_Comunicacion_Hackathon_3l-TAG.csv";
												//"./src/main/resources/data/ES/train/Turismo_Comunicacion_Hackathon_5l-TAG.csv";
												//"./src/main/resources/data/EN/tweetsEN.csv";
	private static final String TESTSET_PATH = "./src/main/resources/data/ES/test/SocialMoriarty_SentimentAnalysis_test1051.csv"; //"./src/main/resources/data/ES/test/InterTASS_ES_DEV.csv"; //"./src/main/resources/data/ES/test/SocialMoriarty_SentimentAnalysis_test1051.csv"; 
	private static final String LABELS_DESCRIPTION_PATH = "./src/main/resources/data/ES/train/labels5"; // "./src/main/resources/data/ES/train/labels4"; //"./src/main/resources/data/ES/train/labels5";
														//"./src/main/resources/data/ES/train/labels3"; 
														//"./src/main/resources/data/ES/train/labels5";
														//"./src/main/resources/data/EN/labelsEN";
	private static final String SENTIMENT_MODEL_PATH = "./src/main/resources/models/ES/ITA/model"; //"./src/main/resources/models/ES/TASS/model"; //"./src/main/resources/models/ES/ITA/model";
														//"./src/main/resources/models/ES/model"; 
														//"./src/main/resources/models/EN/model";
	
    private static final int batchSize = 16;	//Number of examples in each minibatch
    private static final int nEpochs = 20;	//Number of epochs (full passes of training data) to train on
    private static final int hiddenLayerSize = 50;
    private static final double learningRate = 0.00001;
    private static final double clip = 1.0;
    
    private static final int truncateLength = 50; //Number of words per Tweet
	
	private static Logger log = Logger.getLogger(SentimentAnalysisTrain.class.getName());
	
	/*static {
		Nd4j.getMemoryManager().setAutoGcWindow(10000); //https://deeplearning4j.org/workspaces
	}*/

	public static void main(String[] args) {
		//Load Word2Vec model
		//WordVectors wordVectors= WordVectorSerializer.loadStaticModel(new File(WORD2VEC_MODEL_PATH)); //StaticWord2Vec
		Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(WORD2VEC_MODEL_PATH);
        //Collection<String> lst = word2Vec.wordsNearestSum("hola", 10);
        //log.info("10 Words closest to 'hola': {}" + lst);
        
        //Create dataset iterators (train & test)
        //TODO Improve tokenizer/preprocessor
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
            .updater(new RmsProp(learningRate))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)//OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
            .l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(clip) //clipping
            .list()
            //.layer(0, new LSTM.Builder().nIn(inputNeurons).nOut(hiddenLayerSize)
                //.activation(Activation.SOFTSIGN).build())
            .layer(0, new Bidirectional(new LSTM.Builder()
											.nIn(inputNeurons)
											.nOut(hiddenLayerSize)
											//.dropOut(0.5)
											.activation(Activation.SOFTSIGN).build()))
            .layer(1, new RnnOutputLayer.Builder()	
            				//.nIn(hiddenLayerSize)
            				.nIn(2*hiddenLayerSize)
            				.nOut(outputs)
            				.activation(Activation.SOFTMAX)
            				.lossFunction(LossFunctions.LossFunction.MCXENT).build()) //Multi-Class Cross Entropy 
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
            
            //net.rnnClearPreviousState(); //Do I need to clear previous state manually??
        }
        log.info("Training finished!");
        
        /* TODO Check early stopping!
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
        		.epochTerminationConditions(new MaxEpochsTerminationCondition(10))
        		.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(5, TimeUnit.MINUTES))
        		.scoreCalculator(new DataSetLossCalculator(test, true))
                .evaluateEveryNEpochs(1)
        		.modelSaver(new LocalFileModelSaver(SENTIMENT_MODEL_PATH))
        		.build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, conf, train);
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        */
       
        try {
        	ModelSerializer.writeModel(net, SENTIMENT_MODEL_PATH, true);
    		log.info("Model saved. Execution completed successfully.");
        }catch(IOException ioex) {
        	log.info("Model couldn't be saved due to an error: \n" + ioex);
        }
        
	}

}
