package es.unizar.sentiment.analysis;

import java.io.IOException;
import java.util.Collection;
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

import es.unizar.sentiment.analysis.configuration.ConfigurationLoader;
import es.unizar.sentiment.analysis.data.SentimentIterator;

public class SentimentAnalysisTrain {
	
	private static Logger log = Logger.getLogger(SentimentAnalysisTrain.class.getName());
	
	/*static {
		Nd4j.getMemoryManager().setAutoGcWindow(10000); //https://deeplearning4j.org/workspaces
	}*/

	public static void main(String[] args) {
		ConfigurationLoader config = new ConfigurationLoader(null);
		//Load Word2Vec model
		//WordVectors wordVectors= WordVectorSerializer.loadStaticModel(new File(WORD2VEC_MODEL_PATH)); //StaticWord2Vec
		String w2v_model_path = config.word2vecModelPath;
		Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(w2v_model_path);
        //Just to check...
		Collection<String> lst = word2Vec.wordsNearestSum("hola", 10);
        log.info("10 Words closest to 'hola': {}" + lst);
        
        //Create dataset iterators (train & test) (TODO Improve tokenizer/preprocessor)
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        
        String trainset_path = config.trainSetPath;
        String testset_path = config.testSetPath;
        String labels_path = config.labelsPath;
        
        int batchSize = config.batchSize;
        int truncateLength = config.truncateLength;
        
        DataSetIterator train = null;
        DataSetIterator test = null;
        try {
            train = new SentimentIterator(trainset_path, word2Vec, batchSize, truncateLength, tokenizerFactory, labels_path);   
            test = new SentimentIterator(testset_path, word2Vec, batchSize, truncateLength, tokenizerFactory, labels_path); 
        }catch(Exception ex) {
        	log.severe("Exception while constructing SentimentIterator.");
        	ex.printStackTrace();
        }

        
        int inputNeurons = word2Vec.getLayerSize(); //wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        int outputs = train.getLabels().size(); //Number of classes

        //Configure neural network
        String modelPath = config.sentimentModelPath;
        int nEpochs = config.nEpochs;
        double learningRate = config.learningRate;
        int hiddenLayerSize = config.hiddenLayerSize;
        double clip = config.clip;
        
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
        	ModelSerializer.writeModel(net, modelPath, true);
    		log.info("Model saved. Execution completed successfully.");
        }catch(IOException ioex) {
        	log.info("Model couldn't be saved due to an error: \n" + ioex);
        }
        
	}

}
