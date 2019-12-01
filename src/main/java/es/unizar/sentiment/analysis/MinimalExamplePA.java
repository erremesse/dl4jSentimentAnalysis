package es.unizar.sentiment.analysis;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import es.unizar.sentiment.analysis.configuration.ConfigurationLoader;
import es.unizar.sentiment.analysis.data.SentimentIterator;

/**
 * Parameter Averaging minimal example
 * @author erremesse
 *
 */
public class MinimalExamplePA {

    private static final int batchSize = 8;	//Number of examples in each minibatch
    private static final int nEpochs = 2;	//Number of epochs (full passes of training data) to train on
    
    private static final int truncateLength = 50; //Number of words per Tweet
    
/*
    //Use spark local (helper for testing/running without spark submit
    private static boolean useSparkLocal = true;
*/
    //"Number of examples to fit each worker with
    private static int batchSizePerWorker = 16;
    
    private static Logger log = Logger.getLogger(MinimalExamplePA.class.getName());
    
	public static void main(String[] args) {
		ConfigurationLoader config = new ConfigurationLoader(null);
		String w2v_model_path = config.word2vecModelPath;
		
	    SparkConf conf = new SparkConf().
	    		setAppName("Simple MinimalExamplePA").
	    		setMaster("local[*]");	   
	    
		JavaSparkContext sc = new JavaSparkContext(conf);
		
		//Load Word2Vec model
		//WordVectors wordVectors= WordVectorSerializer.loadStaticModel(new File(WORD2VEC_MODEL_PATH)); //StaticWord2Vec
		Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(w2v_model_path);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        
        String trainset_path = config.trainSetPath;
        String testset_path = config.testSetPath;
        String labels_path = config.labelsPath;
        
        int batchSize = config.batchSize;
        int truncateLength = config.truncateLength;
        
		DataSetIterator iterTrain = null;
		DataSetIterator iterTest = null;
        try {
            iterTrain = new SentimentIterator(trainset_path, word2Vec, batchSize, truncateLength, tokenizerFactory, labels_path);   
            iterTest = new SentimentIterator(testset_path, word2Vec, batchSize, truncateLength, tokenizerFactory, labels_path); 
        }catch(Exception ex) {
        	log.severe("Exception while constructing SentimentIterator.");
        	ex.printStackTrace();
        }
        
		//Load the data into memory then parallelize
        List<DataSet> trainDataList = new ArrayList<>();
        while (iterTrain.hasNext()) {
            trainDataList.add(iterTrain.next());
        }

        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
        
        //Idem for test
        List<DataSet> testDataList = new ArrayList<>();
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);

      //Configure neural network
        int numEpochs = config.nEpochs;
        double learningRate = config.learningRate;
        int hiddenLayerSize = config.hiddenLayerSize;
        double clip = config.clip;
        
		//Model setup as on a single node. Either a MultiLayerConfiguration or a ComputationGraphConfiguration
        int inputNeurons = word2Vec.getLayerSize(); //wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        int outputs = iterTrain.getLabels().size(); //Number of classes
        MultiLayerConfiguration model = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) 
		        .updater(new RmsProp())
		        .l2(1e-1)
		        .weightInit(WeightInit.XAVIER)
		        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(clip)
		        .list()
		        .layer(0, new LSTM.Builder().nIn(inputNeurons).nOut(hiddenLayerSize)
		                .activation(Activation.SOFTSIGN).build())
		        .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
		                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(hiddenLayerSize).nOut(outputs).build())
		        .pretrain(false).backprop(true)
		        .build();

		//Create the TrainingMaster instance
		int examplesPerDataSetObject = 1;
		TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
				.averagingFrequency(5)
				.workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
				.batchSizePerWorker(batchSizePerWorker)	// Workarround to solve: "Exception in thread ADSI prefetch" 
				.rddTrainingApproach(RDDTrainingApproach.Direct)
				.build();

		//Create the SparkDl4jMultiLayer instance and fit the network using the training data:
		SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, model, trainingMaster);

		//Execute training:
		for (int i = 0; i < numEpochs; i++) {
		    sparkNet.fit(trainData);
		}
		Evaluation evaluation = sparkNet.evaluate(testData);
        System.out.println(evaluation.stats());
		System.out.println("Training finished!!!");
	}

}
