package es.unizar.sentiment.analysis;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;
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
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode;

import es.unizar.sentiment.analysis.data.SentimentIterator;

/**
 * Gradient Sharing minimal example (Not working yet --> NullPointerException during fit operation (SharedTrainingWrapper.run(SharedTrainingWrapper.java:475))
 * Take a look at: https://gitter.im/deeplearning4j/deeplearning4j?at=5d93477d0e829f60475be9b5)
 * @author erremesse
 *
 */
public class MinimalExampleGS {
	private static final String WORD2VEC_MODEL_PATH = "./src/main/resources/word2vec/ES/SBW-vectors-300-min5.bin.gz"; 
	private static final String TRAINSET_PATH = "./src/main/resources/data/ES/train/Turismo_General_Comunicacion_Hackathon_5l-TAG_sinduplicados.csv";   
	private static final String LABELS_DESCRIPTION_PATH = "./src/main/resources/data/ES/train/labels5"; 
	private static final String TESTSET_PATH = "./src/main/resources/data/ES/test/SocialMoriarty_SentimentAnalysis_test1051.csv"; 
	private static final String SENTIMENT_MODEL_PATH = "./src/main/resources/models/ES/ITA/model"; 
	
    private static final int batchSize = 8;	//Number of examples in each minibatch
    private static final int nEpochs = 10;	//Number of epochs (full passes of training data) to train on
    
    private static final int truncateLength = 50; //Number of words per Tweet

    //"Number of examples to fit each worker with
    private static int batchSizePerWorker = 16;
    
    //Number of epochs for training
    private static int numEpochs = 10;
    
    private static Logger log = Logger.getLogger(MinimalExampleGS.class.getName());
    
	public static void main(String[] args) {
	    SparkConf sconf = new SparkConf().
	    		setAppName("Simple MinimalExamplePA").
	    		setMaster("local[*]");	   
	    
		JavaSparkContext sc = new JavaSparkContext(sconf);
		
		//Load Word2Vec model
		//WordVectors wordVectors= WordVectorSerializer.loadStaticModel(new File(WORD2VEC_MODEL_PATH)); //StaticWord2Vec
		Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(WORD2VEC_MODEL_PATH);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        
		DataSetIterator iterTrain = null;
		DataSetIterator iterTest = null;
        try {
            iterTrain = new SentimentIterator(TRAINSET_PATH, word2Vec, batchSize, truncateLength, tokenizerFactory, LABELS_DESCRIPTION_PATH);   
            iterTest = new SentimentIterator(TESTSET_PATH, word2Vec, batchSize, truncateLength, tokenizerFactory, LABELS_DESCRIPTION_PATH);   
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
            trainDataList.add(iterTest.next());
        }
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);
        //

		//Model setup as on a single node. Either a MultiLayerConfiguration or a ComputationGraphConfiguration
        int inputNeurons = word2Vec.getLayerSize(); //wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        int outputs = iterTrain.getLabels().size(); //Number of classes
		MultiLayerConfiguration model = new NeuralNetConfiguration.Builder()
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

		// Configure distributed training required for gradient sharing implementation
		VoidConfiguration conf = VoidConfiguration.builder()
						.unicastPort(40123)             //Port that workers will use to communicate. Use any free port
						.networkMask("192.168.0.0/16")     //Network mask for communication. Examples 10.0.0.0/24, or 192.168.0.0/16 etc
						.controllerAddress("10.0.2.4")  //IP of the master/driver
						.build();

		//Create the TrainingMaster instance
		TrainingMaster trainingMaster = new SharedTrainingMaster.Builder(conf, 1)
						.batchSizePerWorker(batchSizePerWorker) //Batch size for training
						.updatesThreshold(1e-3)                 //Update threshold for quantization/compression. See technical explanation page
						.workersPerNode(1)      // equal to number of GPUs. For CPUs: use 1; use > 1 for large core count CPUs
		                //.meshBuildMode(MeshBuildMode.MESH)      // or MeshBuildMode.PLAIN for < 32 nodes
						.build();

		//Create the SparkDl4jMultiLayer instance
		SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, model, trainingMaster);

		//Execute training:
		for (int i = 0; i < numEpochs; i++) {
		    sparkNet.fit(trainData);
            log.info("Epoch " + i + " complete. Starting evaluation:");
            
		}
		Evaluation evaluation = sparkNet.evaluate(testData);
        System.out.println(evaluation.stats());
		System.out.println("Training finished!!!");
	}

}
