package es.unizar.ml4bigdata.dl4j.tests;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.util.Pair;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

//import javafx.util.Pair;

/**
 * 1st example. 
 * - Loads dataset using CsvIterator and ParseCsvPreprocessor.
 * It uses INDArray matrices to train and evaluate the model.
 * - LSTM + Rnn network
 * @author erremesse
 *
 */
public class SentimentAnalysisExample1 {
 
	public static final String WORD_VECTORS_PATH = "src/main/resources/word2Vec/eng_model.txt";
	public static final String DATA_PATH = "src/main/resources/tweets20_eng.csv";
	
	public static void main(String[] args) throws Exception {
		int vectorSize = 50;   
        
		//1st. 
		//Generate train and test sets from raw text to word vectors to DataSetIterators
		WordVectors vecModel = WordVectorSerializer.loadFullModel(WORD_VECTORS_PATH);
		//Parse CSV file to get texts and labels
		ArrayList<INDArray> labelVectorList = new ArrayList<>();
		ArrayList<INDArray> avgTweetVectorList = new ArrayList<>();
		
		CsvIterator iterator = new CsvIterator(DATA_PATH, new ParseCsvPreprocessor("\\t"));
		while(iterator.hasNext()){
			 Pair<String[], double[]> pair = iterator.nextSentenceParsedCsv();
			 
			 // Calculate average vector value for the given tweet
             String[] words = pair.getKey();
             INDArray sumTweetVector = Nd4j.zeros(1, vectorSize);
             for (String word : words) {
            	 if(vecModel.getWordVectorMatrix(word) != null){
            		 sumTweetVector.addi(vecModel.getWordVectorMatrix(word));
            	 }
             }
             INDArray averageTweetVector = sumTweetVector.div(words.length);
             avgTweetVectorList.add(averageTweetVector);
		
			 // Add label to label vector list
			 INDArray labelVector = Nd4j.create(pair.getValue());
             labelVectorList.add(labelVector);
			}
		
		//2nd. 
		//Configure neural network
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //.iterations(1)
            .updater(Updater.RMSPROP)
//            .regularization(true).l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
//            .learningRate(0.0018)
            .list()
            .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(200)
                    .activation(Activation.SOFTSIGN).build())
            .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                    .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(2).build())
            .pretrain(false).backprop(true).build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));
		
		//3rd.
		//Train neural network
		List<INDArray> featureList = avgTweetVectorList;
		List<INDArray> targetList = labelVectorList;
		for (int i=0; i < featureList.size(); i++) {
            INDArray featureRow = featureList.get(i);
            INDArray labelRow = targetList.get(i);
            net.fit(featureRow, labelRow);
		}
		
		//4th.
		//Evaluation
		Evaluation eval = new Evaluation();
        for (int i=0; i < featureList.size(); i++) {
            INDArray featureRow = featureList.get(i);
            INDArray labelRow = targetList.get(i);
            INDArray output = net.output(featureRow);
            eval.eval(labelRow, output);
        }
        System.out.println(eval.stats());

	}
}
