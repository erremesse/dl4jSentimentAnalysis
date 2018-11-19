package es.unizar.ml4bigdata.dl4j.tests;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
//import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NetworkBuilder {
	
	
public static MultiLayerNetwork net1(int nIn, int nOut){
		//Basic neural network (Given in example Word2VecSentimentRNN)
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //.iterations(1)
	        .updater(Updater.RMSPROP)
	        //.regularization(true).l2(1e-5)
	        .weightInit(WeightInit.XAVIER)
	        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
	        //.learningRate(0.0018)
	        .list()
	        .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(200)
	                .activation(Activation.SOFTSIGN).build())
	        .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
	                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(nOut).build())
	        .pretrain(false).backprop(true)
	        .build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));
		//net.setListeners(new HistogramIterationListener(1));
		return net;
	}
	
	public static MultiLayerNetwork net2(int nIn, int nOut){
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //.iterations(1)
		        .updater(Updater.RMSPROP)
		        //.regularization(true).l2(1e-5)
		        .weightInit(WeightInit.XAVIER)
		        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
		        //.learningRate(0.0018)
		        .list()
		        .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(50)
		                .activation(Activation.SOFTSIGN).build())
		        .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
		                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(50).nOut(nOut).build())
		        .pretrain(false).backprop(true)
		        .build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));
		return net;
	}
	
	/**
	 * A regular dense layer cannot be used with RNN/LSTMs (they are thought as "2dimensional" layers)
	 */
	public static MultiLayerNetwork net3(int nIn, int nOut){
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //.iterations(1)
		        .updater(Updater.RMSPROP)
		        //.regularization(true).l2(1e-5)
		        .weightInit(WeightInit.XAVIER)
		        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
		        //.learningRate(0.0018)
		        .list()
		        .layer(0, new DenseLayer.Builder().activation(Activation.SIGMOID).nIn(nIn).nOut(100).build()) 
		        .layer(1, new GravesLSTM.Builder().nIn(100).nOut(200)
		                .activation(Activation.SOFTSIGN).build())
		        .layer(2, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
		                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(nOut).build())
		        .pretrain(false).backprop(true)
		        .build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));
		//net.setListeners(new HistogramIterationListener(1));
		return net;
	}
	
	public static MultiLayerNetwork net4(int nIn, int nOut){
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //.iterations(1)
		        .updater(Updater.RMSPROP)
		        //.regularization(true).l2(1e-5)
		        .weightInit(WeightInit.XAVIER)
		        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
		        //.learningRate(0.0018)
		        .list()
		        .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(200)
		                .activation(Activation.SOFTSIGN).build())
		        .layer(1, new GravesLSTM.Builder().nIn(200).nOut(200)
		                .activation(Activation.SOFTSIGN).build())
		        .layer(2, new GravesLSTM.Builder().nIn(200).nOut(200)
		                .activation(Activation.SOFTSIGN).build())
		        .layer(3, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
		                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(nOut).build())
		        .pretrain(false).backprop(true)
		        .build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));
		//net.setListeners(new HistogramIterationListener(1));
		return net;
	}
}
