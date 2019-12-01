package es.unizar.sentiment.analysis.model;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
//import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NetworkBuilder {
	
	public static MultiLayerNetwork simpleLSTM(int inputNeurons, int outputNeurons, int hiddenSize, double learningRate, double clip) {
		MultiLayerConfiguration model = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) 
		        .updater(new RmsProp(learningRate))
		        .l2(1e-1)
		        .weightInit(WeightInit.XAVIER)
		        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
		        .gradientNormalizationThreshold(clip)
		        .list()
		        .layer(0, new LSTM.Builder()
		        		.nIn(inputNeurons)
		        		.nOut(hiddenSize)
		                .activation(Activation.SOFTSIGN).build())
		        .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
		        		.nIn(hiddenSize)
		        		.nOut(outputNeurons)
		                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
		        .pretrain(false).backprop(true)
		        .build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(model);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        //net.setListeners(new HistogramIterationListener(1));  
        return net;
		
	}
	
	public static MultiLayerNetwork biLSTM(int inputNeurons, int outputNeurons, int hiddenSize, double learningRate, double clip) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new RmsProp(learningRate))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)//OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(clip) //clipping
                .list()
                .layer(0, new Bidirectional(new LSTM.Builder()
    											.nIn(inputNeurons)
    											.nOut(hiddenSize)
    											//.dropOut(0.5)
    											.activation(Activation.SOFTSIGN).build()))
                .layer(1, new RnnOutputLayer.Builder()	
                				//.nIn(hiddenLayerSize)
                				.nIn(2*hiddenSize)
                				.nOut(outputNeurons)
                				.activation(Activation.SOFTMAX)
                				.lossFunction(LossFunctions.LossFunction.MCXENT).build()) //Multi-Class Cross Entropy 
                .build();
            		
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            net.setListeners(new ScoreIterationListener(1));
            //net.setListeners(new HistogramIterationListener(1));  
            return net;
	}
	
	/**
	public static MultiLayerNetwork net1(int nIn, int nOut){
		//Basic neural network (Given in example Word2VecSentimentRNN)
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
	        .updater(Updater.RMSPROP)
	        .regularization(true).l2(1e-5)
	        .weightInit(WeightInit.XAVIER)
	        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
	        .learningRate(0.0018)
	        .list()
	        .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(200)
	                .activation("softsign").build())
	        .layer(1, new RnnOutputLayer.Builder().activation("softmax")
	                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(nOut).build())
	        .pretrain(false).backprop(true)
	        .build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));
		return net;
	}
	**/
	
}
