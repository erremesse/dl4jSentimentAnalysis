package es.unizar.ml4bigdata.dl4j.tests;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * 3rd example. 
 * - Loads dataset using SentimentExampleIterator.
 * It uses DataSetIterator to train the model.
 *	-> SentimentExampleIterator3: Replicates file structure of dl4j-examples.
 * It uses INDArrays from the DataSet to evaluate the model with masks.
 * - LSTM + Rnn network
 * @author erremesse
 *
 */
public class SentimentAnalysis {
 
	public static final String WORD_VECTORS_PATH = "src/main/resources/word2Vec/full_eng_model.txt";
	public static final String DATA_PATH = "/datassd/rmontanes/sentimentAnalysisData/data248500/";
	
	public static void main(String[] args) throws Exception {
		int vectorSize = 100;   
		int nEpochs = 1; 
		//1st. 
		//Generate train and test sets from raw text to word vectors to DataSetIterators
		WordVectors vecModel = WordVectorSerializer.loadFullModel(WORD_VECTORS_PATH);
		//Parse files to get texts and labels
		//DataSetIterators for training and testing respectively
        //Using AsyncDataSetIterator to do data loading in a separate thread; this may improve performance vs. waiting for data to load
        DataSetIterator train = new AsyncDataSetIterator(new SentimentIterator(DATA_PATH,vecModel,100,140,true),1);
        DataSetIterator test = new AsyncDataSetIterator(new SentimentIterator(DATA_PATH,vecModel,100,140,false),1);
		
    	//2nd. 
		//Configure neural network
        MultiLayerNetwork net= NetworkBuilder.net4(vectorSize, 2);
		
		//3rd.
		//Train neural network
		System.out.println("Starting training");
        for( int i=0; i<nEpochs; i++ ){
            net.fit(train);
            train.reset();
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

            //4th.
    		//Evaluation
            Evaluation evaluation = new Evaluation();
            while(test.hasNext()){
                DataSet t = test.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = net.output(features,false,inMask,outMask);

                evaluation.evalTimeSeries(lables,predicted,outMask);
            }
            test.reset();

            System.out.println(evaluation.stats());
        }
		

	}
}
