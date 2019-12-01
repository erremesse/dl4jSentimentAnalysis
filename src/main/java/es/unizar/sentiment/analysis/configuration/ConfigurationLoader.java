package es.unizar.sentiment.analysis.configuration;

public class ConfigurationLoader {

	private String path;
	public String word2vecModelPath;
	public String trainSetPath;
	public String testSetPath;
	public String labelsPath;
	public String sentimentModelPath;
	
	public int batchSize;
	public int truncateLength;
	public int nEpochs;
	public double learningRate;
	public int hiddenLayerSize;
	public double clip;
	
	/** Default values **/
	private static final String WORD2VEC_MODEL_PATH = "./src/main/resources/word2vec/ES/SBW-vectors-300-min5.bin.gz";
														//"./src/main/resources/word2vec/ES/TCH_W2V_100.zip"; //TODO
														//"./src/main/resources/word2vec/ES/SBW-vectors-300-min5.bin.gz"; 
														//"./src/main/resources/word2vec/EN/raw_eng_model.zip";
	//private static final String WORD_VECTORS_PATH = ";
	//private static final String DATASET_PATH = "./src/main/resources/data/";
	private static final String TRAINSET_PATH = "./src/main/resources/data/ES/train/Turismo_Comunicacion_Hackathon_5l-TAG.csv"; 
												 //"./src/main/resources/data/ES/train/Turismo_Comunicacion_Hackathon_3l-TAG.csv";
	                                             //"./src/main/resources/data/ES/train/Turismo_Comunicacion_Hackathon_5l-TAG.csv";
	                                             //"./src/main/resources/data/EN/tweetsEN.csv";
	private static final String TESTSET_PATH = "./src/main/resources/data/ES/test/SocialMoriarty_SentimentAnalysis_test1051.csv"; 
	private static final String LABELS_DESCRIPTION_PATH = "./src/main/resources/data/ES/train/labels5";
															//"./src/main/resources/data/ES/train/labels3"; 
															//"./src/main/resources/data/ES/train/labels5";
															//"./src/main/resources/data/EN/labelsEN";
	private static final String SENTIMENT_MODEL_PATH = "./src/main/resources/models/ES/ITA/model";
														//"./src/main/resources/models/ES/model"; 
														//"./src/main/resources/models/EN/model";

	private static final int BATCH_SIZE = 64;	//Number of examples in each minibatch
	private static final int NUM_EPOCHS = 1;	//Number of epochs (full passes of training data) to train on
	private static final int HIDDEN_LAYER_SIZE = 2;
	private static final double LEARNING_RATE = 0.0001;

	private static final int TRUNCATE_LENGTH = 50; //Number of words per Tweet

	public ConfigurationLoader(String configPath) {
		this.path =  configPath;
		if (this.path == null) {
			loadDefault();
		}
	}
	
	private void loadDefault() {
		this.word2vecModelPath = WORD2VEC_MODEL_PATH;
		this.trainSetPath = TRAINSET_PATH;
		this.testSetPath = TESTSET_PATH;
		this.labelsPath = LABELS_DESCRIPTION_PATH;
		this.sentimentModelPath = SENTIMENT_MODEL_PATH;
		this.batchSize = BATCH_SIZE;
		this.truncateLength = TRUNCATE_LENGTH;
		this.nEpochs = NUM_EPOCHS;
		this.learningRate = LEARNING_RATE;
		this.hiddenLayerSize = HIDDEN_LAYER_SIZE;
		this.clip = 1.0;
	}

}
