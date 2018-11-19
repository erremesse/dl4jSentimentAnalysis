package es.unizar.sentiment.analysis.tests;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Map.Entry;

import org.nd4j.linalg.primitives.Pair;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.opencsv.CSVReader;

public class CSVBatchReader {

	private final static String CSV_FILE = "./src/main/resources/data/intertass-ES-development-tagged.csv"; //"./src/test/resources/tweetsEN.csv";
	private final static String LABELS_FILE = "./src/main/resources/data/labels";

	
	private CSVReader reader;
	private int batchSize = 8;
    private int cursor = 0;
    private int totalDocs = 0;
    
    private BiMap<Integer,String> labels = HashBiMap.create(); //https://www.baeldung.com/guava-bimap
    
    public CSVBatchReader() {
    	try {
			this.reader = new CSVReader(new FileReader(CSV_FILE));	
			CSVReader readerAux = new CSVReader(new FileReader(CSV_FILE));	
	    	this.totalDocs = readerAux.readAll().size();
	    	readerAux.close();
	    	
	    	try (BufferedReader brCategories = new BufferedReader(new FileReader(new File(LABELS_FILE)))) {
	    		String line = "";
	    		while ((line = brCategories.readLine()) != null) {
	    			String[] content = line.split(",");
	    			labels.put(Integer.parseInt(content[0]), content[1]);
	    		}
	    		brCategories.close();
	    	}catch(Exception e) {
	    		System.out.println("Exception in reading file :" + e.getMessage());
	    	}
		}catch(IOException ioex) {
			ioex.printStackTrace();
		}
    }
    
	public static void main(String[] args) {	
		CSVBatchReader batching = new CSVBatchReader();
		
		Map<Integer, List<String>> textBatches = new HashMap<>();
		Map<Integer, int[]> classBatches = new HashMap<>();
		int i = 0;
		while(batching.hasNext()) {
			Pair<List<String>, int[]> pair = batching.next();
			textBatches.put(i, pair.getFirst());
			classBatches.put(i, pair.getSecond());
			i++;
		}
	
		
		for(Entry<Integer, List<String>> batch:textBatches.entrySet()){
			System.out.println("Batch" + batch.getKey() + " => " + batch.getValue().size() + " lines:");
			System.out.println(batch.getValue());
		}
		
		for(Entry<Integer, int[]> batch:classBatches.entrySet()){
			System.out.println("Batch" + batch.getKey() + " => " + batch.getValue().length + " labels:");
			System.out.println(batch.getValue());
		}
		
		
	}
	
	/*private static List<String> getNextBatch(int batch){
		List<String> chunk = new ArrayList<>();
		for(int i=0; i<batchSize; i++) {
			try (Stream<String> lines = Files.lines(Paths.get(CSV_FILE))) {
				int skipLines = batchSize*batch + i;
			    String line = lines.skip(skipLines).findFirst().get();
			    chunk.add(line);
			}catch(Exception ex) {
				ex.getStackTrace();
			}
		}
		return chunk;
		
	}*/
	
	private boolean hasNext() {
		return this.cursor < this.totalDocs;
	}
	
	private Pair<List<String>, int[]> next() {
		return next(batchSize);
	}

	private Pair<List<String>, int[]> next(int num) {
		if (cursor >= this.totalDocs) throw new NoSuchElementException();
        try {
            return nextBatch(num);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
	}

	private Pair<List<String>, int[]> nextBatch(int num) throws IOException {
		List<String> docs = new ArrayList<>(num);
        int[] categories = new int[num];
        
        //Load batch
        String[] line;
        for(int i=0; i<num && this.cursor<this.totalDocs; i++) {
        	//reader.skip(this.cursor+1);
        	line = this.reader.readNext();
        	if(line != null ) {
        		if(labels.inverse().get(line[1]) != null) {
        			System.out.println("Record " + i + " -> Text = " + line[0] + " && Label = " + line[1]);
        			docs.add(line[0]);
                	categories[i] = labels.inverse().get(line[1]); //Integer.parseInt(line[1]);
                } else {
                	System.out.println("[WARN]Didn't get a correct label in cursor point = " + cursor + "\n...Skipping line...");
                	i--;
                }
        		cursor++;
        	}
        }
		return new Pair<List<String>, int[]>(docs, categories);
	}

}
