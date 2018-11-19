package es.unizar.sentiment.analysis.tests;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

import com.opencsv.CSVParser;
import com.opencsv.CSVReader;

public class FileSplit {

	private static final String DATASET_PATH = "./src/main/resources/data/intertass-ES-development-tagged.csv";//"./src/main/resources/data/tweetsEN.csv";
	private static final String DATASET_SPLIT_TRAIN_PATH = "./src/main/resources/data/EN/train/";
	private static final String DATASET_SPLIT_TEST_PATH = "./src/main/resources/data/EN/test/";
	
	public static void main(String[] args) throws IOException {

		/*FileInputStream inputStream = null;
		Scanner sc = null;
		try {
		    inputStream = new FileInputStream(DATASET_PATH);
		    sc = new Scanner(inputStream, "UTF-8");
		    while (sc.hasNextLine()) {
		        String line = sc.nextLine();
		        String[] parts = line.split(",");
		        System.out.println(line);
		        System.out.println("==> [0]:"+parts[0]+" \n[1]:"+parts[1]);
		    }
		    // note that Scanner suppresses exceptions
		    if (sc.ioException() != null) {
		        throw sc.ioException();
		    }
		} finally {
		    if (inputStream != null) {
		        inputStream.close();
		    }
		    if (sc != null) {
		        sc.close();
		    }
		}*/
		
		CSVReader readerAll = null;
		CSVReader reader = null;
        try {   
        	readerAll = new CSVReader(new FileReader(DATASET_PATH));
            reader = new CSVReader(new FileReader(DATASET_PATH));
            int totalDocs = readerAll.readAll().size();
            readerAll = null;
            String[] line;
            //int i = 0;
            //while ((line = reader.readNext()) != null) {
            for(int i=0; i<totalDocs; i++) {
            	line = reader.readNext();
                System.out.println("Record " + i + " -> Text = " + line[0] + " && Label = " + line[1]);
                //i++; 
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
		
		
		

	}

}
