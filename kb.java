/*
 * PROGRAM THAT RETURN FOR EACH AMBIGOUS INSTANCE THE MOST COMMON SENSE
 */
package javaapplication2;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import it.uniroma1.lcl.babelnet.BabelNet;
import it.uniroma1.lcl.babelnet.BabelSense;
import it.uniroma1.lcl.babelnet.BabelSynset;
import it.uniroma1.lcl.babelnet.BabelSynsetComparator;
import it.uniroma1.lcl.babelnet.BabelSynsetID;
import it.uniroma1.lcl.jlt.util.Language;

import java.io.FileReader;
import java.io.*;
import java.util.Scanner;

import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;


/**
 *
 * @author lollo
 */
public class Test {
    public static void main(String[] args)
	{
            try {
		//I CHOOSE TO USE AN AUXILIARY DICTIONARY IN WHICH ADD THE (WORD, MOST_COMMON_SENSE) ALREADY MET, AS TO SAVE 
		// AS MUCH BABELCOINS AS POSSIBLE
                BabelNet bn = BabelNet.getInstance();
                Map babel_dict = new HashMap();
                
                int flag=0;// zero value for deviation accuracy calculation, else test set prediction
                if(flag==0){
                    
                                    //Build a dicionary that map senseval2.d001.s079.t002 to its more frequest sense
					//BASED ON THE XML_DEV FILE 
                                    String FILENAME = "/home/lollo/Desktop/NLP/Homework2/DevData/ALL.data.xml";
                                    Scanner file = new Scanner(new FileReader(FILENAME));
                                    Map dictionary = new HashMap();
                                   int counter=0;
                                   
                                      while (file.hasNextLine()) {
                                                String line=file.nextLine();
                                                if(line.startsWith("<instance") && counter<1000){
                                                    counter++;
                                                    System.out.println(counter);
                                                            line=line.replaceAll("<instance id=\"","" );
                                                            line=line.replaceAll("\" lemma=\""," " );
                                                            line=line.replaceAll("\" pos=\""," " );
                                                            line=line.replaceAll("\">"," " );
                                                            line=line.replaceAll("\">"," " );
                                                            line=line.replaceAll("</instance>","" );
                                                            String[] s=line.split(" ",0);
                                                            if(babel_dict.containsKey(s[1])){
                                                            dictionary.put(s[0],babel_dict.get(s[1]));
                                                            }
                                                            else {
                                                                        try{
                                                                            BabelSynset syn = bn.getSynsets(s[1], Language.EN).stream()
                                                                                      .sorted(new BabelSynsetComparator(s[1], Language.EN))
                                                                                      .findFirst().get(); 
                                                                                BabelSynsetID value= syn.getID();
                                                                            dictionary.put(s[0],value );
                                                                            //System.out.println(s[0]+"\t"+value);
                                                                            babel_dict.put(s[1], value);
                                                                        }
                                                                        catch(Exception e){
                                                                        String value="not found";
                                                                           dictionary.put(s[0],value );
                                                                            //System.out.println(s[0]+"\t"+value);
                                                                            babel_dict.put(s[1], value);
                                                                        }catch (OutOfMemoryError e){
                                                                        String value="not found";
                                                                           dictionary.put(s[0],value );
                                                                            //System.out.println(s[0]+"\t"+value);
                                                                            babel_dict.put(s[1], value);
                                                                        }
                                                            }
                                                            //System.out.println(s[0]+" "+s[1]+" "+s[2]+" "+s[3]);
                                                                }
                                                }
                                      System.out.println(dictionary.size());

                                      //BUILD A SECOND DICTIONARY BASED ON THE CORRECT MEANING FOR THE WORDS
                                      String FILENAME_dev = "/home/lollo/Desktop/NLP/Homework2/DevData/ALL.gold.key.bnids.txt";
                                      Scanner file_dev = new Scanner(new FileReader(FILENAME_dev));
                                      Map dictionary_dev = new HashMap();
                                      while (file_dev.hasNextLine()) {
                                                String line=file_dev.nextLine();
                                                            String[] s=line.split(" ",0);
                                                            dictionary_dev.put(s[0], s[1]);
                                                            //System.out.println(s[0]+" "+s[1]);                         
                                                }
                                      System.out.println(dictionary_dev.size());

                                        //CALCULATE ACCURACY IN DEV SET
                                      Set<String> keys= dictionary.keySet();
                                      System.out.println("j");
                                      Object[] kk= keys.toArray();
                                      //System.out.println(kk[0].toString());
                                      double count_correct=0;
                                      for(int i=0;i<kk.length;i++){
                                          System.out.print(kk[i]+"    ");
                                          System.out.print(dictionary.get(kk[i]).toString()+"     ");
                                          System.out.println(dictionary_dev.get(kk[i]).toString());
                                          if(dictionary.get(kk[i]).toString().equals(dictionary_dev.get(kk[i]).toString())){
                                              count_correct++;
                                          }
                                      }
                                      System.out.println(count_correct);
                                      System.out.println("accuracy:"+count_correct/dictionary.size());
                  
                }
                //TEST PREDICTION
                else{

                                //open the testset and calculate for the instances id the most common senseTHEN WRITE 
					//IN A SECOND FILE.TXT
                                String FILENAME_test = "/home/lollo/Desktop/NLP/Homework2/TEST/test_data.txt";
                                PrintWriter writer = new PrintWriter("/home/lollo/Desktop/NLP/Homework2/TEST/results.txt", "UTF-8");
                                Scanner file_test = new Scanner(new FileReader(FILENAME_test));
                                Map dictionary_test = new HashMap();
                                int counter=0;
                                while (file_test.hasNextLine()) {
                                          String line=file_test.nextLine();
                                          String[] s=line.split(" ",0);
                                          //System.out.println(s[0]); 
                                          for(int i=0;i<s.length;i++){
                                          String[] k=s[i].split("\\|",0);
                                          //System.out.println(k[0]+" "+k[1]+" "+k[2]);

                                          if(k.length==4 ){
                                                  counter++;
                                                  //System.out.println(counter);
                                                  //System.out.println("f");
                                                  //System.out.println(k[0]+" "+k[1]+" "+k[2]+" "+k[3]);
                                                  if(babel_dict.containsKey(k[1])){
                                                      //dictionary_test.put(k[3], babel_dict.get(k[1]));
                                                      writer.println(k[3]+"\t"+babel_dict.get(k[1]));
                                                      System.out.println(k[3]+"\t"+babel_dict.get(k[1]));
                                                  }
                                                  else {
                                                      //System.out.println(k[1]);
                                                      try{
                                                            BabelSynset syn = bn.getSynsets(k[1], Language.EN).stream()
                                                                      .sorted(new BabelSynsetComparator(k[1], Language.EN))
                                                                      .findFirst().get(); 
                                                              BabelSynsetID value= syn.getID();
                                                           //dictionary_test.put(k[3],value );
                                                          System.out.println(k[3]+"\t"+value);
                                                          writer.print(k[3]+"\t");
                                                          writer.println(value);
                                                          babel_dict.put(k[1], value);
                                                      }
                                                      catch(Exception e){
                                                      String value="not found";
                                                         dictionary_test.put(k[3],value );
                                                          System.out.println(k[3]+"\t"+value);
                                                          writer.print(k[3]+"\t");
                                                          writer.println(value);
                                                          babel_dict.put(k[1], value);
                                                      }catch (OutOfMemoryError e){
                                                      String value="not found";
                                                         dictionary_test.put(k[3],value );
                                                          System.out.println(k[3]+"\t"+value);
                                                          writer.print(k[3]+"\t");
                                                          writer.println(value);
                                                          babel_dict.put(k[1], value);
                                                      }
                                                  }
                                                  }
                                              }
                                                      //dictionary_dev.put(s[0], s[1]);
                                                      //System.out.println(s[0]+" "+s[1]+" "+s[2]+" "+s[3]);                         
                                          }
                                System.out.println("testing finished");
                                writer.close();
                }
                  
                
        } catch (Exception e) {
	System.out.println(e);
                }
            
          
	}
	
	static public boolean isFromWordNet(String id)
	{
		return id.startsWith("bn:000") || id.startsWith("bn:0010") || id.startsWith("bn:0011");
	}
    
}
