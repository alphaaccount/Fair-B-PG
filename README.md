# Fair-B-PG

All the data files are processed data files obtained after running a GAT model (to be used by QPPG), along with the embeddings of different baselines is available at 
https://drive.google.com/drive/folders/1jPuqA3_DR4dIPtwckNy7FIVzJdQ329uS?usp=share_link    (a)

All the preproccesed data files needed for model training for various datasets is available at 
https://drive.google.com/drive/folders/1HL8gYg2mlKdgGKwban8GAomyoa7wUqVc?usp=share_link    (b)

1. Our main algorithms are at: 

Fair-B/Fair-B_Original/ContinuousFairness_NDCG/src_x_y/QPPG1.py files (where x,y are the sensitive attributes for datases 'Pokec-z' and 'NBA') 

Fair-B/Fair-B_Original/ContinuousFairness_NDCG/QPPG1*.py (where * is one of datasets 'Cora', 'Twitter' and 'Polblog')

The variants are QPPG1_AA.py [Adamic-Adar], QPPG1_ED.py [Edge Density], QPPG1_Community [Community Structure (CS)]

2. To run a Linear Programming solution (not discussed in the main paper)

Fair-B/Fair-B_Original/ContinuousFairness_NDCG/src_age_gender_bin/LPPG1.py files 

3. To run a particular case make sure to set the correct path for the following files in the end of the QPPG*.py files for a particular dataset 'X'
 
                    (Sensitive Attribute)                   Processesed File (a)                               Edge File (b)
 
                    (Gender,Age)                            allPokec_age_gender_bins2.csv                      pokec-z_edge.csv
 
                    (Region,Age)                            allPokec_region_age.csv                                  -do-
                    
                    (Gender, Region)                        allPokec_gender_region.csv                               -do-
                    
                    (Gender)                                allPokec_age_gender_bins2.csv                            -do-
                    
                    (Age)                                   allPokec_age_gender_bins2.csv                            -do-
                    
                    (Region)                                allPokec_region_age.csv                                  -do- 
                    
NBA                 (Country,Age)                           allNBA_country_age.csv                             nba_edge.csv 

                    (Country)                               allNBA_country_age.csv                                   -do-
                    
                    (Age)                                   allNBA_country_age.csv                                   -do-
                    
Political Blogs     (Political Party)                       allpolblog_party.csv                               pol-blog_edge.csv

Cora                (Topic)                                 allcora_topic.csv                                  ora_edge.csv

Twitter             (Political Opinion)                     alltwitter_opinion.csv                             twitter_edge.csv
