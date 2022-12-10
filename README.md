# Fair-B-PG

All the data files are post processed data files along with the embeddings of different baselines is available at 
https://drive.google.com/drive/folders/1jPuqA3_DR4dIPtwckNy7FIVzJdQ329uS?usp=share_link

All the preproccesed data files needed for model training for various datasets is available at 
https://drive.google.com/drive/folders/1HL8gYg2mlKdgGKwban8GAomyoa7wUqVc?usp=share_link

1. Our main algorithms are at: 

Fair-B/Fair-B_Original/ContinuousFairness_NDCG/src_x_y/QPPG1.py files (where x,y are the sensitive attributes for datases 'Pokec-z' and 'NBA') 

Fair-B/Fair-B_Original/ContinuousFairness_NDCG/QPPG1*.py (where * is one of datasets 'Cora', 'Twitter' and 'Polblog')

The variants are QPPG1_AA.py [Adamic-Adar], QPPG1_ED.py [Edge Density], QPPG1_Community [Community Structure (CS)]

2. To run a Linear Programming solution (not discussed in the main paper)

Fair-B/Fair-B_Original/ContinuousFairness_NDCG/src_age_gender_bin/LPPG1.py files 

To run a particular case make sure to set the correct path in the end of the QPPG*.py files for a particular dataset 'X'.
