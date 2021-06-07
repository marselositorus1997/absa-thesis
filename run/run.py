""" This script runs the whole pipeline"""

from pathlib import Path
import sys
sys.path.insert(1, r'C:/Users/Marselo/Documents/Github2/ABSA-thesis') #path is specified to your main directory

import json
import datetime
import os
import warnings
warnings.filterwarnings('ignore')



##
from aspectator.load_data import DataLoader
from aspectator.clean_data import DataCleaner
from aspectator.aspectator import The_Aspectator
from aspectator.aggretation import Aggregate

def main():
    ## prepping
    # setting the run id
    run_id_start_time = datetime.datetime.now()
    print(f"starting with run at time {run_id_start_time}")
    

    f = open('./run/conf.json', 'r')
    conf = json.load(f)
    basefolder = conf['base_folder']
    

    #Load dataset
    sample_df = DataLoader(basefolder).load_data2('raw_scraped/aldi.csv')
    
    #Clean dataset
    clean_df = DataCleaner().remove_unwanted_characters(sample_df)
    token_sent_ls = DataCleaner().tokenize_sentence(clean_df)
    
    ####Execute The Aspectator Algorithm
    #Opinion pair extraction
    opinion_pairs_ls = The_Aspectator().generate_opinion_pair(token_sent_ls)
    cleaned_opinion_pairs_ls = The_Aspectator().clean_opinion_pair(opinion_pairs_ls, token_sent_ls)
    lemmatized_opinion_pair_ls = The_Aspectator().word_lemmatizer(cleaned_opinion_pairs_ls)
    #Aspect term clustering
    pairwise_matrix_npy = The_Aspectator().matrix_creation(lemmatized_opinion_pair_ls)[1]
    norm_pairwise_matrix_npy = The_Aspectator().normalization(pairwise_matrix_npy)
    all_synsets_ls = The_Aspectator().matrix_creation(lemmatized_opinion_pair_ls)[0]
    cluster_names_df = The_Aspectator().cluster_term(norm_pairwise_matrix_npy, all_synsets_ls, lemmatized_opinion_pair_ls)[0]
    label = The_Aspectator().cluster_term(norm_pairwise_matrix_npy, all_synsets_ls, lemmatized_opinion_pair_ls)[1]
    aspect_term_labelled_df = The_Aspectator().assign_cluster_name(lemmatized_opinion_pair_ls, cluster_names_df, label)
    
    #Rate product aspect
    sentiment_with_cluster_df = The_Aspectator().rate_product_aspect(cleaned_opinion_pairs_ls, lemmatized_opinion_pair_ls, aspect_term_labelled_df)
    
    ####Aggregation
    matrix_aggregation = Aggregate().matrix_creation(cluster_names_df)
    variable_dimension_df = Aggregate().assign_variable_dimension(cluster_names_df, matrix_aggregation)
    aggregated_df = Aggregate().aggregate_variable_dimension(sentiment_with_cluster_df, variable_dimension_df)[1]
    print(variable_dimension_df)
    print(aggregated_df)

if __name__ == "__main__":
    main()
