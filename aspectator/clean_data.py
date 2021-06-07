import pandas as pd
from nltk import sent_tokenize
import re

class DataCleaner(object):

    def __init__(self):
        pass

    def remove_unwanted_characters(self, df):
        sample_df = df.dropna().reset_index()
        for row in range(len(sample_df)):
            sample_df['main_comment'][row] = re.sub('[^A-Za-z0-9 ,.!?]+', '', sample_df['main_comment'][row])
            sample_df['main_comment'][row] = sample_df['main_comment'][row].replace('...','.')
            sample_df['main_comment'][row] = sample_df['main_comment'][row].replace('..','.')
            sample_df['main_comment'][row] = sample_df['main_comment'][row].replace('!!!','!')
            sample_df['main_comment'][row] = sample_df['main_comment'][row].replace('!!','!')
            sample_df['main_comment'][row] = sample_df['main_comment'][row].replace('.','. ') 
            return sample_df

    def tokenize_sentence(self, df):
        row_tokenized = []
        for row in range(len(df)):
            row_tokenized += sent_tokenize(df['main_comment'][row])

        #remove the list with single sentence (we need opinion pairs)
        clean_row_tokenized = []
        for sent in row_tokenized:
            if len(re.findall(r'\w+', sent)) > 1:
                clean_row_tokenized.append(sent)
        return clean_row_tokenized
