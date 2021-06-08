from nltk.util import flatten
import pandas as pd
import numpy as np
import re
from collections import Iterable
from sklearn import preprocessing
from sklearn_extra.cluster import KMedoids
### 
import nltk
#nltk.download('vader_lexicon')

import spacy
#!python -m spacy download en_core_web_lg

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nlp = spacy.load("en_core_web_lg") #Load the corpus
sid = SentimentIntensityAnalyzer()

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.stem.wordnet import WordNetLemmatizer

class The_Aspectator(object):
    
    def __init__(self):
        pass

    def find_opinion_pairs(self, doc):     
        cleaned_rule3_pairs = []
        rule3_pairs = []

        add_no_det = False
        add_neg_pfx = False
        for token in doc:
            if token.pos_ == "VERB":
                
                children = token.children
                M = "999"
                A = "999"
                add_neg_pfx = False
                for child in children:
                    
                    if child.dep_ == "ccomp" and child.pos_ != 'AUX' and  child.pos_ != 'ADV':
                        M = child.text
                    if child.dep_ == "acomp":
                        M = child.text

                    if (child.dep_ == "nsubj" and child.pos_ == "PROPN") or (child.dep_ == "nsubj" and child.pos_ == "NOUN"):
                        for w in child.ancestors:
                            for w in w.children:
                                if w.dep_ == "xcomp":
                                    A = child.text
                                    for w in w.children:
                                        if w.dep_ =="acomp":
                                            M = w.text

                                if w.dep_ != "xcomp" :
                                    to_be = ['is', 'am', 'are', 'were', 'was']
                                    if token.text not in to_be:
                                        A = child.text
                                        for w in w.children:
                                            if w.dep_ == "compound":
                                                comp = w.text
                                                A = comp + " " + A
                                            M = token.text
                    if child.dep_ == 'pobj' and child.pos_ == 'NOUN':
                        M = token.text
                        for w in child.ancestors:
                            for w in w.children:
                                if w.dep_ == 'advmod':
                                    M = w.text + " " + M
                        A = child.text
                        for w in child.children:
                            if w.dep_ == "compound":
                                comp = w.text
                                A = comp + " " + A


                    if child.dep_ == "nsubjpass" and child.pos_ == 'NOUN':
                        A = child.text
                        M = token.text
                        for w in child.children:
                            if w.dep_ == "compound":
                                comp = w.text
                                A = comp + " " + A   

                    if child.dep_ =="dobj" and child.pos_ != 'PRON' and child.pos_ != 'NUM' and child.pos_ != 'DET':
                        A = child.text
                        M = token.text
                        for w in child.children:
                            if w.dep_ == "compound":
                                comp = w.text
                                A = comp + " " + A                    
                            if w.dep_ == "amod":
                                M = w.text
                            for w in w.children:
                                if w.dep_ == "advmod":
                                    M = w.text + " " + M
                                    
                    if child.dep_ == 'prep':
                        M = token.text
                        for w in child.children:
                            if w.dep_ == 'pobj' and w.pos_ =='NOUN':
                                A = w.text
                                for w in w.children:
                                    if w.dep_ == "compound":
                                        comp = w.text
                                        A = comp + " " + A    
                
                    #Identify negation
                    if child.dep_ == "neg":
                        neg_pfx = "not"
                        add_neg_pfx = True
                    if child.dep_ =="aux" and child.pos_ == 'MD':
                        for w in child.ancestors:
                            for w in w.children:
                                if w.text in ('have'):
                                    add_neg_pfx = True
                                if w.dep_ == 'neg':
                                    add_neg_pfx = False
                                else:
                                    add_neg_pfx = True
                    if child.dep_ =="aux" and child.pos_ == 'VERB':
                        add_neg_pfx = True   
                        
                if add_neg_pfx == True and M != "999":
                    M = "not" + " " + M

                if M != "999" and A != "999" and sid.polarity_scores(M)['compound'] != 0:
                    rule3_pairs.append((A, M, sid.polarity_scores(M)['compound']))


            if token.pos_ == "AUX":
                children = token.children
                M = "999"
                A = "999"
                add_neg_pfx = False
                for child in children:
                    if child.dep_ == "acomp" and child.pos_ == "VERB":
                        M = child.text
                        for w in child.children:
                            if w.dep_ == "advmod":
                                M = w.text + " " +  M

                    if child.dep_ == "acomp" and child.pos_ != "VERB":
                        M = child.text
                        for w in child.children:
                            if w.dep_ == "advmod":
                                M = w.text + " " +  M  
                        for w in child.children: 
                            if w.dep_ == "prep":
                                for w in w.children:
                                    if w.dep_ == "pobj":
                                        A = w.text
                                        for w in w.children:
                                            if w.dep_ == "compound":
                                                comp = w.text
                                                A = comp + " " + A
                    if (child.dep_ == "nsubj" and child.pos_ == "NOUN") or (child.dep_ == "nsubj" and child.pos_ == "PROPN"):
                        A = child.text
                        for w in child.children:
                            if w.dep_ == "compound":
                                comp = w.text
                                A = comp + " " + A

                    #Identify negation
                    if child.dep_ == "neg":
                        neg_pfx = "not"
                        add_neg_pfx = True


                if add_neg_pfx == True and M != "999":
                    M = "not" + " " + M

                if M != "999" and A != "999" and sid.polarity_scores(M)['compound'] != 0:
                    rule3_pairs.append((A, M, sid.polarity_scores(M)['compound']))

            if token.pos_ == "NOUN":
                children = token.children
                M = "999"
                A = "999"
                add_neg_pfx = False
                for child in children:

                    if child.dep_ == "amod":
                        M = child.text
                        for w in child.children:
                            if w.dep_ == "advmod":
                                M = w.text + " " + child.text
                            

                        A = token.text
                        
                        for w in child.ancestors:
                            if w.pos_ == 'VERB':
                                for w in w.children:
                                    if w.dep_ == 'neg' and w.pos_ == 'PART':
                                        add_neg_pfx = True

                    
                    if child.dep_ == "compound" and A != "999":
                        comp = child.text
                        A = comp + " " + A
                        
                    if child.dep_ == "det" and child.text == "no":
                        add_no_det = True
                        no_det = child.text
                        M = no_det + " " + token.text
                
                    if child.dep_ == "prep" and add_no_det == True:
                        for w in child.children:
                            if w.dep_ == "pobj" and w.pos_ == 'NOUN':
                                A = w.text
                                if w.text == "am" or w.text == "pm": #time
                                    A = 'time'
                                for w in w.children:
                                    if w.dep_ == "compound":
                                        comp = w.text
                                        A = comp + " " + A
                    if child.dep_ =='relcl':
                        A = token.text
                        for w in child.children:
                            ls_of_dep = [w.dep_ for w in child.children]
                            if "nsubj" in ls_of_dep:
                                M = child.text

                            if "aux" in ls_of_dep:
                                if w.dep_ == 'dobj':
                                    A = w.text

                #Identify negation
                if add_neg_pfx == True and M != "999":
                    M = "not" + " " + M

                if M != "999" and A != "999" and sid.polarity_scores(M)['compound'] != 0:
                    rule3_pairs.append((A, M, sid.polarity_scores(M)['compound']))
                        
            if token.pos_ == "ADJ":
                children = token.children
                M = token.text
                A = "999"
                add_neg_pfx = False
                for child in children:

                    if child.dep_ == "advmod":
                        M = child.text + " " + M


                    if child.dep_ == "prep":
                        for w in child.children:
                            if w.dep_ == "pobj" and w.pos_ in ('NOUN', 'PROPN'):
                                A = w.text
                                for w in w.children:
                                    if w.dep_ =="compound":
                                        comp = w.text
                                        A = comp + " " + A
                            

                    for w in child.ancestors:
                        if w.pos_ == 'AUX':
                            for w in w.children:
                                if w.dep_ =='neg':
                                    add_neg_pfx = True

                        if w.pos_ == 'VERB':
                            for w in w.children:
                                if w.dep_ =='neg':
                                    add_neg_pfx = True
                    #Identify negation
                    if child.dep_ == "neg":
                        neg_pfx = "not"
                        add_neg_pfx = True

                if add_neg_pfx == True and M != "999":
                    M = "not" + " " + M

                if M != "999" and A != "999" and sid.polarity_scores(M)['compound'] != 0:
                    rule3_pairs.append((A, M, sid.polarity_scores(M)['compound']))
                    
        #Removing duplicates            
        for ls in rule3_pairs:
            if ls not in cleaned_rule3_pairs:
                cleaned_rule3_pairs.append(ls)
        return cleaned_rule3_pairs

    def generate_opinion_pair(self, clean_row_tokenized):
        opinion_pairs_ls = []
        for sent in clean_row_tokenized:
            sent = ' '.join(re.findall(r"[a-zA-Z,!']+", sent))
            opinion_pair = The_Aspectator().find_opinion_pairs(nlp(sent))
            opinion_pairs_ls.append(opinion_pair)
        return opinion_pairs_ls

    def clean_opinion_pair(self, opinion_pairs_ls, token_sent_ls):
        
        ##Transform the list into dataframe so we can name it
        result_df = pd.DataFrame(token_sent_ls)
        result_df['opinion_pairs'] = opinion_pairs_ls   

        ##Select only the opinion pair which bear sentiment score
        sent_opinion_pairs = []
        for a in range(len(result_df)):
            opin_pairs = []
            for b in range(len(result_df['opinion_pairs'][a])):
                if result_df['opinion_pairs'][a]:
                    if len(result_df['opinion_pairs'][a]) >= 1:
                        if result_df['opinion_pairs'][a][b][2] != 0.0:
                            opin_pair = result_df['opinion_pairs'][a][b]
                            opin_pairs.append(opin_pair)
                        sent_opinion_pairs.append([result_df[0][a], opin_pairs])

        ##Remove the duplicate sentences
        evaluation_df = pd.DataFrame(sent_opinion_pairs).drop_duplicates(subset = 0, keep = 'first').reset_index()
        
        ##Generate a clean list of opinion pairs
        opinion_pairs_ls = evaluation_df[1].tolist()
        clean_opinion_pairs_ls = []
        for i in range(len(opinion_pairs_ls)):
            if len(opinion_pairs_ls[i]) != 0:
                for j in range(len(opinion_pairs_ls[i])):
                    if opinion_pairs_ls[i][j][2] != 0.0:                
                        clean_opinion_pairs_ls.append(opinion_pairs_ls[i][j]) 


        ##Further cleaning the opinion pair list
        cleaned_opinion_pair = []
        for row in range(len(clean_opinion_pairs_ls)):
            cleaned_aspect =  ' '.join(re.findall(r"[a-zA-Z]+", clean_opinion_pairs_ls[row][0]))
            cleaned_aspect = cleaned_aspect.lower()
            
            
            opinion =  ' '.join(re.findall(r"[a-zA-Z]+", clean_opinion_pairs_ls[row][1]))
            opinion = opinion.lower()
            cleaned_opinion_pair.append((cleaned_aspect, opinion, clean_opinion_pairs_ls[row][2]))
        return cleaned_opinion_pair

    def flatten(self, ls):
        for item in ls:
            if isinstance(item, Iterable) and not isinstance(item, str):
                for x in flatten(item):
                    yield x
            else:        
                yield item 

    def synset_generator(self, extracted_term):
        ls_synsets = []
        for word in extracted_term.split():
            ls_synsets.append(wn.synsets(word))
            ls_synsets = list(flatten(ls_synsets))
            
        return ls_synsets
    
    def jcn_score(self, pair1, pair2):
        brown_ic = wordnet_ic.ic('ic-brown.dat') # Wordnet information content file
        # For each synset in the first sentence...
        if len(list(The_Aspectator.synset_generator(self, pair1))) != 0 and  len(list(The_Aspectator.synset_generator(self, pair2))) != 0:

            jcn_scores = []
            syn1 = []
            syn2 = []
            for synset1 in list(The_Aspectator.synset_generator(self, pair1)):

                max_score = 0
                synsetScore = 0

                # For each synset in the second sentence...
                for synset2 in list(The_Aspectator.synset_generator(self, pair2)):


                    # Only compare synsets with the same POS tag. Word to word knowledge
                    # measures cannot be applied across different POS tags.
                    if (synset1.pos() == synset2.pos()) and (synset1.pos() == 'n' and synset2.pos() == 'n'):


                        # Note below is the call to path_similarity mentioned above. 
                        synsetScore = wn.jcn_similarity(synset1, synset2, ic = brown_ic, verbose = False)

                        if synsetScore != None:
                            #print("Path Score %0.2f: %s vs. %s" % (synsetScore, synset1, synset2))
                            jcn_scores.append(synsetScore)
                            syn1.append(synset1)
                            syn2.append(synset2)

                        # If there are no similarity results but the SAME WORD is being
                        # compared then it gives a max score of 1.
                        elif synset1.name().split(".")[0] == synset2.name().split(".")[0]:
                            synsetScore = 1
                            #print("Path MAX-Score %0.2f: %s vs. %s" % (synsetScore, synset1, synset2))
                            jcn_scores.append(synsetScore)
                            syn1.append(synset1)
                            syn2.append(synset2)


            #generate the max value
            if(len(jcn_scores) > 0):
                max_score = max(jcn_scores)


            #create temporary list
            temp_list = list (zip(syn1, syn2, jcn_scores ))

            highest_synset = []
            for a in range(len(temp_list)):
                for b in range(len(temp_list[a])):
                    if temp_list[a][2] == max_score:
                        highest_synset = temp_list[a]
                        return highest_synset

    def word_lemmatizer(self, clean_opinion_pairs_ls):
        lemmatizer  = WordNetLemmatizer()
        clean_lemmatized_ls = []
        for ls in range(len(clean_opinion_pairs_ls)):
            for words in [clean_opinion_pairs_ls[ls][0]]:
                clean_word = re.findall(r"[a-zA-Z]+", words)
                lemmatized = [lemmatizer.lemmatize(word) for word in clean_word]
                lemmatized = ' '.join(lemmatized)
                lemmatized = lemmatized.lower()
                clean_lemmatized_ls.append(lemmatized)
        clean_lemmatized_ls = list(zip(clean_opinion_pairs_ls, clean_lemmatized_ls))
        return clean_lemmatized_ls

    def matrix_creation(self, clean_lemmatized_ls):
        all_aspects = []
        all_synsets = []

        for ls1 in range(len(clean_lemmatized_ls)):
            one_aspect = []
            aspect1 = clean_lemmatized_ls[ls1][1]
            for ls2 in range(len(clean_lemmatized_ls)):
                aspect2 = clean_lemmatized_ls[ls2][1]
                if The_Aspectator.jcn_score(self, aspect1, aspect2) is None:
                    one_aspect.append(0)
                elif The_Aspectator.jcn_score(self, aspect1, aspect2):
                    syn1, syn2, score = The_Aspectator.jcn_score(self, aspect1, aspect2)

                    if score >= 1:
                        one_aspect.append(1)
                    else:
                        one_aspect.append(score)
            all_synsets.append(syn1)
            all_aspects.append(one_aspect) 
        return all_synsets, all_aspects

    def normalization (self, pairwise_matrix):
        aspect_array = np.array(pairwise_matrix)
        norm_aspect_array = preprocessing.normalize(aspect_array) 
        return norm_aspect_array

    def cluster_term (self, matrix_array, all_synsets_ls, lemmatized_opinion_pairs_ls):
        ##Generate label for each centroid/cluster
        n_cluster = int(matrix_array.shape[0]*0.5) #0.5 is derived from the highest silhouette score
        kmedoids = KMedoids(n_clusters = n_cluster, random_state=0, method = 'pam').fit(matrix_array)
        label = list(kmedoids.labels_)

        ##Create cluster_names
        cluster_names = []
        center = kmedoids.cluster_centers_
        center_ls = center.tolist()

        ##remove duplicate values in the centroid
        no_dup = []
        for elem in center_ls:
            if elem not in no_dup:
                no_dup.append(elem)

        non_dup_center = list(set(tuple(sub) for sub in center))
        for a in range(len(matrix_array)):
            for b in range(len(non_dup_center)):
                temp_array = matrix_array[a] == non_dup_center[b]
                if False not in temp_array:
                    if sum(matrix_array[a]) == 0:
                        cluster_names.append(["unidentified", label[a], all_synsets_ls[a]]) #unidentified is for all terms which dont bear any meaning or dont exist in the lexicon            
                    else:
                        cluster_name = lemmatized_opinion_pairs_ls[a][1]
                        cluster_names.append([cluster_name, label[a], all_synsets_ls[a]])

        #remove duplicate values in cluster_names
        cluster_names_df = pd.DataFrame(cluster_names).drop_duplicates(subset = 1, keep = 'first')
        return cluster_names_df, label

    def assign_cluster_name(self, lemmatized_opinion_pair_ls, cluster_names_df, label):
        aspect_term_label = []
        for a in range(len(lemmatized_opinion_pair_ls)):
            aspect_term_label.append(lemmatized_opinion_pair_ls[a][1])
            
        aspect_term_label = pd.DataFrame(list(zip(aspect_term_label, label)))
        aspect_term_label = pd.merge(aspect_term_label, cluster_names_df, left_on = 1, right_on = 1, how = 'left')     
        return aspect_term_label

    def rate_product_aspect(self,  cleaned_opinion_pairs_ls, lemmatized_opinion_pair_ls, aspect_term_labelled_df):
        sentiment_with_cluster_df = pd.DataFrame(cleaned_opinion_pairs_ls)
        sentiment_with_cluster_df[0] = pd.DataFrame(lemmatized_opinion_pair_ls)[1]
        sentiment_with_cluster_df['cluster'] = aspect_term_labelled_df['0_y']
        sentiment_with_cluster_df['pos_or_neg'] = [1 if float(word) > 0 else 0 for word in sentiment_with_cluster_df[2]]
        return sentiment_with_cluster_df