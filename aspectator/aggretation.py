
from pathlib import Path
import sys
sys.path.insert(1, r'C:/Users/Marselo/Documents/Github2/ABSA-thesis')


import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

from aspectator.aspectator import The_Aspectator


variable = {'aesthetics': ["appearance", "sound", "taste", "smell", "packaging", "design", "shapes", "styles", "size" ], 
            'personal_interaction': ["courtesy", "politeness", "courteousness", "helpfulness", "staff", "responsiveness", 
                                 "assurance", "empathy", "trust", "confidence", "safety",  "availability", "knowledge", "friendliness", "promptness", "professionalism"],
            'policy': ["return", 'refund', 'order' 'cancellation', 'repair', 'warranty', 'exchange', 'payment', 'operating hours', 'cards', 
                  'parking', 'delivery', 'shipping'],
            'physical_aspect' : ['layout', 'design', 'cleanliness', 'convenience', 'assortment', 'tills', 'cashiers', 'fitting rooms', 'restroom', 'shopping bags', 'catalogues', 'movement', 
                            'cart', 'facilities', 'check-out', 'website', "place", "location", 'trolley'],
            'durability': ['defect', 'malfunction', 'lifetime', 'expiration', 'endurance'],
            'performance': ['performance'],
            'service_reliability': ['promise', 'efficiency'],
            'price': ['price', 'affordability', 'value', 'worth'],
            'problem_solving': ['complaint', 'report', 'customer service', 'call center', 'support'],
            'meet_customer_needs': ['needs', 'requirement', 'wants', 'satisfaction', 'complacency']}

dimension_dict = {'high_quality': ['aesthetics', 'personal_interaction', 'policy', 'physical_aspect', 'durability', 'performance',
                             'service_reliability'],
            'good_value': ['price'],
            'stands_behind': ['problem_solving'],
            'meet_customer_needs': ['meet_customer_needs']}


class Aggregate(object):
    
    def __init__(self):
        pass

    def find_opinion_pairs(self, synset1, pair):
        brown_ic = wordnet_ic.ic('ic-brown.dat')
        if len(list(The_Aspectator().synset_generator(pair))):
            synsetScore = 0
            jcn_scores = []
            for synset2 in list(The_Aspectator().synset_generator(pair)):
                if synset1.pos() == synset2.pos() and (synset1.pos() == 'n' and synset2.pos() == 'n'):
                    synsetScore = wn.jcn_similarity(synset1, synset2, ic = brown_ic)

                    if synsetScore != None:
                        jcn_scores.append(synsetScore)

                    elif synset1.name().split(".")[0] == synset2.name().split(".")[0]:
                        synsetScore = 1
                        jcn_scores.append(synsetScore)
                    synsetScore = 0

            #generate the max value
            if(len(jcn_scores) > 0):
                max_score = max(jcn_scores)
                return max_score


    def matrix_creation(self, cluster_names_df):
        all_aspect_var = []
        for synset1 in cluster_names_df[2]:
            one_aspect = []
            for v in variable:
                cons_list = variable[v]
                for pair in cons_list:
                    jcn_score = Aggregate().find_opinion_pairs(synset1, pair)
                    one_aspect.append(jcn_score)
            all_aspect_var.append(one_aspect)
        return np.array(all_aspect_var)

    def assign_variable_dimension(self, cluster_names_df, variable_array):
        jcn_threshold = 0.3
        all_clusters = []
        for a in range(len(variable_array)):
            clusters = []
            for b in range(len(variable_array[a])):
                if variable_array[a][b] is not None and variable_array[a][b] >= jcn_threshold:
                    clus = 1
                else:
                    clus = 0
                clusters.append(clus)
            all_clusters.append(clusters)

        #assigning column names according to seed words
        col_names = []
        for a in variable:
            for b in variable[a]:
                col_names.append(b)
        variable_df = pd.DataFrame(data = all_clusters, index = cluster_names_df[0], columns = col_names)
        
        #Group the cluster into variable in the conceptual model
        variable_names = []
        for a in range(len(variable_df)):
            if sum(variable_df.iloc[a][variable['aesthetics']]) > 0:
                variable_names.append('aesthetics')
            elif sum(variable_df.iloc[a][variable['personal_interaction']]) > 0:
                variable_names.append('personal_interaction')
            elif sum(variable_df.iloc[a][variable['policy']]) > 0:
                variable_names.append('policy')
            elif sum(variable_df.iloc[a][variable['physical_aspect']]) > 0:
                variable_names.append('physical_aspect')
            elif sum(variable_df.iloc[a][variable['durability']]) > 0:
                variable_names.append('durability')
            elif sum(variable_df.iloc[a][variable['performance']]) > 0:
                variable_names.append('performance')
            elif sum(variable_df.iloc[a][variable['service_reliability']]) > 0:
                variable_names.append('service_reliability')
            elif sum(variable_df.iloc[a][variable['price']]) > 0:
                variable_names.append('price')
            elif sum(variable_df.iloc[a][variable['problem_solving']]) > 0:
                variable_names.append('problem_solving')
            elif sum(variable_df.iloc[a][variable['meet_customer_needs']]) > 0:
                variable_names.append('meet_customer_needs')
            else:
                variable_names.append('not_included')

        variable_df['variable'] = variable_names

        #Group the variable based on its dimension
        variable_df = variable_df.reset_index()
        dimensions = []
        for a in variable_df['variable']:
            if a in dimension_dict['high_quality']:
                dim = 'high_quality'
            elif a in dimension_dict['good_value']:
                dim = 'good_value'
            elif a in dimension_dict['stands_behind']:
                dim = 'stands_behind'
            elif a in dimension_dict['meet_customer_needs']:
                dim = 'meet_customer_needs'
            else:
                dim = 'not_included'
            dimensions.append(dim)
        variable_df['dimensions'] = dimensions

        return variable_df

    def aggregate_variable_dimension(self,sentiment_with_cluster_df, variable_dimension_df):
        variable_merge = pd.merge(sentiment_with_cluster_df, variable_dimension_df[[0, 'variable', 'dimensions']], left_on = 'cluster', right_on = 0, how = 'right')
        varible_groupby_df = variable_merge[['dimensions','variable', 'pos_or_neg']].groupby(by = ['dimensions', 'variable']).sum('pos_or_neg')/variable_merge[['dimensions', 'variable', 'pos_or_neg']].groupby(by = ['dimensions','variable']).count()*100
        varible_groupby_df = varible_groupby_df.reset_index()
        varible_groupby_df = varible_groupby_df[varible_groupby_df['variable'] != 'not_included']

        #aggregate in dimension level
        final_result_df = varible_groupby_df.groupby(by = 'dimensions').mean().reset_index()    
        return varible_groupby_df, final_result_df
