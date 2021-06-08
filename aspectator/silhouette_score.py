from sklearn_extra.cluster import KMedoids
import pandas as pd
from sklearn.metrics import silhouette_score

class Finding_the_best_cluster(object):
    
    def __init__(self, base_folder):
        self.base_folder = base_folder

    def silhouette_score(self, norm_aspect_array):

        cluster_multiplier = [mul*0.05 for mul in range(1, 20)]
        n_cluster = [int(norm_aspect_array.shape[0]*mul) for mul in cluster_multiplier]

        silhoutte_ls = []
        for a in range(len(n_cluster)):
            clusterer = KMedoids(n_clusters = n_cluster[a], random_state=0, method = 'pam').fit(norm_aspect_array)
            cluster_labels = clusterer.fit_predict(norm_aspect_array)
            silhouette_avg = silhouette_score(norm_aspect_array, cluster_labels)
            silhoutte_ls.append([cluster_multiplier[a], n_cluster[a], silhouette_avg])
            
        silhoutte_df =  pd.DataFrame(silhoutte_ls)
        return silhoutte_df.to_csv('silhouette_score.csv', index = False)