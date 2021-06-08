# absa-thesis
This repo is dedicated to using the Aspectator Algorithm (Bancken et al., 2014) to measure corporate reputation in retail industry. The Aspectator is supervised learning in the domain 
of Aspect Based Sentiment Analysis (ABSA) which means it employs lexicon/dictionary as a source of sentiment score and similarity score. In general, ABSA is initiated with extracting aspects from the reviews text (opinion target extraction - OTE), identifying the entity that is referred to by the aspect (aspect category detection - ACD), 
and ended with classifying the opinion polarity towards the aspect (sentiment polarity - SP). These three steps are included in this repo as well. The reached accuracy of the Aspectator
algorithm is 90% whereas the final accuracy of the model (including the aggregation) is 77%. This algorithm allows us to use online data (i.e. online review) to get the reputation 
of a retail company as opposed to the traditional method which employs sharing questionnaire. This conventional method is exhaustive and laborous. Moreover, a company can get
their corporate reputation quickly and act on it, especially in today's digital world. 
