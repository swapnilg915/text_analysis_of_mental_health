import os
import json
import traceback

""" import scripts"""
from helper.data_labeller import LabelData
data_labeller_obj = LabelData()

from training.text_clustering import ClusterText
text_clustering_obj = ClusterText()

from training.topic_modeling import TopicModeling
topic_modeling_obj = TopicModeling()

import configuration_file as config


class Decider:

	def __init__(self):
		pass


	def main(self, cleaned_sentences, words_docs, lang, embedding_model, number_of_clusters, model_id, algo, model_dir, dictionary_data):
		""" this is the main execution function. All the methods will get called from here. """
		try:
			print("\n number of sentences to train : ", len(cleaned_sentences))

			""" step 2. based on input train the algorithm """
			if algo == "LDA":
				df_dominant_topic = topic_modeling_obj.main(words_docs, cleaned_sentences, lang, model_dir, number_of_clusters)
			elif algo == "KMeans":
				df_dominant_topic = text_clustering_obj.main(words_docs, cleaned_sentences, lang, model_dir, number_of_clusters, embedding_model, model_id)

			"""step 3: label data """
			data_labeller_obj.main(df_dominant_topic, dictionary_data, model_dir)

		except Exception as e:
			print("\n Error in main : ",e)
			print("\n Error details : ", traceback.format_exc())