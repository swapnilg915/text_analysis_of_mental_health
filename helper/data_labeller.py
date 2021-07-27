import os
import json
import itertools
from collections import defaultdict

import pandas as pd 
""" import scripts """
from helper.find_synonyms import SynonymsGenerator
find_synonyms_obj = SynonymsGenerator()

from helper.cleaning_pipeline import Preprocessing
cleaning_obj = Preprocessing()

import configuration_file as config


class LabelData:


	def __init__(self):
		pass


	def check_in_dictionary(self, total_topic_keywords, dictionary_data):
		label_dict = defaultdict()
		for assessment_topic, values_list in dictionary_data.items():
			values_list.append(assessment_topic)
			# values_list = [cleaning_obj.get_lemma(word, "en") for word in values_list]
			total_assessment_keywords = list(set(values_list))
			intersect = list(set(total_topic_keywords) & set(total_assessment_keywords))
			if intersect:
				label_dict[assessment_topic] = float(len(intersect)/len(total_assessment_keywords)) * 100
			# else: label_dict[assessment_topic] = 0
		return dict(label_dict)


	def get_labels_for_topics(self, df_dominant_topic, dictionary_data):
		labelled_sentences_dict = {}
		""" create unique df """
		unique_df = df_dominant_topic.drop_duplicates(subset=["Topic"])
		match_dict = defaultdict()
		for idx, row in unique_df.iterrows():
			keywords = [keyword.strip() for keyword in row["Keywords"].split(",")]
			total_topic_keywords = []
			if config.data_labelling_synonyms:
				for word in keywords:
					synonyms_dict = find_synonyms_obj.find_word_synonyms(word)
					words_list = [ [word] + syns_list for word, syns_list in synonyms_dict.items()]
					words_list = itertools.chain(*words_list)
					total_topic_keywords.extend(words_list)
			else:
				total_topic_keywords = keywords

			# total_topic_keywords = [cleaning_obj.get_lemma(word, "en") for word in total_topic_keywords]
			label_dict = self.check_in_dictionary(total_topic_keywords, dictionary_data)
			match_dict[str(int(row["Topic"]))] = label_dict
		match_dict = dict(match_dict)
		return match_dict
		

	def label_sentences(self, df_dominant_topic, topic_labels, model_dir):
		sents_list = list(df_dominant_topic.Text)
		topics = list(df_dominant_topic.Topic)
		multi_labels = []
		for idx, row in df_dominant_topic.iterrows():
			labels_dict = topic_labels[str(int(row["Topic"]))]
			multi_labels.append(labels_dict)

		multi_label_df = pd.DataFrame(multi_labels)
		final_df = pd.concat([df_dominant_topic, multi_label_df], axis = 1)
		final_df.to_csv(os.path.join(model_dir, "labelled_sentences.csv"))
		print("\n saved csv successfully : ")


	def main(self, df_dominant_topic, dictionary_data, model_dir):
		topic_labels = self.get_labels_for_topics(df_dominant_topic, dictionary_data)
		self.label_sentences(df_dominant_topic, topic_labels, model_dir)
