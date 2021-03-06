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


	def check_in_dictionary(self, keywords, total_lda_keywords, dictionary_data):
		label_dict = defaultdict()
		matched_keywords = []
		un_matched_keywords = []
		for assessment_topic, values_list in dictionary_data.items():
			values_list.append(assessment_topic)
			# values_list = [cleaning_obj.get_lemma(word, "en") for word in values_list]
			total_assessment_keywords = list(set(values_list))
			intersect = list(set(total_lda_keywords) & set(total_assessment_keywords))
			# print("\n assessment_topic : ", assessment_topic)
			# print("\n intersect : ", intersect)
			match_key = assessment_topic + "_matched_words"
			un_match_key = assessment_topic + "_unmatched_words"
			if intersect:
				label_dict[assessment_topic] = float(len(intersect)/10) * 100
				label_dict[match_key] = ", ".join(intersect)
				label_dict[un_match_key] = ", ".join(list(set(total_lda_keywords) - set(intersect)))
				# label_dict["matched_words"] = ", ".join(intersect)
				# label_dict["un-matched_words"] = ", ".join(list(set(total_lda_keywords) - set(intersect)))
			else:
				label_dict[assessment_topic] = 0
				label_dict[match_key] = ""
				label_dict[un_match_key] = ", ".join(keywords)
				# label_dict["matched_words"] = ""
				# label_dict["un-matched_words"] = ", ".join(keywords)

		return dict(label_dict)


	def get_labels_for_topics(self, df_dominant_topic, dictionary_data):
		labelled_sentences_dict = {}
		""" create unique df """
		unique_df = df_dominant_topic.drop_duplicates(subset=["Topic"])
		match_dict = defaultdict()
		for idx, row in unique_df.iterrows():
			print("\n cluster number : ", row["Topic"])
			keywords = [keyword.strip() for keyword in row["Keywords"].split(",")]
			total_lda_keywords = []
			if config.data_labelling_synonyms:
				for word in keywords:
					synonyms_dict = find_synonyms_obj.find_word_synonyms(word)
					words_list = [ [word] + syns_list for word, syns_list in synonyms_dict.items()]
					words_list = itertools.chain(*words_list)
					total_lda_keywords.extend(words_list)
			else:
				total_lda_keywords = keywords

			# import pdb;pdb.set_trace()
			# total_lda_keywords = [cleaning_obj.get_lemma(word, "en") for word in total_lda_keywords]
			label_dict = self.check_in_dictionary(keywords,  total_lda_keywords, dictionary_data)
			print("\n label_dict : ", label_dict)
			match_dict[str(int(row["Topic"]))] = label_dict
		match_dict = dict(match_dict)
		return match_dict
		

	def label_sentences(self, df_dominant_topic, topic_labels, model_dir):
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
