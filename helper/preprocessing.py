import os
import json
import traceback

""" import scripts"""
from helper.cleaning_pipeline import Preprocessing
cleaning_pipeline_obj = Preprocessing()

from helper.find_synonyms import SynonymsGenerator
find_synonyms_obj = SynonymsGenerator()

import configuration_file as config


class ProcessText:

	def __init__(self):
		pass


	def main(self, sentences_list, lang):
		sentences_list = sentences_list[:config.num_of_sents]
		cleaned_sentences = cleaning_pipeline_obj.cleaning_pipeline(sentences_list, lang)
		cleaned_sentences = list(set(cleaned_sentences))
		if config.synonyms: cleaned_sentences = find_synonyms_obj.find_sentence_synonyms(cleaned_sentences)
		words_docs = [cleaning_pipeline_obj.get_lemma_tokens(sent, lang) for sent in cleaned_sentences]
		return cleaned_sentences, words_docs