import os 
import json
import time
import numpy as np
from scipy import spatial
import pandas as pd
from flashtext import KeywordProcessor
keyword_processor = KeywordProcessor()
import gensim
from gensim.models import Phrases
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity


class TrainWord2Vec(object):

	def __init__(self):
		pass


	def trainW2V(self, words_docs, model_dir):
		start_time=time.time()
		# bigram_transformer = Phrases(words_docs)
		# bigram_transformer[words_docs]
		model = Word2Vec(sentences = words_docs, vector_size=32, window=3, min_count=1, workers=4)
		print("\n word2vec training time --- ", time.time() - start_time)
		# print("\n ocd --- ", model.wv['ocd'])
		# print("\n most similar to contamination --- ", model.wv.most_similar('contamination'))
		self.model = model 
		self.index2word_set = set(self.model.wv.index_to_key)
		self.num_features = self.model.vector_size


	def avg_feature_vector(self, sent_tokens):
	    feature_vec = np.zeros((self.num_features, ), dtype='float32')
	    n_words = 0	    
	    for word in sent_tokens:
	        if word in self.index2word_set:
	            n_words += 1
	            feature_vec = np.add(feature_vec, self.model.wv[word])
	    if (n_words > 0):
	        feature_vec = np.divide(feature_vec, n_words)
	    return feature_vec


	def corpus_word_check(self, list1):
		list1 = [word for word in list1 if word in self.index2word_set]
		return list1


	def get_score(self, sent_tokens, training_corpus, result_dict, dictionary_data):
		""" method 1: using cosine distance between averaged vectors
			result: not good
		"""
		# sims_vec = [ (assess_word[0], 1 - spatial.distance.cosine(self.avg_feature_vector(sent_tokens), self.avg_feature_vector(assess_word))) for assess_word in training_corpus]

		""" method 2: using wmdistance
		result: distance is not normalized between 0 to 1, hence score=1- dist is not a correct meansure
		"""
		# sims_vec = [ (assess_word[0], 1 - self.model.wv.wmdistance(sent_tokens, assess_word)) for assess_word in training_corpus]

		""" method 3: using w2v n_similarity 
			result: good
		"""
		sims_vec = [ (assess_word[0], self.model.wv.n_similarity(self.corpus_word_check(sent_tokens), self.corpus_word_check(assess_word))) for assess_word in training_corpus if self.corpus_word_check(sent_tokens) and self.corpus_word_check(assess_word)]
		sims_vec = sorted(sims_vec, key=lambda item: item[1], reverse=True)

		sent_labels = []
		for idx in range(len(sims_vec)):
			dict_key = ""
			assess_tool_label = self.keyword_processor.extract_keywords(sims_vec[idx][0])
			if assess_tool_label: dict_key = assess_tool_label[0]
			if dict_key and dict_key not in sent_labels:
				sent_labels.append(dict_key)
				index = len(sent_labels)
				key_word = "label_" + str(index)
				key_score = "score_" + str(index) 
				result_dict[key_word].append(dict_key)
				result_dict[key_score].append(sims_vec[idx][1])
			if len(sent_labels) == 3: break
		return result_dict


	def find_similarity(self, cleaned_sentences, words_docs, dictionary_data, model_dir):
		""" train wmd on dictionary data because we need labels for each sentence """
		training_corpus = []
		for key, val_list in dictionary_data.items():
			val_list.append(key)
			for word in val_list:
				training_corpus.append([word])
		
		result_dict = {"sent":[], "label_1":[], "score_1":[], "label_2":[], "score_2":[], "label_3":[], "score_3":[]}
		for idx in range(len(words_docs)):
			sent_tokens = words_docs[idx]
			result_dict["sent"].append(cleaned_sentences[idx])
			result_dict = self.get_score(sent_tokens, training_corpus, result_dict, dictionary_data) 
		return result_dict


	def save_labelled_data(self, result_dict, model_dir):
		""" save results in cvs """
		df = pd.DataFrame.from_dict(result_dict)
		df.to_csv(os.path.join(model_dir, "labelled_data_w2v.csv"))
		print("\n successfully saved labelled data in csv!")


	def main(self, cleaned_sentences, words_docs, dictionary_data, model_id, model_dir):
		st = time.time()
		print("\n number of sentences to train word2vec model --- ", len(words_docs))
		
		""" step 1: train word2vec model on all sentences """
		self.trainW2V(words_docs, model_dir)

		""" step 2: store dictionary_data in flashtext object """
		keyword_processor.add_keywords_from_dict(dictionary_data)
		self.keyword_processor = keyword_processor

		""" step 3: find similarity and label data """
		result_dict = self.find_similarity(cleaned_sentences, words_docs, dictionary_data, model_dir)

		""" step 4: save labelled data """
		result_dict = self.save_labelled_data(result_dict, model_dir)
		print("\n total_time : ", time.time() - st)

if __name__ == '__main__':
	obj = TrainWord2Vec()
	obj.main()
