import os
import json
import time
import traceback
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

""" import gensim modules"""
import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

""" import sklearn modules """
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn import metrics

from sklearn.model_selection import train_test_split

from spacy.lang.en import English
spacy_en = English(disable=['parser', 'ner'])
from spacy.lang.en.stop_words import STOP_WORDS as stopwords_en
stopwords_en = list(stopwords_en)
mapping_dict = {"en":[stopwords_en, spacy_en]} 

""" import scripts"""
from helper.cleaning_pipeline import Preprocessing
cleaning_pipeline_obj = Preprocessing()

from training.topic_modeling import TopicModeling
topic_modeling_obj = TopicModeling()

import configuration_file as config


class ClusterText():

	def __init__(self):
		self.oov_list=[]


	def visualize_data(self, pred_dict, lang, model_id, sentences, model_dir):
		dir_name = os.path.join(model_dir, "wordcloud")
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
			print("\n dir created --- ",dir_name)

		for k,v in pred_dict.items():
			tokens_list = " ".join([word for sent in sentences for word in sent.split()])
			wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = mapping_dict[lang][0], min_font_size = 10).generate(tokens_list)
			plt.figure(figsize = (8, 8), facecolor = None) 
			plt.imshow(wordcloud)
			plt.axis("off") 
			plt.tight_layout(pad = 0)
			fig_name= "cluster_" + str(k) + ".png"
			plt.savefig(os.path.join(dir_name, fig_name))
			# plt.clf()
		return pred_dict


	def train_model(self, sentences, text_vector, number_of_clusters, lang, model_id, model_dir):
		try:
			""" training """
			st = time.time()
			model = KMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=10, n_init=10, random_state=42)
			# km = AgglomerativeClustering(n_clusters=number_of_clusters, affinity='euclidean')
			# km = DBSCAN(eps=0.4, min_samples=2)
			print("\n training started ... ")
			model.fit(text_vector)
			print("\n training time --- ", time.time() - st)

			""" Insights """
			order_centroids = []
			terms = []
			order_centroids = model.cluster_centers_.argsort()[:, ::-1]

			""" classify each sentence to its respective cluster """
			pred = model.labels_
			pred_dict = defaultdict()
			for idx in range(len(pred)):
				if str(pred[idx]) not in pred_dict: pred_dict[str(pred[idx])] = []
				if sentences[idx] not in pred_dict[str(pred[idx])]: pred_dict[str(pred[idx])].append(sentences[idx])

			pred_dict = self.visualize_data(pred_dict, lang, model_id, sentences, model_dir)

			""" write clustering results to json """
			with open(os.path.join(model_dir, "clustering_results.json"), "w+") as fs:
				fs.write(json.dumps(pred_dict, indent=4))
		except Exception as e:
			print("\n Error in train_model : ",e)
			print("\n Error details : ", traceback.format_exc())
		return model, pred_dict


	def evaulate_clusters(self, pred_dict, model_dir):
		""" check the top important terms for each cluster """
		clustering_dict = {"Topic":[], "Text":[], "Keywords": []}
		for cluster_num, sents_list in pred_dict.items():
			print("\n cluster number : ", cluster_num)
			print("\n number of sents : ", len(sents_list))
			tfidf_vec = TfidfVectorizer(use_idf=True, sublinear_tf=True, max_df=0.8, max_features=20, ngram_range=(1,5), min_df=1)
			X_tfidf = tfidf_vec.fit_transform(sents_list).toarray()
			total_tfidf = tfidf_vec.get_feature_names()
			for sent in sents_list:
				clustering_dict["Topic"].append(cluster_num)
				clustering_dict["Text"].append(sent)
				clustering_dict["Keywords"].append(",".join(total_tfidf))
		""" save the clusters to csv file """
		df_dominant_topic = defaultdict(list) 
		df_dominant_topic["Topic"] = clustering_dict["Topic"]
		df_dominant_topic["Text"] = clustering_dict["Text"]
		df_dominant_topic["Keywords"] = clustering_dict["Keywords"]
		df_dominant_topic = pd.DataFrame(df_dominant_topic)
		df_dominant_topic.to_csv(os.path.join(model_dir, "cluster_sentence_topic_mapping.csv"))
		return df_dominant_topic


	def load_word2vec(self):
		st = time.time()
		self.model = KeyedVectors.load_word2vec_format(config.word2vec_model_path, limit=config.w2v_words_to_load, binary=True)
		print("\n time taken to load the word2vec model --- ", time.time() - st)


	def create_w2v_vectors(self, list_of_docs):
		self.load_word2vec()
		features = []
		for tokens in list_of_docs:
			zero_vector = np.zeros(self.model.vector_size)
			vectors = []
			for token in tokens:
				if token in self.model:
					try:
						vectors.append(self.model[token])
					except KeyError:
						self.oov_list.append(token)
						continue
			if vectors:
				vectors = np.asarray(vectors)
				avg_vec = vectors.mean(axis=0)
				features.append(avg_vec)
			else:
				features.append(zero_vector)
		print("\n out of vocab words : ", len(self.oov_list))
		return features
		

	def create_tfidf_vectors(self, sentences):
		try:
			tfidf_vec = TfidfVectorizer(use_idf=True, sublinear_tf=True, max_df=0.8, max_features=5000, ngram_range=(1,5), min_df=5)
			X = tfidf_vec.fit_transform(sentences).toarray()
			print("\n vector shape : ", X.shape)
		except Exception as e:
			print("\n Error in create_tfidf_vectors : ",e)
			print("\n Error details : ", traceback.format_exc())
		return X


	def get_new_snentences(self, sent, tok, lemmas_for_synset):
		new_sents = []
		for syn in lemmas_for_synset:
			sent = sent.replace(tok.text, syn)
			new_sents.append(sent)
		return new_sents


	def find_synonyms(self, cleaned_sentences):
		nlp = spacy.load('en_core_web_sm')
		nlp.add_pipe(WordnetAnnotator(nlp.lang))
		new_cleaned_sentences = []
		for sent in cleaned_sentences:
			sentence = nlp(sent)
			for tok in sentence:
				synsets = tok._.wordnet.synsets()
				lemmas_for_synset = list(set([lemma for s in synsets for lemma in s.lemma_names()]))
				new_cleaned_sentences.extend(self.get_new_snentences(sent, tok, lemmas_for_synset))
		print("\n number of sents before adding synonyms : ", len(cleaned_sentences))
		print("\n sents after adding synonyms : ", len(new_cleaned_sentences))
		return new_cleaned_sentences


	def main(self, words_docs, cleaned_sentences, lang, model_dir, number_of_clusters, embedding_model, model_id):
		""" this is the main execution function. All the methods will get called from here. """
		try:
			if embedding_model == "tfidf": text_vector = self.create_tfidf_vectors(cleaned_sentences)
			elif embedding_model == "word2vec": text_vector = self.create_w2v_vectors(words_docs)
			model, pred_dict = self.train_model(cleaned_sentences, text_vector, number_of_clusters, lang, model_id, model_dir)
			df_dominant_topic = self.evaulate_clusters(pred_dict, model_dir)

		except Exception as e:
			print("\n Error in main : ",e)
			print("\n Error details : ", traceback.format_exc())

		return df_dominant_topic


if __name__ == "__main__":
	obj = ClusterText()
