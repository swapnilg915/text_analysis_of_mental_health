import os
import json
from collections import defaultdict

import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.models import Phrases
from gensim.models.phrases import Phraser

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from helper.cleaning_pipeline import Preprocessing
cleaning_pipeline_obj = Preprocessing()


class TopicModeling():

	def __init__(self):
		pass  


	def visualize(self, lda_model, corpus, id2word):
		#Creating Topic Distance Visualization 
		vis = gensimvis.prepare(lda_model, corpus, id2word)
		print(vis)



	def process_topics(self, topic_string):
		return [(topic.split('*')[0], topic.split('*')[1]) for topic in topic_string.replace('"', '').split('+')]


	def train(self, sentences_list, words_docs):
		# Create Dictionary as Gensim requires dic of all terms and their respective 
		# Create document term matrix

		approach = "tfidf_multigram" # tfidf_multigram / tfidf / BOW

		if approach == "BOW":
			### approach 1: BOW (Term Document Frequency)
			id2word = corpora.Dictionary(words_docs) 
			training_corpus = [id2word.doc2bow(words_doc) for words_doc in words_docs]
			print(training_corpus[:1])

		elif approach == "tfidf":
			### approach 2: Gensim's TFIDF
			id2word = corpora.Dictionary(words_docs) 
			corpus = [id2word.doc2bow(words_doc) for words_doc in words_docs]
			tfidf = gensim.models.TfidfModel(corpus)
			training_corpus = tfidf[corpus]
			print(training_corpus[:1])

		elif approach == "tfidf_multigram":
			### approach 3: tfidf + multigrams
			multigrams = Phrases(words_docs, min_count=1, threshold=5, delimiter=b' ')
			multigrams_phraser = Phraser(multigrams)
			multigrams_tokens = [multigrams_phraser[sent] for sent in words_docs]
			id2word = corpora.Dictionary(multigrams_tokens)
			training_corpus = [id2word.doc2bow(text) for text in multigrams_tokens]


		# Have the corpus(term-doc matrix) and id2word (dictionary), need to specify no of topics and iteration
		lda_model = gensim.models.ldamodel.LdaModel(corpus=training_corpus,  # BoW
												   id2word=id2word, # vocanulary of the corpus
												   num_topics=10, 
												   random_state=100, 
												   update_every=1, 
												   chunksize=100, 
												   passes=50,
												   alpha='auto',
												   per_word_topics=True)
		topics = lda_model.print_topics()
		print("\n all topics: \n")
		new_topics = [(tpl[0], self.process_topics(tpl[1])) for tpl in topics]
		for topic in new_topics:
			print("\n", topic[0], "=> ", topic[1])
		return lda_model, training_corpus, id2word, new_topics


	def format_topics_sentences(self, ldamodel, corpus, texts, model_dir):
		# Init output
		sent_topics_df = pd.DataFrame()

		# Get main topic in each document
		for i, row in enumerate(ldamodel[corpus]):
			row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
			# Get the Dominant topic, Perc Contribution and Keywords for each document
			for j, (topic_num, prop_topic) in enumerate(row):
				if j == 0:  # => dominant topic
					wp = ldamodel.show_topic(topic_num)
					topic_keywords = ", ".join([word for word, prop in wp])
					sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
				else:
					break
		sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']


		# Add original text to the end of the output
		contents = pd.Series(texts)
		df_topic_sents_keywords = pd.concat([sent_topics_df, contents], axis=1)
		df_dominant_topic = df_topic_sents_keywords.reset_index()
		df_dominant_topic.columns = ['Document_No', 'Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
		df_dominant_topic.to_csv(os.path.join(model_dir, "cluster_sentence_topic_mapping.csv"))
		return df_dominant_topic
		


	def get_label(self, clusters_dict, sent):
		for cluster, topics_list in clusters_dict.items():
			if len(set(topics_list) & set(sent.split())): return cluster
		return "other"


	def label_data(self, sentences_list, new_topics, clusters_dict, model_dir):
		""" save labelled data in excel"""
		labeled_dict = {sent:self.get_label(clusters_dict, sent) for sent in sentences_list}

		""" save labelled data in csv """
		labelled_dict = defaultdict(list)
		labelled_dict["comments"].extend(labeled_dict.keys())
		labelled_dict["labels"].extend(labeled_dict.values())
		labelled_df = pd.DataFrame(labelled_dict)
		labelled_df.to_csv(os.path.join(model_dir, "labelled_data.csv"))


	def save_topics(self, new_topics, model_dir):
		clusters_dict = {tpl[0]: [small_tpl[1].strip() for small_tpl in tpl[1] ] for tpl in new_topics }
		with open(os.path.join(model_dir, "clustering_topics.json"), "w+") as js:
			js.write(json.dumps(clusters_dict, indent=4))
			print("\n clusters topics saved successfully in json !!!")
		return clusters_dict


	def process_data(self, sentences_list, lang):
		return [cleaning_pipeline_obj.get_lemma_tokens(sent, lang) for sent in sentences_list]


	def main(self, sentences_list, lang, model_dir):
		sentences_list = sentences_list[:3000]
		words_docs = self.process_data(sentences_list, lang)
		lda_model, corpus, id2word, new_topics = self.train(sentences_list, words_docs)
		# clusters_dict = self.save_topics(new_topics, model_dir)
		# self.label_data(sentences_list, new_topics, clusters_dict, model_dir)
		# self.visualize(lda_model, corpus, id2word)	
		self.format_topics_sentences(lda_model, corpus, sentences_list, model_dir)


