import os
import json
from collections import defaultdict

import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.models import Phrases
from gensim.models.phrases import Phraser

from helper.cleaning_pipeline import Preprocessing
cleaning_pipeline_obj = Preprocessing()


class TopicModeling():

	def __init__(self):
		pass  


	def process_topics(self, topic_string):
		return [(topic.split('*')[0], topic.split('*')[1]) for topic in topic_string.replace('"', '').split('+')]


	def train(self, sentences_list, words_docs, number_of_clusters):
		### approach 3: tfidf + multigrams
		multigrams = Phrases(words_docs, min_count=1, threshold=5)
		multigrams_phraser = Phraser(multigrams)
		multigrams_tokens = [multigrams_phraser[sent] for sent in words_docs]
		id2word = corpora.Dictionary(multigrams_tokens)
		training_corpus = [id2word.doc2bow(text) for text in multigrams_tokens]

		# Have the corpus(term-doc matrix) and id2word (dictionary), need to specify no of topics and iteration
		lda_model = gensim.models.ldamodel.LdaModel(corpus=training_corpus,  # BoW
												   id2word=id2word, # vocanulary of the corpus
												   num_topics=number_of_clusters, 
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


	def get_topics_sentences(self, ldamodel, corpus, texts, model_dir):
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
		contents = pd.Series(texts)
		df_topic_sents_keywords = pd.concat([sent_topics_df, contents], axis=1)
		df_dominant_topic = df_topic_sents_keywords.reset_index()
		df_dominant_topic.columns = ['Document_No', 'Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
		df_dominant_topic.to_csv(os.path.join(model_dir, "cluster_sentence_topic_mapping.csv"))
		return df_dominant_topic
		

	def main(self, words_docs, cleaned_sentences, lang, model_dir, number_of_clusters):
		lda_model, corpus, id2word, new_topics = self.train(cleaned_sentences, words_docs, number_of_clusters)
		df_dominant_topic = self.get_topics_sentences(lda_model, corpus, cleaned_sentences, model_dir)
		return df_dominant_topic


