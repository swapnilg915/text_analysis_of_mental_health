import os 
import json
import time
import gensim
from gensim.models import Phrases
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity


class TrainWord2Vec(object):

	def __init__(self):
		pass


	def trainW2V(self, words_docs, model_dir):

		# size = embeddings dimesion
		# sg = 1-skipgram, 0-cbow
		# window = context words
		# min_count = ignore the words with frequency < 5
		# workers = number of worker threads
		# seed = the answer to the universe life nd everything

		start_time=time.time()
		# bigram_transformer = Phrases(words_docs)
		# bigram_transformer[words_docs]
		model = Word2Vec(sentences = words_docs, vector_size=32, window=3, min_count=1, workers=4)
		print("\n word2vec training time --- ", time.time() - start_time)
		# print("\n ocd --- ", model.wv['ocd'])
		# print("\n most similar to contamination --- ", model.wv.most_similar('contamination'))
		# model.save(os.path.join(model_dir, "word2vec.model"))
		return model


	def get_score(self, sent_tokens, instance_wmd, training_corpus):
		import pdb;pdb.set_trace()
		wmd_sims = instance_wmd[sent_tokens]
		wmd_sims = sorted(enumerate(wmd_sims), key=lambda item: -item[1])
		similar_docs = [(score, training_corpus[idx]) for idx, score in wmd_sims[:3]]
		return similar_docs


	def find_similarity(self, cleaned_sentences, words_docs, dictionary_data, model, model_dir):
		""" train wmd on dictionary data because we need labels for each sentence """
		training_corpus = []
		for key, val_list in dictionary_data.items():
			val_list.append(key)
			for word in val_list:
				training_corpus.append([word])
		
		instance_wmd = WmdSimilarity(training_corpus, model)
		model_path = os.path.join(model_dir, "wmd.model")
		instance_wmd.save(model_path)

		instance_wmd = gensim.similarities.docsim.Similarity.load(model_path)
		similarity_sentences = []
		similarity_labels = []
		
		for idx in range(len(words_docs)):
			sent_tokens = words_docs[idx]
			similar_docs = self.get_score(sent_tokens, instance_wmd, training_corpus) 
			similarity_sentences.append(cleaned_sentences[idx])
			similarity_labels.append(similar_docs)

		import pdb;pdb.set_trace()
		""" save results in cvs """
		labelled_data = defaultdict(list)
		labelled_data['sentences'].extend(similarity_sentences)
		labelled_data['labels'].extend(similarity_labels)
		df = pd.DataFrame(labelled_data)


	def load_w2v(self, model_dir):
		w2v_path = os.path.join(model_dir, "word2vec.model")
		model = Word2Vec.load(w2v_path)
		return model


	def main(self, cleaned_sentences, words_docs, dictionary_data, model_id, model_dir):
		print("\n number of sentences to train word2vec model --- ", len(words_docs))
		model = self.trainW2V(words_docs, model_dir)
		# model = self.load_w2v(model_dir)
		self.find_similarity(cleaned_sentences, words_docs, dictionary_data, model, model_dir)


if __name__ == '__main__':
	obj = TrainWord2Vec()
	obj.main()
