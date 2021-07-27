import os
import json

""" variables """
port = 5001
extensions_list = ['.pdf']

""" paths """
path = os.path.abspath(__file__)
BASE_PATH = os.path.dirname(path)
data_path = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "models")

if not os.path.exists(MODEL_PATH):
	os.makedirs(MODEL_PATH)

word2vec_model_path = "/home/swapnil/Projects/Embeddings/word2vec/GoogleNews-vectors-negative300.bin.gz"
w2v_words_to_load = 200000

num_of_sents = 10000
synonyms=False
data_labelling_synonyms=True