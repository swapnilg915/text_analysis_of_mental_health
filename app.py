import os
import re
import json
import time
import traceback
import numpy as np
import pandas as pd

from flasgger import Swagger, swag_from
from flask import Flask, request, jsonify, render_template

from helper.preprocessing import ProcessText
preprocessing_obj = ProcessText()

from helper.decider import Decider
decider_obj = Decider()

from training.word2vec_training import TrainWord2Vec
train_w2v_obj = TrainWord2Vec()

import configuration_file as cf

# for reproducible results
np.random.seed(777)

app = Flask(__name__)
app.debug = True
app.url_map.strict_slashes = False
Swagger(app)

""" Training """
@app.route("/clustering", methods=['POST'])
@swag_from("swagger_files/clustering_training_api.yml")
def clustering():
	res_json = {"status":"", "model_id":"", "lang":""}
	try:
		st = time.time()
		""" read input request """ 
		training_data_path = request.form["TrainingDataPath"]
		json_data = json.load(open(training_data_path, "r"))
		sentences_list = json_data["TrainingData"] # training sentences

		dictionary_data_path = request.form["DictionaryDataPath"]
		dictionary_data = json.load(open(dictionary_data_path, "r"))

		number_of_clusters = int(request.form["NumberOfClusters"]) # number of clusters you want to cluster your data into
		lang = request.form["Language"] # en
		model_id = request.form["ModelId"] # any number of character
		algo = request.form["Algorithm"] # LDA / Kmeans
		embedding_model = request.form["EmbeddingModel"] # tfidf / w2v / BERT (*applicable only if algorithm is Kmeans)

		""" print details """
		print("\n training_data_path --- ", training_data_path)
		print("\n dictionary_data_path --- ", dictionary_data_path)
		print("\n number of clusters --- ",number_of_clusters)
		print("\n language --- ",lang)
		print("\n model_id --- ", model_id)
		print("\n embedding_model --- ",embedding_model)

		""" create model dir """
		model_dir = os.path.join(cf.MODEL_PATH, model_id)
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)

		""" preprocess text. Clean, lemmatize, tokenize data and find synonyms """
		cleaned_sentences, words_docs = preprocessing_obj.main(sentences_list, lang)

		""" train clustering model """
		decider_obj.main(cleaned_sentences, words_docs, lang, embedding_model, number_of_clusters, model_id, algo, model_dir, dictionary_data)
		res_json["model_id"] = model_id
		res_json["lang"] = lang
		res_json["status"] = "200"
		res_json["message"] = "Trained model successfully"
		print("\n total time --- ",time.time() - st)

	except Exception as e:
		print("\n Error in clustering app main() :", traceback.format_exc())
		res_json["status"] = "400"
		res_json["message"] = "Error occurred! Training failed! Please check inputs!!!"
	return jsonify(res_json)


@app.route("/train_word2vec", methods=['POST'])
@swag_from("swagger_files/word2vec_training_api.yml")
def w2v_training():
	res_json = {"status":"", "model_id":"", "lang":""}
	try:
		st = time.time()
		""" read input request """ 
		training_data_path = request.form["TrainingDataPath"]
		json_data = json.load(open(training_data_path, "r"))
		sentences_list = json_data["TrainingData"] # training sentences

		dictionary_data_path = request.form["DictionaryDataPath"]
		dictionary_data = json.load(open(dictionary_data_path, "r"))

		model_id = request.form["ModelId"] # any number of character
		lang = request.form["Language"] # en

		""" print details """
		print("\n training_data_path --- ", training_data_path)
		print("\n dictionary_data_path --- ", dictionary_data_path)
		print("\n model_id --- ", model_id)
		print("\n language --- ",lang)
		
		""" create model dir """
		model_dir = os.path.join(cf.MODEL_PATH, model_id)
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)

		""" preprocess text. Clean, lemmatize, tokenize data and find synonyms """
		cleaned_sentences, words_docs = preprocessing_obj.main(sentences_list, lang)
		
		""" train clustering model """
		train_w2v_obj.main(cleaned_sentences, words_docs, dictionary_data, model_id, model_dir)
		res_json["model_id"] = model_id
		res_json["lang"] = lang
		res_json["status"] = "200"
		res_json["message"] = "Trained model successfully"
		print("\n total time --- ",time.time() - st)

	except Exception as e:
		print("\n Error in clustering app main() :", traceback.format_exc())
		res_json["status"] = "400"
		res_json["message"] = "Error occurred! Training failed! Please check inputs!!!"
	return jsonify(res_json)

""" Prediction """
@app.route("/prediction", methods=['POST'])
@swag_from("swagger_files/clustering_prediction_api.yml")
def prediction():
	res_json = {"status":"", "model_id":""}
	try:
		st = time.time()
		""" read input request """ 
		cluster_name = int(request.form["ClusterName"]) # number of clusters you want to cluster your data into
		model_id = request.form["ModelId"] # any number of character
		print("\n number of clusters --- ",cluster_name)
		print("\n model_id --- ", model_id)

		model_dir = os.path.join(cf.MODEL_PATH, model_id)
		cluster_sentence_mapping_df = pd.read_csv(os.path.join(model_dir, "cluster_sentence_topic_mapping.csv"))
		cluster_rows = cluster_sentence_mapping_df[cluster_sentence_mapping_df.Topic == cluster_name]
		topic_sentences = list(cluster_rows.Text)
		topic_keywords = [topic.strip() for topic in cluster_rows.Keywords.iloc[0].split(",")]
		clusters_dict = {"cluster": cluster_name, "topics": topic_keywords, "number_of_sentences":len(topic_sentences) ,"sentences": topic_sentences}

		""" get sentences and topics of the given cluster """
		res_json["result"] = clusters_dict
		res_json["model_id"] = model_id
		res_json["status"] = "200"
		print("\n total prediction time --- ",time.time() - st)

	except Exception as e:
		print("\n Error in clustering app main() :",traceback.format_exc())
		res_json["status"] = "400"

	return jsonify(res_json)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=cf.port)
