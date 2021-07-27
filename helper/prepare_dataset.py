# import json
# from sklearn.datasets import fetch_20newsgroups
# dataset = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
# sentences_lst= dataset.data
# labels=list([int(num)for num in dataset.target])
# num_clusters = len(dataset.target_names)

# required_json = {"TrainingData":sentences_lst, "labels":labels,"NumberOfClusters":num_clusters, "Language":"en", "BotId":"bot_1"}
# with open("datasets/clustering_20newsgroup_training_data.json", "w+") as fs:
# 	fs.write(json.dumps(required_json, indent=4))
# 	print("\n created json successfully!")

import os
import json
import pandas as pd
import configuration_file as config

source_path = os.path.join(config.BASE_PATH, "datasets","OCD2.csv")
destination_path = os.path.join(config.BASE_PATH, "datasets", "OCD2.json")

df = pd.read_csv(source_path)
text_key = "Description"
sentences_lst= list(df[text_key])
required_json = {"TrainingData":sentences_lst}
with open(destination_path, "w+") as fs:
	fs.write(json.dumps(required_json, indent=4))
	print("\n created json successfully!")