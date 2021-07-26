# Clustering text documents using advanced NLP embeddings 

Cluster your documents into different clusters/groups.
It uses 
1. TF-IDF
2. google's state of the art (SOTA) Word2vec model


setup the project environment
1. Create virtual environment using the command : python3.7 -m venv clustering_docs_env_3.7
2. Activate the virtual environment using : source clustering_docs_env_3.7/bin/activate
3. Run the above command to install setup tools : python3 -m pip install setuptools pip install -U wheel
4. Install all the required python packages using : python3 -m pip install -r requirements.txt
	download spacy model: python3 -m spacy download en_core_web_sm
5. Run the flask API : python3 clustering_api.py
6. In browser run: http://0.0.0.0:5001/apidocs
7. enter the absolute path of your dataset json file in the swagger. And provide necessary inputs 

Download google's pre-trained word2vec model from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

NOTE: Please change the path of the word2vec in configuration file before running the service.

References:
https://github.com/hanxiao/bert-as-service
https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
https://blog.eduonix.com/artificial-intelligence/clustering-similar-sentences-together-using-machine-learning/
https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
https://stackoverflow.com/questions/59465247/type-error-while-finding-dominant-topics-in-each-sentence-in-gensim
https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
https://stackoverflow.com/questions/51426107/how-to-build-a-gensim-dictionary-that-includes-bigrams/54344757
https://github.com/swapnilg915/Topic_Modeling_using_LDA/blob/master/topic_modeling_using_LDA.py
https://scikit-learn.org/

