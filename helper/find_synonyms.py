import json
import spacy
import traceback
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(WordnetAnnotator(nlp.lang))


class SynonymsGenerator:

	
	def __init__(self):
		pass


	def find_word_synonyms(self, word):
		synonyms_dict = {}
		try:
			token = nlp(word)[0]
			synsets = token._.wordnet.synsets()
			lemmas_for_synset = list(set([str(lemma) for s in synsets for lemma in s.lemma_names()]))
			token = str(token)
			if token in lemmas_for_synset: lemmas_for_synset.remove(token)
			synonyms_dict[token] = lemmas_for_synset
		except Exception as e:
			print("\n token : ", word, token)
			print("\n error in find_word_synonyms() : ", traceback.format_exc())
		return synonyms_dict


	def get_new_snentences(self, sent, tok, lemmas_for_synset):
		new_sents = []
		for syn in lemmas_for_synset:
			sent = sent.replace(tok.text, syn)
			new_sents.append(sent)
		return new_sents


	def find_sentence_synonyms(self, sentences):
		new_sentences = []
		for sent in sentences:
			sentence = nlp(sent)
			for tok in sentence:
				synsets = tok._.wordnet.synsets()
				lemmas_for_synset = list(set([lemma for s in synsets for lemma in s.lemma_names()]))
				new_cleaned_sentences.extend(self.get_new_snentences(sent, tok, lemmas_for_synset))
		print("\n number of sents before adding synonyms : ", len(sentences))
		print("\n sents after adding synonyms : ", len(new_sentences))
		return new_sentences


	def find_dictionary_synonyms(self, input_dict):
		new_input_dict = input_dict
		synonyms_dict = {}
		for key, value_list in input_dict.items():
			all_keywords = []
			all_keywords.extend(value_list)
			all_keywords.append(key)
			for word in all_keywords:
				token = nlp(word)[0]
				synsets = token._.wordnet.synsets()
				lemmas_for_synset = list(set([lemma for s in synsets for lemma in s.lemma_names()]))
				lemmas_for_synset = [syn for syn in lemmas_for_synset if syn != key and syn not in value_list]
				new_input_dict[key].extend(lemmas_for_synset)
		return new_input_dict


if __name__ == "_main__":
	obj = SynonymsGenerator()

	""" test """
	# word = "anxiety"
	# synonyms_dict = find_word_synonyms(word)

	# input_dict = {"good": ["excellent", "best"], "bad": ["worst", "pathetic"]}
	input_dict = json.load(open("file_path/abc.json"))
	new_input_dict = find_dictionary_synonyms(input_dict)
	print(new_input_dict)

	with open("file_path/abc_with_synonyms.json", "w+") as fs:
		fs.write(json.dumps(new_input_dict, indent=4))
