	# def sent_vectorizer(self, sent):
	# 	sent_vec = np.zeros((1,300))
	# 	numw = 0
	# 	try:
	# 		for w in sent.split():
	# 			try:
	# 				if numw == 0:
	# 					sent_vec = self.model[w]
	# 				else:
	# 					sent_vec = np.add(sent_vec, self.model[w])
	# 				numw+=1
	# 			except:
	# 				self.oov_list.append(w)
	# 				pass
	# 		sentence_vector = np.asarray(sent_vec) / numw
	# 	except Exception as e:
	# 		print(sent_vec, numw)
	# 		print("\n Error in sent_vectorizer : ",e)
	# 		print("\n Error details : ", traceback.format_exc())
	# 	return sentence_vector


	def sent_vectorizer(self, sent):
		# sent_vec = np.zeros((1,300))
		sent_vec=[]
		numw = 0
		try:
			for w in sent.split():
				try:
					sent_vec.append(self.model[w])
				except:
					print()
					self.oov_list.append(w)
					pass
			# sentence_vector = np.asarray(sent_vec) / numw
			sentence_vector = np.asarray(sent_vec)
		except Exception as e:
			print(sent_vec, numw)
			print("\n Error in sent_vectorizer : ",e)
			print("\n Error details : ", traceback.format_exc())
		return sentence_vector


	def create_w2v_vectors(self, sentences):
		print("\n sentences for vectorization : ", len(sentences))
		self.load_word2vec()
		st = time.time()
		X = [self.sent_vectorizer(sent) for sent in sentences ]
		X = np.array(X)
		print("\n shape --- ", X.shape)
		print("\n time for w2v vectorization : ", time.time() - st)
		del self.model
		return X
