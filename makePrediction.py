# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import random
import os
from sklearn.metrics import accuracy_score
import pickle
import sys

class FactOrFeelModel(object):
	log_model = LogisticRegression()
	vectorizer = CountVectorizer()

	def __init__(self):
		# load the model from disk
		filename = 'finalized_model.sav'
		if (sys.version_info > (3, 0)): # if python3
			with open(filename,'rb') as f:
				self.log_model = pickle.load(f, encoding='latin1')
		else:
			self.log_model = pickle.load(open(filename, 'rb'))


		#load the vectorizer from the disk
		filename2 = 'vectorizer.sav'
		if (sys.version_info > (3, 0)):	# if python 3
			with open(filename2,'rb') as f:
				self.vectorizer = pickle.load(f, encoding='latin1')
		else:
			self.vectorizer = pickle.load(open(filename2, 'rb'))

	def example(self):
		text1 = "You should be proud of yourself"
		text2 = "The lab coat is white"
		data = [text1,text2]
		print_results(data)

	# data can be of the form string or [string]
	# returns ['fact'] or ['feel']
	def make_prediction(self,data):
		prediction = ''
		if type(data) == str:
			data = [data]
			return self.log_model.predict(self.vectorizer.transform(data).toarray())
		elif type(data) == list:
			return self.log_model.predict(self.vectorizer.transform(data).toarray())
		else:
			raise ValueError("data must be either list of strings or a string but is of type " + str(type(data)))

	# text is a string
	# num_sentences_per_eval is the number of sentences for each prediction (NOT CURRENT IN USE)
	# returns the percent of feel and fact
	def evaluateText(self,text):
		factCounter = 0
		feelCounter = 0

		model = FactOrFeelModel()
		splitText = text.split('.')
		splitText.pop()
		# splitText = [x+y for x,y in zip(splitText[0::2], splitText[1::2])] #each prediciton is two sentences

		preds = model.make_prediction(splitText)

		for pred in preds:
			if type(pred) != str:
				pred = pred.decode("utf-8")
			if pred == 'fact':
				factCounter+=1
			else:
				feelCounter+=1

		percentFacts = int(float(factCounter)/float(factCounter+feelCounter) * 100)
		percentFeels = int(float(feelCounter)/float(feelCounter+factCounter) * 100)

		return [percentFacts,percentFeels]

	def printEvaluations(self, percentages):
		print("facts: " + str(percentages[0]) + "% | feels: " + str(percentages[1]) + "% | Predictions accuracy: 73%")


if __name__ == "__main__":
	model = FactOrFeelModel()
	path = '/Users/andrei/fact_vs_feel/texts/CNNTrumpArticle.txt'
	with open(path, 'r') as content_file:
		content = content_file.read()
		percentages = model.evaluateText(content)
		model.printEvaluations(percentages)

	# while(True):
	# 	data = raw_input("Enter a sentance. (type 'q' to quit)\n")
	# 	if data == "q":
	# 		break
	# 	print(model.make_prediction(data))


