'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
TAMSIN ROGERS
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
from sklearn import metrics


class NaiveBayes:
	'''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
	 number of classes)'''
	def __init__(self, num_classes):
		'''Naive Bayes constructor

		TODO:
		- Add instance variable for `num_classes`
		'''
		# class_priors: ndarray. shape=(num_classes,).
		#	Probability that a training example belongs to each of the classes
		#	For spam filter: prob training example is spam or ham
		self.class_priors = None
		# class_likelihoods: ndarray. shape=(num_classes, num_features).
		#	Probability that each word appears within class c
		self.class_likelihoods = None
		self.num_classes = num_classes

	'''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
		class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
		class likelihoods (the probability of a word appearing in each class — spam or ham)'''
	def train(self, data, y):
	
		num_samps, num_features = data.shape
		unique, frequency = np.unique(y, return_counts=True)
		self.class_priors = frequency / num_samps
		cdat = []
		
		for i in range(len(unique)):
			cidx = np.where(y == unique[i])[0]
			cdat.append(data[cidx, :])
			
		likelihoods = []
		
		for i in cdat:
			counts = np.sum(i, axis=0)
			likely = (counts+1) / (np.sum(i) + num_features)
			likelihoods.append(likely)
		
		self.class_likelihoods = np.array(likelihoods)


	'''combine the class likelihoods and priors to compute the posterior distribution so the predicted class 
		for a test sample from `data` is the class that yields the highest posterior probability'''
	def predict(self, data):
		
		num_test_samps, num_features = data.shape
		
		classes = []
		
		for i in range(num_test_samps):
			posteriors = np.log(self.class_priors) + np.log(self.class_likelihoods) @ data[i].T
			classes.append(np.argmax(posteriors))	# pick out the biggest posteriors
		
		return np.array(classes)

	'''computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
		that match the true values `y`'''
	def accuracy(self, y, y_pred):
		
		a = np.sum(y == y_pred) / y.shape[0]
		return a

	'''creates a confusion matrix based on the ground truth class labels (`y`) and those predicted
		by the classifier (`y_pred`)'''
	def confusion_matrix(self, y, y_pred):
		
		falsepos = 0
		falseneg = 0
		truepos = 0
		trueneg = 0
		
		count = 0
		
		# for keeping track of the indices
		falseposi = []
		falsenegi = []
		trueposi = []
		truenegi = []

		for y, y_pred in zip(y, y_pred):
			count += 1
			
			if y_pred == y: 				# check for matches (true)
				if y_pred == 1: 
					truepos += 1
					trueposi.append(count)
				else: 
					trueneg += 1
					truenegi.append(count)
					
			else: 							# check for nonmatches (false)
				if y_pred == 1:
					falsepos += 1
					falseposi.append(count)
				else: 
					falseneg += 1
					falsenegi.append(count)
			
		cm = [[trueneg, falsepos], [falseneg, truepos]]
		cm = np.array(cm)
		return cm, falseposi, falsenegi, trueposi, truenegi