'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
TAMSIN ROGERS
CS 251 Data Analysis Visualization, Spring 2021
'''
import re
import os
import numpy as np


'''transforms an email into a list of words'''
def tokenize_words(text):
   
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

'''determine the count of each word in the entire dataset (across all emails)'''
def count_words(email_path='data/enron'):
    
    my_dict = {}
    f_path = os.listdir(email_path)
    count = 0
    
    newlist = []
    
    for i in f_path:
    	if (i != ".DS_Store"):
    		newlist.append(i)
    
    for i in newlist:
    	n_path = os.path.join(email_path, i)
    	emails = os.listdir(n_path)
    	count = count + len(emails)
    	
    	for j in emails:
    		line = os.path.join(n_path, j)
    		myfile = open(line)
    		txt = myfile.read()
    		words = tokenize_words(txt)
    		
    		for i in words:
    			if i in my_dict:
    				my_dict[i] = my_dict[i] + 1
    			else:
    				my_dict[i] = 1
    
    return my_dict, count


'''given the dictionary of the words that appear in the dataset and their respective counts,
	compile a list of the top `num_features` words and their respective counts'''
def find_top_words(word_freq, num_features=200):
    
    values = list(word_freq.values())
    keys = list(word_freq.keys())
    idx = np.argsort(values)
    fixed = idx[::-1]
    fin = fixed[0:num_features]
    top_words = []
    counts = []
    
    for i in range(num_features):
    	top_words.append(keys[fin[i]])
    	counts.append(values[fin[i]])
    return top_words, counts

'''count the occurance of the top W (`num_features`) words in each individual email, turn into a feature vector of counts'''
def make_feature_vectors(top_words, num_emails, email_path='data/enron'):
    
    f_path = os.listdir(email_path)
    
    feats = np.zeros((num_emails, len(top_words)))
    y = []
    count = 0
    
    newlist = []
    
    for i in f_path:
    	if (i != ".DS_Store"):
    		newlist.append(i)
    
    for i in range(len(newlist)):
    	n_path = os.path.join(email_path, newlist[i])
    	emails = os.listdir(n_path)
    	
    	for j in emails:
    		y.append(i)
    		line = os.path.join(n_path, j)
    		myfile = open(line)
    		txt = myfile.read()
    		words = tokenize_words(txt)
    		values = np.zeros(len(top_words))
    		
    		for w in words:
    			if w in top_words:
    				idx = top_words.index(w)
    				values[idx] += 1
    		feats[count] = values
    		
    		count += 1
    	
    y = np.array(y)
    return feats, y

'''divide up the dataset `features` into subsets ("splits") for training and testing. 
	The size of each split is determined by `test_prop`'''
def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
   
    inds = np.arange(y.size)
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]

    # Your code here:
    num = np.floor((y.size * (1 - test_prop))).astype('int')
    
    x_train = features[0:num, :]
    y_train = y[0:num]
    inds_train = inds[0:num]
    
    x_test = features[num:, :]
    y_test = y[num:]
    inds_test = inds[num:-1]
    
    return x_train, y_train, inds_train, x_test, y_test, inds_test

'''obtain the text of emails at the indices `inds` in the dataset'''
def retrieve_emails(inds, email_path='data/enron'):
   
    my_dict = {}
    f_path = os.listdir(email_path)
    count = 0
    
    newlist = []
    wordList = []
    
    for i in f_path:
    	if (i != ".DS_Store"):
    		newlist.append(i)
    
    for i in newlist:
    	n_path = os.path.join(email_path, i)
    	emails = os.listdir(n_path)
    	count = count + len(emails)
    	
    	for j in emails:
    		line = os.path.join(n_path, j)
    		myfile = open(line)
    		txt = myfile.read()
    		words = tokenize_words(txt)
    		wordList.append(words)
    	
    select = []
    
    for a in inds:
    	select.append(wordList[a])
    
    return select
    