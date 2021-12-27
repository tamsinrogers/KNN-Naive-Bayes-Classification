'''knn.py
K-Nearest Neighbors algorithm for classification
TAMSIN ROGERS
CS 251 Data Analysis Visualization, Spring 2021
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from palettable import cartocolors
from scipy import stats

'''K-Nearest Neighbors supervised learning algorithm'''
class KNN:
    
    '''KNN constructor'''
    def __init__(self, num_classes):
        
        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None
        self.num_classes = num_classes

    '''trains the KNN classifier on the data `data`, where training samples have corresponding class labels in `y`'''
    def train(self, data, y):
    
        self.exemplars = data
        self.classes = y

    '''uses the trained KNN classifier to predict the class label of each test sample in `data`'''
    def predict(self, data, k):
       
        classes = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            xval = np.subtract(self.exemplars[:, 0], data[i,0])**2
            yval = np.subtract(self.exemplars[:, 1], data[i,1])**2
            dist = np.power(np.add(xval, yval), .5)
            
            hold = np.argsort(dist)
            y = self.classes[hold[0:k]]
            classes[i] = stats.mode(y)[0]
            
        return classes
    
    '''computes the accuracy based on percent correct: Proportion of predicted class labels `y_pred` that match the true values `y`'''
    def accuracy(self, y, y_pred):
       
        equal = np.sum(y == y_pred)
        accuracy = equal/len(y_pred)
        
        return accuracy

    '''paints the data space in colors corresponding to which class the classifier would hypothetically assign to data samples appearing in each region'''
    def plot_predictions(self, k, n_sample_pts):
        
        colors = cartocolors.qualitative.Safe_4.mpl_colors
        L1 = ListedColormap(colors)
        line = np.linspace(-40, 40, n_sample_pts)
        xsamp, ysamp = np.meshgrid(line, line)
        x = xsamp.flatten()
        y = ysamp.flatten()
        pairs = np.column_stack((x,y))
        pairs = pairs.reshape(n_sample_pts * n_sample_pts, 2)
        prediction = self.predict(pairs, k)
        prediction = prediction.reshape(n_sample_pts, n_sample_pts)
        plt.pcolormesh(xsamp, ysamp,prediction, cmap=L1, shading='auto')
        plt.colorbar()
        
    '''creates a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`)'''
    def confusion_matrix(self, y, y_pred):
        
        falsepos = 0
        falseneg = 0
        truepos = 0
        trueneg = 0
        
        count = 0
        
        falseposi = []
        falsenegi = []
        trueposi = []
        truenegi = []

        for y, y_pred in zip(y, y_pred):
            count += 1
            if y_pred == y: 
                if y_pred == 1: 
                    truepos += 1
                    trueposi.append(count)
                else: 
                    trueneg += 1
                    truenegi.append(count)
            else: 
                if y_pred == 1:
                    falsepos += 1
                    falseposi.append(count)
                else: 
                    falseneg += 1
                    falsenegi.append(count)
            
        cm = [[trueneg, falsepos], [falseneg, truepos]]
        cm = np.array(cm)
        return cm, falseposi, falsenegi, trueposi, truenegi
        