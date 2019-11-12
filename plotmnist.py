import csv
from sklearn.metrics import confusion_matrix
from scipy.special import expit
import matplotlib.pyplot as mplot
import numpy as np
import math
training_data = 'mnist_train.csv'
testing_data = 'mnist_test.csv'
learning_rate = 0.1
#n_hidden_layers = 20
#training_output = 'nntest_output20.csv'
#testing_output = 'nntrain_output20.csv'
#n_hidden_layers = 50
#training_output = 'nntest_output50.csv'
#testing_output = 'nntrain_output50.csv'
n_hidden_layers = 100
training_output = 'nntest_output100.csv'
testing_output = 'nntrain_output100.csv'

x1, y1 = np.loadtxt(training_output, delimiter=',', unpack=True)
x2, y2 = np.loadtxt(testing_output, delimiter=',', unpack=True)
mplot.plot(x1, y1, label="Training Set")
mplot.plot(x2, y2, label="Testing Set")
mplot.xlabel('Epochs')
mplot.ylabel('Accuracy (%) ')
mplot.legend()
if(n_hidden_layers==20):
    mplot.title('For 20 hidden layers')
    mplot.savefig('plotrate1.png')
if(n_hidden_layers == 50):
    mplot.title('For 50 hidden layers')
    mplot.savefig('plotrate1.png')
if(n_hidden_layers == 100):
    mplot.title('For 100 hidden layers')
    mplot.savefig('plotrateE2half.png')

mplot.show()