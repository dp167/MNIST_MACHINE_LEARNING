#This program is created by David Poole for Project 2 of CS445 at Portland State University
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as mplot
import pandas
from sklearn.model_selection import train_test_split


class NN_Perceptron:

    def __init__(self,traincsv,testcsv):

        #import training data and read in to array experiement 1
        training_data_file = open(traincsv,'r')
        training_data = csv.reader(training_data_file)
        training_data_list = list(training_data)
        self.training_data_array = np.array(training_data_list)

        #import data and use sklearn train test split to create a half and quarter training data array for experiment 2
        #training_half= pandas.read_csv('mnist_train.csv')
        #train_half,train_quarter = train_test_split(training_half,train_size = 0.5,test_size =0.25)

        #choose half or quarter of training set with these lines here
        #self.training_data_array = np.array(train_half)
        #self.training_data_array = np.array(train_quarter)

        #print (self.training_data_array.shape)
        self.training_accuracy = []

        #import test data and read in to array
        test_data_file = open(testcsv,'r')
        test_data = csv.reader(test_data_file)
        test_data_list = list(test_data)
        self.test_data_array = np.array(test_data_list)
        self.test_accuracy = []

        #set bias to 1
        self.bias = 1
        #set alpha value to 0.9
        self.alpha = 0.9
        #make learning rate globaly available
        global learning_rate
        #make n_hidden_layers global
        global n_hidden_layers


        #establish an array of weights for input to hidden layers
        self.weights_input_to_hidden = np.random.uniform(-0.05,0.05,(785,n_hidden_layers))
        #establish an array of weights for hidden to output layer
        self.weights_hidden_to_output = np.random.uniform(-0.05, 0.05, (n_hidden_layers+1,10))
        #set hidden layer input array to a 1x#of hidden layer array
        self.hidden_layer_input = np.zeros((1,n_hidden_layers+1))
        #set value 0,0 to bias in array
        self.hidden_layer_input[0,0]= self.bias


#this function loops through the input data and applies the backpropogation algorithm when training
    def NN_forward(self,epoch,input_data,training_flag):

            #set up two arrays one for the list of prediction values and one for the list of actual values
            self.prediction_list = []
            self.actual_list = []

            #for loop that iterates through every input value along the vertical of the data set
            for i in range(input_data.shape[0]):
                    self.target_class = input_data[i,0].astype('int')#set target class to be the first value of the input row
                    self.actual_list.append(self.target_class)#append target class value to list of actual values
                    self.x_vector = input_data[i].astype('float16')/255#scale all the values in the row by 255 as a float16 value
                    self.x_vector[0]= self.bias#set the first value to the bias
                    self.x_vector = self.x_vector.reshape(1,785)#turn the column data in to a 1 by 785 dimension vector

                    # acquire the z value for the hidden layer by taking the dot product of the x vector and the weights from the input to hidden layer
                    hidden_layer_z = np.dot(self.x_vector,self.weights_input_to_hidden)
                    self.sigmoid_of_hidden_layer = self.sigmoid(hidden_layer_z) #take the sigmoid of the z value for the input to hidden layer
                    self.hidden_layer_input[0,1:] = self.sigmoid_of_hidden_layer#feed the input layer activation in to the hidden layer by assigning the sigmoid of the input to hidden layer to the hidden layer input

                    #acquire the z values of the output layer by taking the dot product of the hidden layer output and the hidden to output layer weights
                    output_layer_z = np.dot(self.hidden_layer_input,self.weights_hidden_to_output)#
                    self.sigmoid_of_output_layer = self.sigmoid(output_layer_z) #take the sigmoid of the output layer z value

                    #store the best prediction from the activation of the output layer z values with the argmax function
                    prediction = np.argmax(self.sigmoid_of_output_layer)
                    self.prediction_list.append(prediction)

                #if training run the backpropogation alforithm
                    if(epoch > 0  and training_flag ==1):
                            self.backprop(input_data,i)

            #compute confusion matrix from the scipy confusion matrix built in function with the actual list vs prediction list
            print(confusion_matrix(self.actual_list,self.prediction_list))
            accuracy = 0

            #use confusion matrix to calculate the acuracy from the diaginol of the matrix divided by number of training data points
            for p in range(10):
                    accuracy += confusion_matrix(self.actual_list,self.prediction_list)[p,p]

            accuracy = accuracy/confusion_matrix(self.actual_list,self.prediction_list).sum()
            accuracy = accuracy *100
            print(accuracy)
            return accuracy



    #this function is the implementation of the backpropogation algorithm
    def backprop(self,input_data,i):
            n_target_class = input_data[i, 0].astype('int')  # grab the truth label from the first column
            target_class_list = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # create a list of 10 classes
            target_class_list[n_target_class] = self.alpha  # set the nth class to one where n is the class 0 -9
            self.old_weights_input_to_hidden = np.zeros((785, n_hidden_layers))
            self.old_weights_hidden_to_output = np.zeros((n_hidden_layers + 1, 10))

            #calculate the output layer error from the formula delta_k < Ok(1-Ok)(tk-Ok)
            output_layer_error = self.sigmoid_of_output_layer * (1-self.sigmoid_of_output_layer)*(target_class_list-self.sigmoid_of_output_layer)
            #calculate the hidden layer error from the formula delta_n < hn(1-hn)(W_kn*delta_k)
            hidden_layer_error = self.sigmoid_of_hidden_layer*(1-self.sigmoid_of_hidden_layer)*np.dot(output_layer_error,self.weights_hidden_to_output[1:,:].T)

            #calculate deltas using old values of weights for hidden to output
            delta_of_weights_hidden_to_output = (learning_rate*output_layer_error* self.hidden_layer_input.T)+(self.alpha * self.old_weights_hidden_to_output)
            self.old_weights_hidden_to_output = delta_of_weights_hidden_to_output
            self.weights_hidden_to_output = self.weights_hidden_to_output + delta_of_weights_hidden_to_output

            #calculate deltas using old values for weight for input to hidden
            delta_of_weights_input_to_hidden = (learning_rate * hidden_layer_error * self.x_vector.T )+(self.alpha* self.old_weights_input_to_hidden)
            self.old_weights_input_to_hidden = delta_of_weights_input_to_hidden
            self.weights_input_to_hidden = self.weights_input_to_hidden + delta_of_weights_input_to_hidden


    #sigmoid function returns the input array with sigmoid activation using numpy to calculate the exponent
    def sigmoid(self,x):
            return 1/(1+np.exp(-x))

    #this function stores the accuracy data to a csv file
    def store_accuracy(self,index,accuracy,input_data):
        with open(input_data,'a',newline ='')as myfile:
                wr = csv.writer(myfile)
                wr.writerow([index,accuracy])
    #runs the forward pass for training and test set then stores the accuracy
    def run(self):
            for each in range(50):
                    train_accuracy = self.NN_forward(each,input_data = self.training_data_array,training_flag = 1)
                    test_accuracy = self.NN_forward(each, input_data = self.test_data_array,training_flag = 0)
                    self.store_accuracy(each,train_accuracy,'nntrain_output'+str(n_hidden_layers)+'.csv')
                    self.store_accuracy(each,test_accuracy,'nntest_output'+str(n_hidden_layers)+'.csv')

# inititalize hidden layer values and naming scheme for experiment 1
training_data = 'mnist_train.csv'
testing_data = 'mnist_test.csv'
learning_rate = 0.1
#n_hidden_layers = 20
#testing_output = 'nntest_output20.csv'
#training_output = 'nntrain_output20.csv'
#n_hidden_layers = 50
#testing_output = 'nntest_output50.csv'
#training_output = 'nntrain_output50.csv'
n_hidden_layers = 100
testing_output = 'nntest_output100.csv'
training_output = 'nntrain_output100.csv'
NN1 = NN_Perceptron(training_data,testing_data)

NN1.run()



#this code creates a plot of the accuracy data then saves it to a .png file

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
    mplot.savefig('plotrate1.png')

mplot.show()
