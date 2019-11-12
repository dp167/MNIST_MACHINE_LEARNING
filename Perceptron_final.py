#David Poole CS445 Project 1 this file has all the needed libraries and code to train and test thre learning rates on a single layer perceptron classifying digits from the mnist data set
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as mplot

class Perceptron:

    def __init__(self,traincsv,testcsv):
#import training data and read in to array
        training_data_file = open(traincsv,'r')
        training_data = csv.reader(training_data_file)
        training_data_list = list(training_data)
        self.training_data_array = np.array(training_data_list)
        self.training_accuracy = []

#import test data and read in to array
        test_data_file = open(testcsv,'r')
        test_data = csv.reader(test_data_file)
        test_data_list = list(test_data)
        self.test_data_array = np.array(test_data_list)
        self.test_accuracy = []

#set bias to 1
        self.bias = 1
#make learning rate available globaly
        global learning_rate
#create an 10x785 dimension array establishing weights with a normal distrobution between -0.05 and 0.05
        self.weights_array = np.random.uniform(-0.05,0.05,(10,785))


    def perceptron_learning_algorithm(self,epoch,input_data,training_flag):
 #intialize two lists one for the predicted classifications and another for the actual classification
        prediction_list =[]
        actual_list = []

#set up the 10 target classes as a list to be cycled through for testing
        for i in range(0,input_data.shape[0]):
            n_target_class = input_data[i,0].astype('int')# grab the truth label from the first column
            target_class_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#create a list of 10 classes
            target_class_list[n_target_class] = 1#set the nth class to one where n is the class 0 -9
            actual_list.append(n_target_class)

            activation_list = []
            output_list = []

 # create vector of x values from each row and scale each value by 255
            x_vector = input_data[i].astype('float16')/255
            x_vector[0] = self.bias # replace the first value(truth label) in the column with the bias
            x_vector = x_vector.reshape(1,785)#turn the column data in to a 1x785 dimension array
# for loop that does dot product of the weight vectors and the x value vector
            for j in range(10):
                activation = np.inner(x_vector,self.weights_array[j,:])
                if(activation <= 0):
                    prediction = 0
                else:
                    prediction = 1
#fill list of activation values and output values
                activation_list.append(activation)
                output_list.append(prediction)

#after for loop create array of pre activation values and append the highest value to a list of best predictions
            activation_array = np.array(activation_list)
            prediction_list.append(np.argmax(activation_array))
#after each epoch update the weights on training set only

            if epoch > 0 and training_flag == 1:
                for perceptron_i in range(10):
                    self.weights_array[perceptron_i,:] = self.weights_array[perceptron_i,:] + (learning_rate * (target_class_list[perceptron_i] - output_list[perceptron_i])*x_vector)

#compute accuracy from the comparison of the list of predictions and the list of actual values i calculate accuracy from summing the vertical on the confusion matrix and dividing by number of inputs
        print(confusion_matrix(actual_list,prediction_list))
        accuracy = 0
        for p in range(10):
          accuracy += confusion_matrix(actual_list,prediction_list)[p,p]

        accuracy = accuracy/confusion_matrix(actual_list,prediction_list).sum()
        accuracy = accuracy *100
        print(accuracy)
        return accuracy
#function to save the accuracy in to a csv file so it can be plotted
    def save_accuracy(self, accuracy_index, accuracy, input_data):
        with open(input_data, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow([accuracy_index, accuracy])

#iterate 50 epochs running through training testing and storeing the accurcy
    def run(self):


        for each in range(50):
            train_accuracy = self.perceptron_learning_algorithm(epoch=each,
                                                                      input_data=self.training_data_array,
                                                                      training_flag=1)
            self.training_accuracy.append(train_accuracy)

            testing_accuracy = self.perceptron_learning_algorithm(epoch=each + 1,
                                                                        input_data=self.test_data_array,
                                                                        training_flag=0)
            self.test_accuracy.append(testing_accuracy)

            self.save_accuracy(each, train_accuracy, 'training_output' + str(learning_rate) + '.csv')
            self.save_accuracy(each, testing_accuracy, 'testing_output' + str(learning_rate) + '.csv')

#initiate the variables for data files and the three diferent learning rate instaniations of class perceptron
training_data = 'mnist_train.csv'
testing_data = 'mnist_test.csv'
perceptron01 = Perceptron(training_data, testing_data)
perceptron001 = Perceptron(training_data, testing_data)
perceptron0001 = Perceptron(training_data, testing_data)

#run with all three learning rates outputting the graphs for accuracy after each 50 epoch cycle
for iterate_learning_rates in range(3):
    if(iterate_learning_rates == 0):
        learning_rate = 0.1
        training_output = 'training_output0.1.csv'
        testing_output = 'testing_output0.1.csv'
        perceptron01.run()
    elif(iterate_learning_rates == 1):
        learning_rate = 0.01
        training_output = 'training_output0.01.csv'
        testing_output = 'testing_output0.01.csv'
        perceptron001.run()
    else:
        learning_rate = 0.001
        training_output = 'training_output0.001.csv'
        testing_output = 'testing_output0.001.csv'
        perceptron0001.run()

#this code creates a plot of the accuracy data then saves it to a .png file
    x1, y1 = np.loadtxt(training_output, delimiter=',', unpack=True)
    x2, y2 = np.loadtxt(testing_output, delimiter=',', unpack=True)
    mplot.plot(x1, y1, label="Training Set")
    mplot.plot(x2, y2, label="Testing Set")
    mplot.xlabel('Epochs')
    mplot.ylabel('Accuracy (%) ')
    mplot.legend()
    if(iterate_learning_rates==0):
        mplot.title('For Learning rate 0.1')
        mplot.savefig('plotrate1.png')
    if(iterate_learning_rates==1):
        mplot.title('For Learning rate 0.01')
        mplot.savefig('plotrate01.png')
    if(iterate_learning_rates==2):
        mplot.title('For Learning rate 0.001')
        mplot.savefig('plotrate001.png')
    mplot.show()



