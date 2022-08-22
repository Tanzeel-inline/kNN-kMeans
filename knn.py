# %%
import pandas as pd
import numpy as np
import math
import collections
class KNN:
	def __init__(self, train_data, train_data_classes, k):
		self.train_data = train_data
		self.train_data_classes = train_data_classes
		self.k = k
	def individual_euclidean_distance(self, x1, x2):
		return np.sqrt(np.sum(((x1 - x2) ** 2)))
	def euclidean_distance(self, x):
		return [self.individual_euclidean_distance(x,x1) for x1 in self.train_data]
	#Calculate the distance from each point in train data
	#sort on the basis of index
	#get first k sorted index
	#find the most common index 
	#that is class of test data
	def classify_data(self, x):
		e_distance = self.euclidean_distance(x)
		nearest_neighbours = np.argsort(e_distance)
		first_k_classes = [int(self.train_data_classes[nearest_neighbours[i]]) for i in range(self.k)]
		most_nearest = collections.Counter(first_k_classes).most_common(1)
		return most_nearest[0][0]
		#This was returning wrong result
		#first_k_classes = np.array(first_k_classes)
		#most_nearest_class = np.bincount(first_k_classes).argmax()
		#return most_nearest_class
	#Calls classify data for each individual row
	def classification(self, X):
		classify = [self.classify_data(x) for x in X]
		return np.array(classify)
	

"""--------------------------------------------------------------------"""	
#Read data from the file into panda dataframe
data = pd.read_csv('balance-scale.data')

#Change these attributes if you want to change the data file
#---------------------------------------------------------------
class_column_index = 0
#data = data.drop('Sex',1)
classes_name = data.Class.unique()
total_classes = len(classes_name)
classes_classification = dict()
classes_backward_classification = dict()
j = 0
for i in range(total_classes):
	classes_classification[classes_name[i]] = j
	classes_backward_classification[j] = classes_name[i]
	j += 1
#Create the dictionary to convert classes to numeric for easeness
#classes_classification = {'L':0, 'B':1, 'R':2}
#classes_backward_classification = {0:'L', 1:'B', 2:'R'}
#---------------------------------------------------------------
#Change the classes value in dataframe
data.Class = [classes_classification[index] for index in data.Class]

#Converting the dataframe into numpy array
X = data.to_numpy()
#Shuffle the dataset
np.random.shuffle(X)
#Save the class column into numpy array
y = X[:,class_column_index]
#Delete the class column, 0 = index, 1 = axis
X = np.delete(X, class_column_index, 1)

"""-----------------------------------------------------------"""
#Data Split Here
#Spliting the dataset 75% for training / other for testing
train_dataset_length = math.floor((len(X) / 100 ) * 75)
test_dataset_length = len(X) - train_dataset_length

train_X = X[0:train_dataset_length,:]
train_y = y[0:train_dataset_length]

test_X = X[train_dataset_length : len(X),:]
test_y = y[train_dataset_length : len(X)]

print(f"Train dataset length is {len(train_X)}")
print(f"Test dataset length is {len(test_y)}")
"""-----------------------------------------------------------"""

"""--------------------------------------------------------------------"""
#Implemented KNN
for i in range(4,10):
	kNN = KNN(train_X,train_y,i)
	predicted = kNN.classification(test_X)
	acc = np.sum(predicted == test_y) / len(test_y)
	print("------------------------------------------------")
	print(f"Implemented KNN Overall-Accuracy : {(acc * 100):.3f} %")
	print("------------------------------------------------")
	#Making confusion matrix

	"""
								Actual
						Left	Balanced	Right
				Left
	Predicted	Balanced
				Right
	"""
	#Since we have three possible classes
	confusion_matrix = np.zeros((total_classes,total_classes),dtype=int)
	#Update the confusion matrix
	for i in range(len(test_y)):
		confusion_matrix[int(predicted[i])][int(test_y[i])] += 1

	print("Confusion Matrix : ")
	for i in range(total_classes):
		print(confusion_matrix[i])
	#Now that we have confusion matrix, we can calculate the precision and recall for class
	#Left class precision
	macro_f1_score = 0
	for i in range(total_classes):
		#Calculating the stats for each class
		total_predicted = 0
		actutal_data = 0
		correctly_predicted = confusion_matrix[i][i]
		for j in range(total_classes):
			total_predicted += confusion_matrix[i][j]
			actutal_data += confusion_matrix[j][i]
		precision = 0
		if ( total_predicted != 0 ):
			precision = (correctly_predicted / total_predicted ) * 100
		recall = 0
		if ( actutal_data != 0 ):
			recall = (correctly_predicted / actutal_data ) * 100
		f1_score = 0
		if precision != 0 or recall != 0:
			f1_score = 2*((precision*recall)/(precision+recall))
		#Printing the stats for each class
		print(f"Precision for class {classes_backward_classification[i]} is : {precision:.3f} %")
		print(f"Re-call for class {classes_backward_classification[i]} is : {recall:.3f} %")
		print(f"F1-score for class {classes_backward_classification[i]} is : {f1_score:.3f} %")
		print("------------------------------------------------")
		macro_f1_score += (f1_score / 100)

	print(f"Macro-F1 score is : {(macro_f1_score / total_classes) * 100:.3f} %")

"""--------------------------------------------------------------------"""
# %%
