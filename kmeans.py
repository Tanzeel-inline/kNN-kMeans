# %%
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def intra_euclidean_distance(x1, x2):
		return np.sum(((x1 - x2) ** 2))
class Kmean:
	def __init__(self, k, data):
		if k > len(data):
			print(f"Invalid number of clusters!!!")
			exit(-1)
		#Divinding total data by k, will give us chunks
		#We will pick a random value as centeroid in each chunk for each k
		self.X = data
		self.k = k
		#Divinding data into chunks, so we can pick random centeroid from each chunk
		data_division = math.floor(len(self.X) / self.k)
		m, n = self.X.shape
		self.initial_centeroids = np.zeros((self.k, n))
		np.shape(self.initial_centeroids)
		#Picking a random data from each chunk
		for i in range(self.k):
			if i == self.k - 1:
				self.initial_centeroids[i] = self.X[random.randint(i * data_division , len(self.X) - 1)]
				break
			self.initial_centeroids[i] = self.X[random.randint(i * data_division , (i + 1) * data_division)] 
	def individual_euclidean_distance(self, x1, x2):
		return np.sqrt(np.sum(((x1 - x2) ** 2)))
	def find_closest_centroids(self, centroids): 
		m = self.X.shape[0] 
		k = centroids.shape[0] 
		idx = np.zeros(m) # array to assign the centriod
		
		i = 0
		for x in self.X:
			current_score = math.inf
			current_centroid = centroids[0]
			j = 0
			for y in range(len(centroids)):
				calculate_euclidean_score = self.individual_euclidean_distance(centroids[y], x)
				if calculate_euclidean_score < current_score:
					current_score = calculate_euclidean_score
					current_centroid = centroids[y]
					j = y
			idx[i] = j
			i += 1
		return idx
	
	def compute_centroids(self, idx):
		m, n = self.X.shape
		centroids = np.zeros((self.k, n)) 
		
		centroid_average = np.zeros((self.k, n))
		centroid_count = np.zeros(self.k)
		#For every index
		for i in range(len(idx)):
			#For every index, it have n variables
			for x in range(n):
				centroid_average[int(idx[i])][x] += self.X[i][x]
			centroid_count[int(idx[i])] += 1
		
		for i in range(self.k):
			for x in range(n):
				centroids[i][x] = centroid_average[i][x] / centroid_count[i]
		return centroids    

	def run_k_means(self, max_iters):
		m, n = self.X.shape
		idx = np.zeros(m)
		centroids = self.initial_centeroids
		for i in range(max_iters):
			
			# find closest centroid to each vector
			idx = self.find_closest_centroids(centroids)
			#Save previous centroids
			self.initial_centeroids = centroids
			#Update centroids
			centroids = self.compute_centroids(idx)
			#Check if same centroids
			if ( (self.initial_centeroids == centroids).all() ):
				print(f"Iteration : {i}, centeroid matched, breaking out of the loop")
				break
		return idx, centroids
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
#K-MEAN CLUSTER
m, n = X.shape
for i in range(3,10):
	k = i
	print(f"K is {k}")
	kmean = Kmean(k, X)
	idx, centroids = kmean.run_k_means(100)
	idx = idx.astype(int)

	"""-----------------------------------------------------------"""
	#Davies-Bouldin Index
	print("Davien Bouldin Index : ")
	#Step 1: Calculate intra-cluster dispersion
	print("\nStep-1 :")
	intra_cluster_dispersion = np.zeros((len(centroids)))
	total_vectors_in_each_cluster = np.zeros((len(centroids)))
	#For every vector, calclaute its distance from its centroid
	for j in range (len(idx)):
		intra_cluster_dispersion[idx] += intra_euclidean_distance(X[j], centroids[idx])
		total_vectors_in_each_cluster[idx] += 1

	intra_cluster_dispersion = np.sqrt(intra_cluster_dispersion / total_vectors_in_each_cluster)
	print(f"\n\nIntra cluster dispersion is : {(intra_cluster_dispersion)}")


	#Step 2: Calculate separation measure
	print("\nStep-2 :")
	seperation_measures = np.zeros((len(centroids), len(centroids)))
	for i in range(len(centroids)):
		for j in range(len(centroids)):
			if j >= i:
				break
			seperation_measures[i][j] = np.sqrt(intra_euclidean_distance(centroids[i], centroids[j]))
			seperation_measures[j][i] = seperation_measures[i][j]
	print("Seperation Measure : ")
	for i in range(len(centroids)):
		print(f"{(seperation_measures[i])}")
	#Step 3: Calculate similarity between clusters
	print("\nStep-3 :")
	similarity_measures = np.zeros((len(centroids), len(centroids)))
	for i in range(len(centroids)):
		for j in range(len(centroids)):
			if j >= i:
				break
			similarity_measures[i][j] = ( intra_cluster_dispersion[i] + intra_cluster_dispersion[j] ) / seperation_measures[i][j]
			similarity_measures[j][i] = similarity_measures[i][j]
	print("Similarity Measure : ")
	for i in range(len(centroids)):
		print(f"{(similarity_measures[i])}")
	#Step 4: Find most similar cluster for each cluster i
	print("\nStep-4 :")
	most_similar_cluster = np.zeros((len(centroids)))
	for i in range(len(centroids)):
		for j in range(len(centroids)):
			if i == j:
				continue
			if similarity_measures[i][j] > most_similar_cluster[i]:
				most_similar_cluster[i] = similarity_measures[i][j]
	print(f"Most similar clusters are : {(most_similar_cluster)}")

	#Step 5: Calculate the Davies-Bouldin Index
	print("\nStep-5 :")
	davies_bouldin_index = np.average(most_similar_cluster)

	print(f"Resulting DAVIES-BOULDIN INDEX is : {(davies_bouldin_index):.3f} %\n\n")
	"""--------------------------------------------------------------------"""

	#matlibplot
	k_Meancluster = []
	for i in range(k):
		k_Meancluster.append(X[np.where(idx == i)[0],:])
	#k_Meancluster.append(X[np.where(idx == 1)[0],:])
	#k_Meancluster = X[np.where(idx == 2)[0],:]

	#fig, ax = plt.subplots(figsize=(16,12))
	#ax.scatter.jitter(0.5, 0.5)
	plt.title("Visualizer")
	plt.xlabel("Distance difference")
	plt.ylabel("Weight difference")

	colors = cm.rainbow(np.linspace(0, 1, k))
	for i in range(k):
		plt.scatter( (k_Meancluster[i][:,0] - k_Meancluster[i][:,2]), ( k_Meancluster[i][:,1] - k_Meancluster[i][:,3] ), s=20, color=colors[i], label='Cluster' + str(i), alpha=1/5, marker=i)
	#plt.scatter( (k_Meancluster1[:,0] - k_Meancluster1[:,2]), ( k_Meancluster1[:,1] - k_Meancluster1[:,3] ), s=20, color='r', label='Cluster 1', alpha=1/5, marker=4)
	#plt.scatter((k_Meancluster2[:,0] - k_Meancluster2[:,2]), ( k_Meancluster2[:,1] - k_Meancluster2[:,3] ), s=30, color='g', label='Cluster 2', alpha=1/5, marker=5)
	#plt.scatter((k_Meancluster3[:,0] - k_Meancluster3[:,2]), ( k_Meancluster3[:,1] - k_Meancluster3[:,3] ), s=30, color='b', label='Cluster 3', alpha=1/5, marker=6)
	#ax.legend()
	plt.show()
# %%
