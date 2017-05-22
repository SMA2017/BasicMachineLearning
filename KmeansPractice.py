import matplotlib.pyplot as plt
import numpy as np
import random


class Kmean:	
	def __init__(self, tol = 0.0001, max_iter= 300):
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data, k = 2):
		self.centroids = {}

		colors = ['r', 'g', 'b']
		
		print(data.shape)
		#initialization
		for i in range(k):
			self.centroids[i] = data[random.randint(0, data.shape[0]-1)]
		

		for ite in range(self.max_iter):
			self.groupdata = {}
			#initialization group data
			for i in range(k):
				self.groupdata[i] = []

			#run over data size
			for feature in data:
				distances = [np.linalg.norm(feature - self.centroids[centroid]) for centroid in self.centroids] #broadcast
				
				index = distances.index(np.min(distances))
				self.groupdata[index].append(feature)

			if ite == 0:
				displayData(colors, self.groupdata)
				displayCentroidData(colors, self.centroids)
				# plt.show()
				plt.savefig('initialization.png')
				plt.close()

			prev_centroids = dict(self.centroids)
			optimized = True

			for index in self.centroids:
				self.centroids[index] = np.average(self.groupdata[index], axis = 0)

			for i in self.centroids:
				original_centroid = prev_centroids[i]
				current_centroid = self.centroids[i]
				if(np.sum((current_centroid - original_centroid)/original_centroid*1000.0) > self.tol):
					optimized = False

			print('iteration_%d.png', ite)
			plt.cla()
			displayData(colors, self.groupdata)
			displayCentroidData(colors, prev_centroids)
			plt.show()
			#plt.savefig('iteration_{}.png'.format(ite))
			# plt.pause(1.0)
			plt.close()
			

			if optimized:
				break			

	def predict(self, data):
		distances = [np.linalg.norm(data - self.centroids[centroid] for centroid in self.centroids)]
		classification = distances.index(min(distances))
		return classifcation


def displayCentroidData(colors, centroids):

	for centroid in centroids:
		plt.scatter(centroids[centroid][0], centroids[centroid][1], marker = '+', color = 'k', s = 150)
	plt.hold(True)

def displayData(colors, groupdata):
	for groupIndx in groupdata.iterkeys():
		color = colors[groupIndx]
		for datapoint in groupdata[groupIndx]:
			plt.scatter(datapoint[0], datapoint[1], marker = 'o', color = color, s = 150)
	plt.hold(True)	

def test_Kmeans():
	X = np.array([[1, 2],
		[1.5, 1.8],
		[3.0, 4.0],
		[2.0, 1.1],
		[2.3, 4.0],
		[5.0, 6.0],
		[5.4, 6.2],
		[4.2, 7.8],
		[3.5, 4.9],
		[4.0, 5.2],
		[1.5, 1.2],
		[2.0, 1.3]])

	mean1 = [1.0, 1.5]
	cov1 = [[1, 0], [0, 1]]
	X1 = np.random.multivariate_normal(mean1, cov1, 100)
	
	mean2 = [5.5, 4.3]
	cov2 = [[1, 0], [0, 1]]
	X2 = np.random.multivariate_normal(mean2, cov2, 100)

	mixture_gaussian_data =  np.concatenate((X1, X2), axis=0)

	colors = ['r', 'g', 'b']

	kmean = Kmean()

	kmean.fit(mixture_gaussian_data)

	for centroidIndx in kmean.centroids:
		print(kmean.centroids[centroidIndx])


#call function
test_Kmeans() 