import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invwishart, multivariate_normal
import pdb

class GibbsSamplingGMM:

	def __init__(self, iteration):
		self.iteration = iteration
		# self.k = numberOfModels

	def initialization(self,N, K, D):
		self.N = N
		self.K = K
		self.D = D

		self.Z = np.zeros((N, K))  # N elements
		self.pi = {} # k elements
		self.mu = {} # k elements
		self.sigma = {} #k elements

		self.alpha = tuple(10 for i in range(K))

		self.mu0 = np.ones((D,))
		self.sigma0 = np.eye(D)
		for i in range(K):
			self.mu[i] = np.random.multivariate_normal(self.mu0, self.sigma0)
			self.sigma[i] = invwishart.rvs(df = D, scale = np.eye(D), size = 1)
			
		self.pi[0] = np.random.dirichlet(self.alpha, 1)#should be alpha1, alpha2, alpha3,..., alphaK
		for i in range(N):
			self.Z[i]  = np.random.multinomial(1, sum(self.pi[0]), size=1) #sum(self.pi[i]) to convert np.array of shape (1,K) to array of length K elements 

	def sampling(self, N, K, D, data):
		self.initialization(N, K, D) #4 : number of data points   #2 : number of mixture models #4: dimension vectors

		for i in range(self.iteration):
			if i%500 == 0:
				print(i)
				print(self.mu[0])
				print(self.mu[1])
				print(self.sigma[0])
				
			self.samplingPI()
			self.samplingZ(data)
			self.samplingMu(data)
			self.samplingCovariance(data)

	def samplingPI(self):
		
		new_alpha = list(self.alpha)

		for i in range(self.K):
			Ni = np.sum(self.Z[:, i] == 1)
			new_alpha[i] += Ni

		tuple_alpha = tuple(new_alpha)

		self.pi[0] = np.random.dirichlet(tuple_alpha, 1)
		# return 0

	def samplingMu(self, data):

		for i in range(self.K):
			Nk = np.sum(self.Z[:, i] == 1) 
			same_group_k = data[self.Z[:, i] == 1 , ] 
			sum_same_group_k = np.sum(same_group_k, axis = 0)
			Vk = np.linalg.inv(self.sigma0) +   Nk*np.linalg.inv(self.sigma[i])
			sigmak = np.linalg.inv(Vk)
			mk = sigmak.dot(np.linalg.inv(self.sigma0).dot(self.mu0) + np.linalg.inv(self.sigma[i]).dot(sum_same_group_k))
			self.mu[i] = np.random.multivariate_normal(mk, sigmak)
		# return 0

	def samplingCovariance(self, data):
		sum_scale_mat = np.zeros_like(self.sigma0)
		for i in range(self.K):
			Nk = np.sum(self.Z[:, i] == 1)
			same_group_k = data[self.Z[:, i] == 1, ]
			broad_cast_diff = same_group_k - self.mu[i]
			scale_mat_k = np.transpose(broad_cast_diff).dot(broad_cast_diff)
			sum_scale_mat += scale_mat_k
			new_DF = self.D + Nk
			new_scale_mat = self.sigma0 + sum_scale_mat

			self.sigma[i] = invwishart.rvs(df = new_DF, scale = new_scale_mat, size = 1)

		# return 0

	def samplingZ(self, data):

		for i in range(self.N):
			numerator = []
			denomirator = 0.0
			for k in range(self.K):
				denomirator += self.pi[0][:, k]*multivariate_normal.pdf(data[i,:], self.mu[k], self.sigma[k])
				numerator.append(self.pi[0][:, k]*multivariate_normal.pdf(data[i,:], self.mu[k], self.sigma[k]))

			# pdb.set_trace()
			out_coefficient = numerator/denomirator
			out_coefficient = out_coefficient.reshape(out_coefficient.shape[0]*out_coefficient.shape[1],)
			mutinomial_coefficient = tuple(out_coefficient.tolist())
			# pdb.set_trace()
			self.Z[i] = np.random.multinomial(1, mutinomial_coefficient, size =1)
		return 0

def testGibbsSamplingGMM():
	N = 200
	K = 2
	D = 2

	mean1 = [1.0, 1.5]

	cov1 = [[1, 0], [0, 1]]

	X1 = np.random.multivariate_normal(mean1, cov1, 100)

	
	mean2 = [5.5, 4.3]

	cov2 = [[1, 0], [0, 1]]

	X2 = np.random.multivariate_normal(mean2, cov2, 100)

	mixture_gaussian_data = np.concatenate((X1, X2), axis = 0)

	gibbsDemo = GibbsSamplingGMM(2000)

	# gibbsDemo.sampling()
	# print(gibbsDemo.Z)
	gibbsDemo.sampling(N,K, D, mixture_gaussian_data)

	print(gibbsDemo.mu[0])
	print(gibbsDemo.mu[1])
	print(gibbsDemo.sigma[0])



testGibbsSamplingGMM()


