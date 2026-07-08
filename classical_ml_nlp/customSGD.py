class customSGD(object):
	"""docstring for customSGD"""
	Weights = None
	def __init__(self, loss,penalty,alpha,max_epoch,batch_size):
		super(customSGD, self).__init__()
		self.loss = loss
		self.penalty = penalty
		self.alpha = alpha
		self.max_iter = max_epoch
		self.batch_size = batch_size
		import numpy as np 
		from matplotlib import pyplot as plt
		from matplotlib import animation

	def init_weight(self,data_shape):
		w = np.random.normal(loc=0.0,scale=0.1,size=data_shape)
		print("Weights Initiliazed !! Shape is: ", w.shape)
		return w

	def SGD(self,x,y): 
		# sampling x without replacement
		#data_mat = np.c_[x,y] 
		#x_sample = np.random.choice(data_mat,size=(100*data_mat.shape[1]),replace=False)
		# initializing the weight vector
		# Adding the bias term to training data to calculate the intercept as x0 * w0 
		x = np.c_[np.ones((x.shape[0])),x]
		w = self.init_weight(data_shape=x.shape[1])
		w_next = None
		iter_count = 0 
		loss = list()
		print("starting Batch SGD now ... ")
		for k in range(self.batch_size,(y.shape[0]-self.batch_size)+1,self.batch_size):
			if iter_count < self.max_iter and abs(loss[iter_count] - loss[iter_count-1]) > 0.01:
				w_next = w + (2*alpha) * (np.matmul(x[k:k+self.batch_size].T,(y[k:k+self.batch_size] - self.y_pred(w,x[k:k+self.batch_size]))) / self.batch_size)
				# plotting the current loss vs iterations
				loss.append(self.mse_loss(x=x[k:k+self.batch_size],y[k:k+self.batch_size],w)) # loss caused by previous w instead of next
				iter_count +=1
				w = w_next
				# stopping SGD with tolerance of 0.01
			else:
				print("*********** SGD has converged to minima ***************")
				fig,ax = plt.subplots()
				ax.plot(iter_count,loss)
				ax.set_title("MSE vs iterations ")
				plt.show()
				return w_next
		
	# compute the MSE loss and weight difference 
	def mse_loss(self,x,y,w):
		yield np.sum((y - self.y_pred(w,x))**2) / x.shape[0]

	# predict method to predict hyper planes

	def y_pred(self,w,x):
		return np.matmul(x,w)

	# fit method to fit the training data 
	def fit(self,x,y):
		# call the SGD function to converge 
		customSGD.Weights = self.SGD(x,y)

	# tranform the given matrix
	def transform(self,x):
		return self.y_pred(customSGD.Weights,x)

	def coffecients_(self):
		return customSGD.Weights

		
