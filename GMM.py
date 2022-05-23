import csv
import math
import scipy as sp
import scipy.special as spe
import scipy.integrate as integrate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense,Input
from keras.models import Model
import keras
import sklearn.metrics 
import time

# Data without the noise z (I think we should start with this one until we figure out how the noise was added)
def make_GMM(dim, N, var, plot): 
  '''
  input:  dim = dimension D
          N = number of samples 
          var = standard deviation (sigma) for all clusters 
  output: X = data points of shape [N, dim]
          Y = labels of shape [N]
  '''

  classes = 2
  xor_dim = 2

  # cluster means of the 4 GMs
  mu1 = [0,1]
  mu2 = [0,-1]
  mu3 = [1,0]
  mu4 = [-1,0]

  # set means of higher dimensions to zero
  if dim>2:
    mu1 = np.append(mu1, np.zeros((dim-2), dtype=int))
    mu2 = np.append(mu2, np.zeros((dim-2), dtype=int))
    mu3 = np.append(mu3, np.zeros((dim-2), dtype=int))
    mu4 = np.append(mu4, np.zeros((dim-2), dtype=int))

  # make shared diagonal coariance matrix 
  
  cov = np.eye(dim)* var 

  # make datapoints for the 4 clusters 

  cluster1 = np.random.multivariate_normal(mu1, cov, size = int(N/4) , check_valid='warn', tol=1e-8)
  cluster2 = np.random.multivariate_normal(mu2, cov, size = int(N/4) , check_valid='warn', tol=1e-8)
  cluster3 = np.random.multivariate_normal(mu3, cov, size = int(N/4) , check_valid='warn', tol=1e-8)
  cluster4 = np.random.multivariate_normal(mu4, cov, size = int(N/4) , check_valid='warn', tol=1e-8)

  # make labels for the 4 clusters 
  label1 = np.ones(int(N/4), dtype=int)*(-1)
  label2 = np.ones(int(N/4), dtype=int)*(-1)
  label3 = np.ones(int(N/4), dtype=int)*(1)
  label4 = np.ones(int(N/4), dtype=int)*(1)

  if plot==True:
    # visualize 
    plt.scatter(cluster1[:,0],cluster1[:,1] , color='red')
    plt.scatter(cluster2[:,0],cluster2[:,1] , color='red')
    plt.scatter(cluster3[:,0],cluster3[:,1] , color='blue')
    plt.scatter(cluster4[:,0],cluster4[:,1] , color='blue')

    plt.title('{} dimensios'.format(dim))
    plt.xlabel('first dimension')
    plt.ylabel('second dimension')
    plt.show()

    
    plt.scatter(cluster1[:,1],cluster1[:,2] , color='red')
    plt.scatter(cluster2[:,1],cluster2[:,2] , color='red')
    plt.scatter(cluster3[:,1],cluster3[:,2] , color='blue')
    plt.scatter(cluster4[:,1],cluster4[:,2] , color='blue')

    plt.title('{} dimensios'.format(dim))
    plt.xlabel('second dimension')
    plt.ylabel('third dimension')
    plt.show()
  
    plt.scatter(cluster1[:,2],cluster1[:,3] , color='red')
    plt.scatter(cluster2[:,2],cluster2[:,3] , color='red')
    plt.scatter(cluster3[:,2],cluster3[:,3] , color='blue')
    plt.scatter(cluster4[:,2],cluster4[:,3] , color='blue')

    plt.title('{} dimensios'.format(dim))
    plt.xlabel('third dimension')
    plt.ylabel('fourth dimension')
    plt.show()
  
  return np.vstack((cluster1, cluster2, cluster3, cluster4)), np.hstack((label1, label2, label3, label4)) 

# Currently the dataset is ordered (-1) the first half and (1) the second half. We will now
# shuffle the data and then split it into the 3 subsets.

def make_splits(N, X, Y):
  indices = np.arange(N)
  np.random.shuffle(indices)

  X = X[indices]
  Y = Y[indices]
  X_train = X[0:int(N*0.6),:]
  X_val   = X[int(N*0.6):int(N*0.8),:]
  X_test  = X[int(N*0.8):N,:]

  Y_train = Y[0:int(N*0.6)]
  Y_val   = Y[int(N*0.6):int(N*0.8)]
  Y_test  = Y[int(N*0.8):N]

  #print('Shapes of train datasets: ',np.shape(X_train), np.shape(Y_train))
  return X_train, X_val, X_test, Y_train, Y_val, Y_test



def train_and_test(X_train, X_val, X_test, Y_train, Y_val, Y_test):

  def neural(nr_samples, nr_dim, nr_hln, std_weights):
      inputs = Input(shape=(nr_dim))
      initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=std_weights)
      x = Dense(nr_hln, activation="relu",kernel_initializer=initializer)(inputs)/ math.sqrt(nr_samples)
      output = Dense(1)(x)
      return Model(inputs=inputs,outputs=output)

  N = np.shape(X_train)[0]
  D = np.shape(X_train)[1]
  K = 4
  lr = 0.1
  std_weights = 1
  batch_size = 1 # for online learning 
  num_epochs = 100

  neural_network = neural(batch_size, D, K, std_weights)
  opt = tf.keras.optimizers.SGD(learning_rate=lr, clipvalue=1.0)
  metric = tf.keras.metrics.MeanSquaredError()
  loss_binary = tf.keras.losses.MeanSquaredError()
  neural_network.compile(loss=loss_binary, optimizer=opt, metrics=metric)

  history     = neural_network.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, shuffle=True, validation_data=(X_val,Y_val), verbose=0)
  evaluation  = neural_network.evaluate(X_val,Y_val, batch_size=batch_size, verbose=0) 

  # test network 
  final_test = neural_network.predict(X_test, batch_size=batch_size) 
  
  return final_test



################################################################################
N   = 25*4
dim = 100
num_sigmas = 10*3
sigma1 = np.logspace(-2, -1, num= int(num_sigmas/3))
sigma2 = np.logspace(-1, 0,  num= int(num_sigmas/3))
sigma3 = np.logspace(0 , 1,  num= int(num_sigmas/3))
sigma  = np.round(np.append(sigma1, np.append(sigma2, sigma3)), 5)

pred_1000 = np.zeros((num_sigmas, int(N*0.2)))
error_1000 = np.zeros((num_sigmas))
for i in range(0, num_sigmas):
  start = time.time()
  X1000, Y1000  = make_GMM(dim = dim, N=N , var = sigma[i], plot = False )
  X_train, X_val, X_test, Y_train, Y_val, Y_test = make_splits(N, X1000, Y1000)
  final_test  = np.squeeze(train_and_test(X_train, X_val, X_test, Y_train, Y_val, Y_test))
  # apply heaviside step function to network output 
  final_test[final_test>0] = 1
  final_test[final_test<=0] = -1
  # use zero one loss as error metric 
  error_1000[i]  = sklearn.metrics.zero_one_loss(Y_test, final_test, normalize=True, sample_weight=None)
  pred_1000[i, :]  = final_test
  end             = time.time()
  duration        = end-start

  print('For Variance {} we have an accuracy of {}. Estimated remaining time: {} min'.format(sigma[i], np.round(error_1000[i], 3), int(duration*(int(num_sigmas)-i)/60)) )

np.save('/Users/aliyaablitip/Desktop/my_data/pred100', pred_1000, allow_pickle=True, fix_imports=True)
np.save('/Users/aliyaablitip/Desktop/my_data/error_100', error_1000, allow_pickle=True, fix_imports=True)


################################################################################
