import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles



class NeuralLayer():
    def __init__(self, n_conn, n_neur, act_f) -> None:
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur)*2-1
        self.W = np.random.rand(n_conn, n_neur)*2-1
        
    def forward(self, X):
        z= X @ self.W + self.b
        return [self.act_f[0](x) for x in z], z
        
sigm = (lambda x: 1/(1 + np.e ** (-x)),
        lambda x: x * (1-x))

relu = (lambda x: np.maximum(0,x),)

class NeuralNet():
    def __init__(self, topology, act_f):
        self.nn = []
        for i, layer in enumerate(topology[:-1]):
            self.nn.append(NeuralLayer(topology[i], topology[i+1], act_f))
    
    def forward(self, X):
        output = X
        
        for layer in self.nn:
            output = layer.forward(output)[0]
        return output





def train():
    n= 500
    p =2
    X,Y = make_circles(n_samples=n,factor=0.5,noise=0.05)
    print(X[0])
    l2_cost = (lambda Yp, Yr: np.mean((Yp-Yr)**2),
           lambda Yp, Yr: (Yp - Yr))
    neuralNetwork = NeuralNet([p,5,5,1],relu)
    print(l2_cost[0](neuralNetwork.forward(X[0]),Y))
    
    #train
    
    
    
train()