import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

n= 500
p =2

X,Y = make_circles(n_samples=n,factor=0.5,noise=0.05)

class NeuralLayer():
    def __init__(self, n_conn, n_neur, act_f) -> None:
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur)*2-1
        self.W = np.random.rand(n_conn, n_neur)*2-1
        
sigm = (lambda x: 1/(1 + np.e ** (-x)),
        lambda x: x * (1-x))

relu = (lambda x: np.maximum(0,x),)

class NeuralNet():
    def __init__(self, topology) -> None:
        l0 = NeuralLayer(p, 4, sigm)
        l1 = NeuralLayer(4, 8, sigm)
        