import math
class Perceptron:
    def __init__(self, inputs, weihts = [], bias=0., activation=lambda x: x):
        self.inputs = inputs
        self.weihts = weihts
        self.bias = bias
        self.activation = activation
        self.last_out = None
        self.last_X = None
        
    def forward(self, X):
        n = 0.
        for i in range(len(X)):
            n += X[i]*self.weihts[i]
        n += self.bias
        out = self.activation(n)
        self.last_out = out
        return out
    def backward(self, target, lr):
        delta = self.last_out * (1 - self.last_out) * (target - self.last_out)
        for weith in self.weihts:
            weith = weith + (lr*delta)
        
class Layer:
    def __init__(self, inputs, outputs, activation) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation
        self.perceptrones:list[Perceptron] = [Perceptron(inputs, activation=activation) for i in range(outputs)]
    def forward(self, X):
        return [self.perceptrones[i].forward(X) for i in range(len(self.perceptrones))] 
    def learn(self, Target, lr):
        deltas
        
log_sigmoid = lambda x : 1/(1+pow(math.e,x))
        
class NeuralNetwork:
    def __init__(self) -> None:
        self.h = Layer(2, 2, log_sigmoid)
        self.o = Layer(2, 1, log_sigmoid)
        self.h.perceptrones[0].weihts = [0.1,-0.7]
        self.h.perceptrones[1].weihts = [0.5,0.3]
        self.o.perceptrones[0].weihts = [0.2,0.4]
        
    def forward(self, X):
        X = self.h.forward(X)
        X = self.o.forward(X)    
        return X
    
    def learn(self, Target, lr):
        deltas_o = self.o.learn(Target)

data = [
    [[0,0],[0]],
    [[0,1],[1]],
    [[1,0],[1]],
    [[1,1],[0]]
]
lr = 0.25
nn = NeuralNetwork()
out = nn.forward(data[0][0])
nn.learn(data[0][1], lr)

