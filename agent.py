import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from model2 import TetrisModel

class TetrisAgent:
    def __init__(self, gamma=0.93, learning_rate=0.00001, epsilon=0.8, epsilon_min=0.175, 
                 epsilon_decay=0.9955, batch_size=69, max_memory=10_000, update_target=15, 
                 epsiont_every = 10):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.update_target = update_target
        self.learning_rate = learning_rate
        self.epsiont_every = epsiont_every
        
        self.memory = deque(maxlen=max_memory)
        self.model = TetrisModel().to(self.get_device())
        self.target_model = TetrisModel().to(self.get_device())
        self.update_target_model()
        self.espisode = 1
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon and not (self.espisode % self.epsiont_every == 0):
            return random.randrange(8)  # Random action
        
        field = torch.FloatTensor(state['field']).unsqueeze(0).unsqueeze(0).to(self.get_device())
        other_state = torch.FloatTensor(state['other_state']).unsqueeze(0).to(self.get_device())
        
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(field, other_state)
        self.model.train()
        
        return torch.argmax(act_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            field = torch.FloatTensor(state['field']).unsqueeze(0).unsqueeze(0).to(self.get_device())
            other_state = torch.FloatTensor(state['other_state']).unsqueeze(0).to(self.get_device())
            
            next_field = torch.FloatTensor(next_state['field']).unsqueeze(0).unsqueeze(0).to(self.get_device())
            next_other_state = torch.FloatTensor(next_state['other_state']).unsqueeze(0).to(self.get_device())
            #print("input to model:")
            #print(field)
            #print(other_state)
            #print("input2 to model:")
            #print(next_field)
            #print(next_other_state)
            target = self.model(field, other_state)#get prediction of action vetor with the main model
            if done:#check if is a terminal state
                target[0][action] = reward #if it is, sets the Q value for this action with the reward
            else:
                next_target = self.target_model(next_field, next_other_state)#get action prection for the nextstate with target model
                target[0][action] = reward + self.gamma * torch.max(next_target[0]).item() # sets the Q value with the formula: reward * discount * Q value of best action
            
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(field, other_state), target)#get the loss with MSE of the model's prediction vs target Q values
                loss.backward()#backpropagation
                self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.espisode += 1
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
