import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from collections import deque
import random
from tetrisEnv import TetrisEnv, dict_to_int_list
from collections import deque
from helper import plot
from model import Linear_QNet, QTrainer, cuda

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 1
        self.epsilon = 60000 #randomness
        self.gamma = 0.25 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #popleft
        self.model = Linear_QNet(210,80,8)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def get_state(self, game: TetrisEnv):
        state = game._get_obs()
        return np.array(dict_to_int_list(state),dtype=int)
    
    def remember(self, state ,action, reward, next_state, done):
        self.memory.append((state ,action, reward, next_state, done)) #popleft if max meory is reached
    
    def train_long_memory(self):
        if len(self.memory)>BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)#list of tuples
        else:
            mini_sample = self.memory
        states ,actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states ,actions, rewards, next_states, dones)
        
    
    def train_short_memory(self, state ,action, reward, next_state, done):
        self.trainer.train_step(state ,action, reward, next_state, done)
    
    def get_action(self, state):
        #random moves: exploration
        if self.epsilon > 0:
            self.epsilon = 60000 - self.n_games
        final_move = [0,0,0,0,0,0,0,0]
        if random.randint(0,100000) < self.epsilon:
            move = random.randint(0,7)
            final_move[move]=1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=cuda)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = TetrisEnv()
    while(True):
        #get old state
        state_old = agent.get_state(game)
        #get move
        final_move = agent.get_action(state_old)
        #preform move and get new state
        state_new_dict, reward, done, _, score_aux = game.step(final_move)
        score = score_aux['score']
        
        state_new = dict_to_int_list(state_new_dict)
        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        #remember
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            #train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, "\nScore", score, '\nRecord: ', record)
        
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            if agent.n_games % 5 == 0:
                try:
                    title = "gamma_" + str(agent.gamma) + " LR_" + str(LR)    
                    plot(plot_scores,plot_mean_scores,title)
                except: pass
        
        

if __name__ == '__main__':
    train()