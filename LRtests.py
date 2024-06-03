lrs = [0.0005, 0.0001,0.00005, 0.000025, 0.00001, 0.000005, 0.0000025, 0.000001, 0.0000005,
       0.00000025, 0.000000001]

from agent import TetrisAgent
from tetrisEnv import TetrisEnv
from helper import plot
import torch

means = {}

for lr in lrs:
    
    env = TetrisEnv()

    agent = TetrisAgent(gamma=0.992,learning_rate=lr)
    agent.model.load_state_dict(torch.load("models/tetris_dqn2_IOpieces3.h5"))
    agent.target_model

    num_episodes = 5000
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    for episode in range(1,num_episodes):
        
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            score = info['score']
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / episode
                plot_mean_scores.append(mean_score)
                if episode % 5 == 0:
                    #try:
                    title = "gamma_" + str(agent.gamma) + "LR_" + str(agent.learning_rate)    
                    plot(plot_scores,plot_mean_scores,title)
                    #except: pass
                break
            if score > record:
                record = score
                agent.save("modelsgammatests/tetris_dqn2_IOpieces"+ str(agent.learning_rate) +".h5")
        agent.replay()
        print('Game', episode, "\nScore", score, '\nRecord: ', record, '\nEpsion: ', agent.epsilon)
        if episode % agent.update_target == 0:
            agent.update_target_model()