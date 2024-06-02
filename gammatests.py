gammas = [0.910, 
          0.915, 
          0.920, 
          0.925,
          0.930,
          0.935,
          0.940,
          0.945,
          0.950,
          0.955,
          0.960,
          0.965,
          0.970]

from agent import TetrisAgent
from tetrisEnv import TetrisEnv
from helper import plot
import torch
means = {}
for gamma in gammas:
    
    env = TetrisEnv()

    agent = TetrisAgent(gamma=gamma)
    agent.model.load_state_dict(torch.load("models/tetris_dqn2_IOpieces3.h5"))
    agent.target_model

    num_episodes = 3000
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
        agent.replay()
        print('Game', episode, "\nScore", score, '\nRecord: ', record, '\nEpsion: ', agent.epsilon)
        if episode % agent.update_target == 0:
            agent.update_target_model()
    means[gamma] = total_score / episode
print(means)