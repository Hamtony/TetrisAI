from agent import TetrisAgent
from tetrisEnv import TetrisEnv
from helper import plot
import torch
import json
env = TetrisEnv(render_mode="human")

agent = TetrisAgent(gamma=0.97, learning_rate=0.000005)
#agent.model.load_state_dict(torch.load("tetris_dqn2_IOpieces0806.h5",map_location=agent.model.device))
agent.target_model

num_episodes = 100_000
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
                data = {"data":[]}
                for state, action, reward, next_state, done in agent.memory:
                    data["data"].append((state, action, reward, next_state, done))
                with open('train.json', 'w') as f:
                    json.dump(data, f)
                try:
                    title = "gamma_" + str(agent.gamma) + "LR_" + str(agent.learning_rate)    
                    plot(plot_scores,plot_mean_scores,title)
                except: pass
            break
        if score > record:
            record = score
            #agent.save("tetris_dqn3_IOpieces1006.h5")

                
    agent.replay()
    #print('Game', episode, "\nScore", score, '\nRecord: ', record, '\nEpsion: ', agent.epsilon)
    if episode % agent.update_target == 0:
        agent.update_target_model()
        
