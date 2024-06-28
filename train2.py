from agent import TetrisAgent
from tetrisEnv import TetrisEnv
from helper import plot
import torch
metrics = {
    "drop": 0.2,
    "height": 8,
    "bumpiness": 13,
    "total_height": 14,
    "holes": 15
}
env = TetrisEnv(metrics=metrics,render_mode="human") 

agent = TetrisAgent(gamma=0.93, learning_rate=2.5e-6, epsilon=0.65)
agent.model.load_state_dict(torch.load("models/tetris_dqn3_IOpieces2406_2.h5",map_location=agent.model.device))
agent.target_model.load_state_dict(torch.load("models/tetris_dqn3_IOpieces2406_2.h5",map_location=agent.model.device))
num_episodes = 300_000
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
                try:
                    title = "gamma_" + str(agent.gamma) + "LR_" + str(agent.learning_rate)    
                    plot(plot_scores,plot_mean_scores,title)
                except: pass
            break
        if score > record:
            record = score
            agent.save("models/tetris_dqn3_IOpieces2506.h5")
    agent.replay()
    print('Game', episode, "\nScore", score, '\nRecord: ', record, '\nEpsion: ', agent.epsilon)
    if episode % agent.update_target == 0:
        agent.update_target_model()
        
