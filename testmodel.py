from agent import TetrisAgent
from tetrisEnv import TetrisEnv, dict_to_int_list
import torch
agent =  TetrisAgent(epsilon=0, epsilon_min=0)
agent.model.load_state_dict(torch.load("models0106/tetris_dqn2_IOpieces414.h5",map_location=agent.model.device))
game = TetrisEnv(render_mode="human")
record = -999
score = 0
episode = 0
while(True):
    final_move = agent.act(game._get_obs())
    #preform move and get new state
    state_new_dict, reward, done, _, score_aux = game.step(final_move)
    score = score_aux['score']
    if done:
        game.reset()
        episode += 1
        if score > record:
            record = score
        print('Game', episode, "\nScore", score, '\nRecord: ', record)
        
