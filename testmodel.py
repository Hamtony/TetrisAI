from agent import TetrisAgent
from tetrisEnv import TetrisEnv, dict_to_int_list
import torch
import sys
def main(argv):
    file = argv[1]
    agent =  TetrisAgent(epsilon=0, epsilon_min=0)
    agent.model.load_state_dict(torch.load(file,map_location=agent.model.device))
    game = TetrisEnv(render_mode="human")
    record = -999
    score = 0
    episode = 0
    while(True):
        final_move = agent.act(game._get_obs())
        #preform move and get new state
        state_new_dict, reward, done, _, score_aux = game.step(final_move)
        print(game._get_obs())
        score = score_aux['score']
        if done:
            game.reset()
            episode += 1
            if score > record:
                record = score
            print('Game', episode, "\nScore", score, '\nRecord: ', record)
        
if __name__ == '__main__':
    main(sys.argv)