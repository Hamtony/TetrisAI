from tetrisTrain import Agent
from tetrisEnv import TetrisEnv, dict_to_int_list
agent =  Agent()
agent.epsilon = 0
agent.min_epsilon = 0
agent.model.load(file_name="model.pht")
game = TetrisEnv(render_mode="human")
record = -999
score = 0
while(True):
    state_old = agent.get_state(game)
    #get move
    final_move = agent.get_action(state_old)
    #if (game._action_to_direction[final_move.index(1)] == "drop"):
    #    print("drop")
    #preform move and get new state
    state_new_dict, reward, done, _, score_aux = game.step(final_move)
    score = score_aux['score']
    if done:
        game.reset()
        agent.n_games += 1
        if score > record:
            record = score
        print('Game', agent.n_games, "\nScore", score, '\nRecord: ', record)
        