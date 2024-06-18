from tetrisEnv import TetrisEnv
from agent import TetrisAgent

def fitness(params : dict):
    agent = TetrisAgent(gamma=params['gamma'], learning_rate=params['lr'])
    metrics = {
        'drop' : params['drop'],
        'height' : params['height'],
        'bumpiness' : params['bumpiness'],
        'total_height' : params['total_height'],
        'holes' : params['holes']
    }
    env = TetrisEnv(metrics=metrics)