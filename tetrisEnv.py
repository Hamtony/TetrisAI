from gymnasium import spaces
import gymnasium
import pygame
import numpy as np
import random
from Tetris import Tetris
from gymnasium.envs.registration import register
from enum import Enum


class TetrisEnv(gymnasium.Env):
    metadata = {"render_mode": ["human","rgb_array"], "render_fps":4}
    
    def __init__(self, render_mode=None):
        self.width = 10
        self.height = 20
        self.game = Tetris(self.height,self.width)
        self.actualscore = self.game.score
        self.observation_space = spaces.Dict(
            {
                "x_piece":spaces.Discrete(self.width,start=-2),
                "y_piece":spaces.Discrete(self.height,start=-2),
                "piece_type":spaces.Discrete(6),
                "piece_rotation":spaces.Discrete(3),
                "field":spaces.Box(low=0,high=1, shape=(20,10),dtype=np.int8),
                "hold":spaces.Discrete(6),
                "queue":spaces.Box(low= 0, high=6, shape=(5,),dtype=np.int8)
            }
        )
        #ACTIONS = ['Down','Left','Right','Rotatec','Rotatecc', 'Rotate180', 'hold', 'drop' ]
        self.action_space = spaces.Discrete(8,start=0)
        self._action_to_direction = {
            0: 'Down',
            1: 'Left',
            2: 'Right',
            3: 'Rotatec',
            4: 'Rotatecc',
            5: 'Rotate180',
            6: 'hold',
            7: 'drop',
        }
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
    def _get_info(self):
        return {"score": self.game.score}
    
    def _get_obs(self):
        return {
                "x_piece":self.game.figure.x,
                "y_piece":self.game.figure.y,
                "piece_type":self.game.figure.type,
                "piece_rotation":self.game.figure.rotation,
                "field":self.game.get_simplified_field(),
                "hold":self.game.hold_piece.type,
                "queue":self.game.queue
            }
    
    def reset(self, seed=random.randint(0,99999), options=None):
        super().reset(seed=seed)

        self.game.__init__(self.height, self.width)

        observation = self._get_obs()
        info = self._get_info()

        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, info
    
    def step(self, action):
         #ACTIONS = ['Down','Left','Right','Rotatec','Rotatecc', 'Rotate180', 'hold', 'drop' ]

        if self._action_to_direction[action] == 'Down':
            self.game.go_down()
        elif self._action_to_direction[action] == 'Left':
            self.game.go_side(-1)
        elif self._action_to_direction[action] == 'Right':
            self.game.go_side(1)
        elif self._action_to_direction[action] == 'Rotatec':
            self.game.rotate(-1)
        elif self._action_to_direction[action] == 'Rotatecc':
            self.game.rotate(1)
        elif self._action_to_direction[action] == 'Rotate180':
            self.game.rotate(2)
        elif self._action_to_direction[action] == 'hold':
            self.game.hold()
        elif self._action_to_direction[action] == 'drop':
            self.game.drop()
            

        terminated = self.game.state == "gameover"
        
        if self.game.score > self.actualscore:
            reward = self.game.score - self.actualscore
            print(reward)
        else:
            reward = 0

        observation = self._get_obs()
        
        info = self._get_info()

        if self.render_mode == "human":
            pass
            #self._render_frame()

        return observation, reward, terminated, False, info
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            
    def render(self, mode = "human"):
        pass
    def seed(self, seed = None):
        pass
    
register(
    id='TetrisEnv-v0',
    entry_point='tetrisEnv:TetrisEnv',
    max_episode_steps=2999,
)

env = TetrisEnv()
env.__init__()
env.action_space.seed()

observation, info = env.reset()

for _ in range(1099900):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()
