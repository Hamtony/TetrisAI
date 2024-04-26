from gymnasium import spaces
import gymnasium
import pygame
import numpy as np
import random
from Tetris import Tetris

from gymnasium.envs.registration import register
from enum import Enum

colors = [
    (255, 255, 255),#I
    (98, 255, 245),#I
    (32, 24, 255),#J
    (252, 144, 21),#L
    (42, 245, 83),#S
    (255, 0, 0),#Z
    (255, 24, 247),#T
    (255, 208, 2),#O
]   
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

def dict_to_int_list(observation_space:dict):
    int_list = []
    for key, value in observation_space.items():
        if isinstance(value, int):
            int_list.append((value))
        elif isinstance(value, list):
            if isinstance(value[0], int):
                int_list.extend((value))
            elif isinstance(value[0], list):
                for other_list in value:
                    int_list.extend(other_list)
    return int_list

class TetrisEnv(gymnasium.Env):
    metadata = {"render_modes": ["human","none"], "render_fps":4}
    
    def __init__(self, height=20, width=10, render_mode="none"):
        self.width = width
        self.height = height
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
        if self.render_mode == "human":
            
            pygame.init()
            self.size = (400, 500)
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("Tetris")
            self.clock = pygame.time.Clock()
            self.fps = 60
            self.window = "active"
        else:
            self.window = "none"
        
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
        
    def get_state(self):
        return dict_to_int_list({
                "x_piece":self.game.figure.x,
                "y_piece":self.game.figure.y,
                "piece_type":self.game.figure.type,
                "piece_rotation":self.game.figure.rotation,
                "field":self.game.get_simplified_field(),
                "hold":self.game.hold_piece.type,
                "queue":self.game.queue
            })
    
    def reset(self, seed=random.randint(0,99999), options=None):
        super().reset(seed=seed)

        self.game.__init__(self.height, self.width)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

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
            self.render()

        return observation, reward, terminated, False, info
    
    def close(self):
        if self.window != "none":
            pygame.display.quit()
            pygame.quit()
            
    def render(self, mode = "human"):
        self.screen.fill(WHITE)
        for i in range(self.game.height):
            for j in range(self.game.width):
                pygame.draw.rect(self.screen, GRAY, [self.game.x + self.game.zoom * j, self.game.y + self.game.zoom * i, self.game.zoom, self.game.zoom], 1)
                if self.game.field[i][j] > 0:
                    pygame.draw.rect(self.screen, colors[self.game.field[i][j]],
                                    [self.game.x + self.game.zoom * j + 1, self.game.y + self.game.zoom * i + 1, self.game.zoom - 2, self.game.zoom - 1])

        if self.game.figure is not None:
            for i in range(len(self.game.figure.image())):
                for j in range(len(self.game.figure.image())):
                    if self.game.figure.image()[i][j]:
                        pygame.draw.rect(self.screen, colors[self.game.figure.color],
                                        [self.game.x + self.game.zoom * (j + self.game.figure.x) + 1,
                                        self.game.y + self.game.zoom * (i + self.game.figure.y) + 1,
                                        self.game.zoom - 2, self.game.zoom - 2])

        font = pygame.font.SysFont('Calibri', 25, True, False)
        font1 = pygame.font.SysFont('Calibri', 65, True, False)
        text = font.render("Score: " + str(self.game.score), True, BLACK)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

        self.screen.blit(text, [0, 0])
        if self.game.state == "gameover":
            self.screen.blit(text_game_over, [20, 200])
            self.screen.blit(text_game_over1, [25, 265])

        pygame.display.flip()
        self.clock.tick(self.fps)
        
    def seed(self, seed = None):
        pass
    
register(
    id='TetrisEnv-v0',
    entry_point='tetrisEnv:TetrisEnv',
    max_episode_steps=2999,
)
"""
env = TetrisEnv()
obs = env.reset()[0]

for i in range(9999999):
    rand_action = env.action_space.sample()
    obs, reward, terminated, _, _ = env.step(rand_action)
    if terminated:
        env.reset()
"""