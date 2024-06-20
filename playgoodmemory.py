import pygame
import random
from figure import Figure
import time
from tetrisEnv import TetrisEnv
from collections import deque
import json
keys = {
    'up': pygame.K_UP,
    'down': pygame.K_DOWN,
    'left': pygame.K_LEFT,
    'right': pygame.K_RIGHT,
    'rotatec': pygame.K_z,
    'rotatecc': pygame.K_x,
    'rotatec180': pygame.K_a,
    'hold': pygame.K_c,
    'pause': pygame.K_ESCAPE,
    'drop': pygame.K_SPACE,
    'reset': pygame.K_r
}
metrics = {
    "drop": 1,  
    "height": 16,
    "bumpiness": 16,
    "total_height": 7,
    "holes": 1
}
env = TetrisEnv(metrics=metrics,render_mode="human")
good_memory = {"data":[]}
state, _ = env.reset()
while True:
    action = -1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == keys['rotatecc']:
                action = 4
            if event.key ==  keys['rotatec']:
                action = 3
            if event.key == keys['rotatec180']:
                action = 7
            if event.key == keys['down']:
                action = 5
            if event.key == keys['left']:
                action = 1
            if event.key == keys['right']:
                action = 2
            if event.key == keys['drop']:
                action = 0
            if event.key == keys['reset']:
                env.reset()
            if event.key == keys['hold']:
                action = 6
            if event.key == pygame.K_ESCAPE:
                json.dump(good_memory, open("real_data.json"))
    if action != -1:
        next_state, reward, done, _, info = env.step(action)
        score = info['score']
        good_memory['data'].append((state, action, reward, next_state, done))
        state = next_state