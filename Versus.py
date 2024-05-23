from Tetris import Tetris
from tetrisEnv import TetrisEnv
import pygame
from pieces import pieces
from tetrisTrain import Agent
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
class Versus():
    
    def __init__(self, bot_delay=40):
        self.gamePlayer = Tetris(20,10)
        self.counter = 0
        self.tetrisAgent = TetrisEnv()
        self.agent = Agent()
        self.agent.epsilon = 0
        self.agent.min_epsilon = 0
        pygame.init()
        self.size = (400+400, 500)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.pps = 0.80
        self.zoom = 20
        self.pressing_down = False
        self.bot_delay = bot_delay
        self.actual_bot_delay = 0
    def draw(self):
        self.screen.fill(GRAY)
        
        for i in range(self.gamePlayer.height):
            for j in range(self.gamePlayer.width):
                pygame.draw.rect(self.screen, BLACK, [self.gamePlayer.x + self.gamePlayer.zoom * j, self.gamePlayer.y + self.gamePlayer.zoom * i, self.gamePlayer.zoom, self.gamePlayer.zoom], 1)
                if self.gamePlayer.field[i][j] > 0:
                    pygame.draw.rect(self.screen, colors[self.gamePlayer.field[i][j]],
                                    [self.gamePlayer.x + self.gamePlayer.zoom * j + 1, self.gamePlayer.y + self.gamePlayer.zoom * i + 1, self.gamePlayer.zoom - 2, self.gamePlayer.zoom - 1])
        
        if self.gamePlayer.figure is not None:
            for i in range(len(self.gamePlayer.figure.image())):
                for j in range(len(self.gamePlayer.figure.image())):
                    if self.gamePlayer.figure.image()[i][j]:
                        pygame.draw.rect(self.screen, colors[self.gamePlayer.figure.color],
                                        [self.gamePlayer.x + self.gamePlayer.zoom * (j + self.gamePlayer.figure.x) + 1,
                                        self.gamePlayer.y + self.gamePlayer.zoom * (i + self.gamePlayer.figure.y) + 1,
                                        self.gamePlayer.zoom - 2, self.gamePlayer.zoom - 2])
                        
        for i in range(self.tetrisAgent.game.height):
            for j in range(self.tetrisAgent.game.width):
                pygame.draw.rect(self.screen, BLACK, [400+(self.tetrisAgent.game.x + self.tetrisAgent.game.zoom * j), self.tetrisAgent.game.y + self.tetrisAgent.game.zoom * i, self.tetrisAgent.game.zoom, self.tetrisAgent.game.zoom],1)
                if self.tetrisAgent.game.field[i][j] > 0:
                    pygame.draw.rect(self.screen, colors[self.tetrisAgent.game.field[i][j]],
                                    [400+(self.tetrisAgent.game.x + self.tetrisAgent.game.zoom * j + 1), 
                                     (self.tetrisAgent.game.y + self.tetrisAgent.game.zoom * i + 1),
                                     (self.tetrisAgent.game.zoom - 2), 
                                     (self.tetrisAgent.game.zoom - 1)])
        piece_order = 0
        for actual_piece in self.gamePlayer.queue:
            for i in range(len(pieces[actual_piece][0])):
                for j in range(len(pieces[actual_piece][i])):
                    if pieces[actual_piece][0][i][j] > 0:
                        pygame.draw.rect(self.screen, colors[actual_piece+1],
                                        [self.gamePlayer.x + self.gamePlayer.zoom * (j + self.gamePlayer.width) + 1,
                                        self.gamePlayer.y + self.gamePlayer.zoom * (i + (piece_order)*4) + 1,
                                        self.gamePlayer.zoom - 2, self.gamePlayer.zoom - 2])
            piece_order += 1
        for i in range(len(self.gamePlayer.hold_piece.image())):
            for j in range(len(self.gamePlayer.hold_piece.image()[i])):
                if self.gamePlayer.hold_piece.image()[i][j] > 0:
                    pygame.draw.rect(self.screen, colors[self.gamePlayer.hold_piece.color],
                                    [self.gamePlayer.x + self.gamePlayer.zoom * (j -4) + 1,
                                    self.gamePlayer.y + self.gamePlayer.zoom * (i) + 1,
                                    self.gamePlayer.zoom - 2, self.gamePlayer.zoom - 2])
                    
        piece_order = 0
        for actual_piece in self.tetrisAgent.game.queue:
            for i in range(len(pieces[actual_piece][0])):
                for j in range(len(pieces[actual_piece][i])):
                    if pieces[actual_piece][0][i][j] > 0:
                        pygame.draw.rect(self.screen, colors[actual_piece+1],
                                        [400+(self.tetrisAgent.game.x + self.tetrisAgent.game.zoom * (j + self.tetrisAgent.game.width) + 1),
                                        self.tetrisAgent.game.y + self.tetrisAgent.game.zoom * (i + (piece_order)*4) + 1,
                                        self.tetrisAgent.game.zoom - 2, self.tetrisAgent.game.zoom - 2])
            piece_order += 1
        for i in range(len(self.tetrisAgent.game.hold_piece.image())):
            for j in range(len(self.tetrisAgent.game.hold_piece.image()[i])):
                if self.tetrisAgent.game.hold_piece.image()[i][j] > 0:
                    pygame.draw.rect(self.screen, colors[self.tetrisAgent.game.hold_piece.color],
                                    [400+(self.tetrisAgent.game.x + self.tetrisAgent.game.zoom * (j -4) + 1),
                                    self.tetrisAgent.game.y + self.tetrisAgent.game.zoom * (i) + 1,
                                    self.tetrisAgent.game.zoom - 2, self.tetrisAgent.game.zoom - 2])
        
        if self.tetrisAgent.game.figure is not None:
            for i in range(len(self.tetrisAgent.game.figure.image())):
                for j in range(len(self.tetrisAgent.game.figure.image())):
                    if self.tetrisAgent.game.figure.image()[i][j]:
                        pygame.draw.rect(self.screen, colors[self.tetrisAgent.game.figure.color],
                                        [(self.tetrisAgent.game.x + self.tetrisAgent.game.zoom * (j + self.tetrisAgent.game.figure.x) + 1)+400,
                                        self.tetrisAgent.game.y + self.tetrisAgent.game.zoom * (i + self.tetrisAgent.game.figure.y) + 1,
                                        self.tetrisAgent.game.zoom - 2, self.tetrisAgent.game.zoom - 2])
                    
                    
        font = pygame.font.SysFont('Calibri', 25, True, False)
        font1 = pygame.font.SysFont('Calibri', 65, True, False)
        text = font.render("Score: " + str(self.gamePlayer.score), True, BLACK)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

        self.screen.blit(text, [0, 0])
        if self.gamePlayer.state == "gameover":
            self.screen.blit(text_game_over, [20, 200])
            self.screen.blit(text_game_over1, [25, 265])

        pygame.display.flip()
        self.clock.tick(self.fps)
        
    def agent_play(self):
        if(self.actual_bot_delay >= self.bot_delay):
            state_old = self.agent.get_state(self.tetrisAgent)
            final_move = self.agent.get_action(state_old)
            obs, reward, terminated, _, _ = self.tetrisAgent.step(final_move)
            self.actual_bot_delay = 0
        else:
            self.actual_bot_delay += 1
            
    def user_inputs(self):
        self.counter += 1
        if self.counter > 100000:
            self.counter = 0
        if self.counter % (self.fps // self.gamePlayer.level // 2) == 0 or self.pressing_down:
            if self.gamePlayer.state == "start":
                self.gamePlayer.go_down()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == keys['rotatecc']:
                    self.gamePlayer.rotate(1)
                if event.key ==  keys['rotatec']:
                    self.gamePlayer.rotate(-1)
                if event.key == keys['rotatec180']:
                    self.gamePlayer.rotate(2)
                if event.key == keys['down']:
                    self.pressing_down = True
                if event.key == keys['left']:
                    self.gamePlayer.go_side(-1)
                if event.key == keys['right']:
                    self.gamePlayer.go_side(1)
                if event.key == keys['drop']:
                    self.gamePlayer.drop()
                if event.key == keys['reset']:
                    self.gamePlayer.__init__(20, 10)
                if event.key == keys['hold']:
                    self.gamePlayer.hold()
        if len(pygame.event.get()) == 0:
            self.pressing_down = False
        
    
    def update(self):
        self.user_inputs()
        self.agent_play()
        self.draw()
        
        
        
match = Versus(40)
while(True):
    match.update()