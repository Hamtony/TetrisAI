import pygame
import random
from figure import Figure
import time
from Tetris import Tetris
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


        
        
#keys
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

# Initialize the game engine
pygame.init()

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

size = (400, 500)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Tetris")

# Loop until the user clicks the close button.
done = False
clock = pygame.time.Clock()
fps = 25
game = Tetris(20, 10)
counter = 0
pressing_down = False

while not done:
    cur = time.time()
    if game.figure is None:
        game.new_figure()
        
    counter += 1
    if counter > 100000:
        counter = 0

    if counter % (fps // game.level // 2) == 0 or pressing_down:
        if game.state == "start":
            game.go_down()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == keys['rotatecc']:
                game.rotate(1)
            if event.key ==  keys['rotatec']:
                game.rotate(-1)
            if event.key == keys['rotatec180']:
                game.rotate(2)
            if event.key == keys['down']:
                pressing_down = True
            else:
                pressing_down = False
            if event.key == keys['left']:
                game.go_side(-1)
            if event.key == keys['right']:
                game.go_side(1)
            if event.key == keys['drop']:
                game.drop()
            if event.key == keys['reset']:
                game.__init__(20, 10)
            if event.key == keys['hold']:
                game.hold()



    screen.fill(WHITE)

    for i in range(game.height):
        for j in range(game.width):
            pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
            if game.field[i][j] > 0:
                pygame.draw.rect(screen, colors[game.field[i][j]],
                                 [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2, game.zoom - 1])

    if game.figure is not None:
        for i in range(len(game.figure.image())):
            for j in range(len(game.figure.image())):
                if game.figure.image()[i][j]:
                    pygame.draw.rect(screen, colors[game.figure.color],
                                     [game.x + game.zoom * (j + game.figure.x) + 1,
                                      game.y + game.zoom * (i + game.figure.y) + 1,
                                      game.zoom - 2, game.zoom - 2])

    font = pygame.font.SysFont('Calibri', 25, True, False)
    font1 = pygame.font.SysFont('Calibri', 65, True, False)
    text = font.render("Score: " + str(game.score), True, BLACK)
    text_game_over = font1.render("Game Over", True, (255, 125, 0))
    text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

    screen.blit(text, [0, 0])
    if game.state == "gameover":
        screen.blit(text_game_over, [20, 200])
        screen.blit(text_game_over1, [25, 265])

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()