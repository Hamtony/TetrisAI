import pygame
import random
from figure import Figure
import time
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

class Tetris:
    def __init__(self, height, width):
        self.level = 2
        self.score = 0
        self.state = "start"
        self.field = []
        self.height = 0
        self.width = 0
        self.x = 100
        self.y = 60
        self.zoom = 20
        self.figure = None
        self.no_auto_freeze = 9999999999
        self.limit_no_freeze = 20
        self.hold_piece = Figure(3, 0, -1)
        self.pool = [0,1,2,3,4,5,6]
        self.queue = []
        self.hold_avaible = True
        self. just_rotate = False
    
        self.height = height
        self.width = width
        self.score = 0
        self.state = "start"
        
        for i in range(height):
            new_line = []
            for j in range(width):
                new_line.append(0)
            self.field.append(new_line)
        self.update_queue()
            

    def rand_fig(self):
        if len(self.pool) == 0:
            self.pool = [0,1,2,3,4,5,6]
        piece_index = random.randint(0,len(self.pool)-1)
        piece = self.pool[piece_index]
        self.pool.remove(piece)
        return piece
        
    def update_queue(self):
        if len(self.queue) < 5:
            self.queue.append(self.rand_fig())
            self.update_queue()
    def prinField(self, field):
        print("field:")
        for line in field:
            print(line)

    def new_figure(self):
        self.figure = Figure(3, 0, self.queue.pop(0))

    def intersects(self):
        intersection = False
        for i in range(len(self.figure.image())):
            for j in range(len(self.figure.image())):
                if self.figure.image()[i][j] != 0:
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True
        return intersection
    
    def check_tspin(self, t_field):
        if self.figure.type == 5:
            #print(self.figure.image())
            #self.prinField(self.figure.temp_field)
            for i in range(len(self.figure.image())):
                for j in range(len(self.figure.image()[i])):
                    if self.figure.image()[i][j] != 0 :
                        if self.figure.temp_field[i + self.figure.y-1][j + self.figure.x] > 0:
                            return True
        
    def all_clear(self):
        for i in range(len(self.field)):
            for j in range(len(self.field[i])):
                if self.field[i][j] != 0:
                    return False
        return True
    
    def calculate_score(self, lines, t_field):
        gained_score = 0
        if(self.figure.type == 5 and self.just_rotate):
            
            if self.check_tspin(t_field):
                print("Tspin!!!")
                gained_score = lines*2
                self.score = self.score+gained_score
                return gained_score
        if lines == 4:
            gained_score =  4
        if lines == 3:
            gained_score = 2
        if lines == 2:
            gained_score = 1
        if self.all_clear():
            gained_score = gained_score + 10
        self.score = self.score+gained_score
        return gained_score
            
        
            
            
    def break_lines(self, t_field):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        if lines > 0:
            self.calculate_score(lines, t_field)
            self.just_rotate = False

    def go_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            if self.no_auto_freeze < 0 and self.no_auto_freeze < 0: 
                self.freeze()
            else:
                self.limit_no_freeze = self.limit_no_freeze -1
                self.no_auto_freeze = self.no_auto_freeze - 1
        else:
            self.just_rotate = False

    def freeze(self):
        t_field = self.field.copy()
        for i in range(len(game.figure.image())):
            for j in range(len(game.figure.image())):
                if self.figure.image()[i][j] != 0:
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.break_lines(t_field)
        self.new_figure()
        if self.intersects():
            self.state = "gameover"
        self.no_auto_freeze = 5
        self.limit_no_freeze = 20
        self.update_queue()

    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x  
        else:
            self.just_rotate = False

    def rotate(self, direction):
        if self.figure.rotate(rot = direction, field = self.field, height = self.height, width = self.width):
            self.no_auto_freeze = 5
            self.just_rotate = True
        
        

    def hold(self):
        if self.hold_piece.type != -1:
            aux = self.hold_piece
            self.hold_piece = Figure(3, 0, self.figure.type)
            self.figure = aux
        else:
            self.hold_piece = Figure(3, 0, self.figure.type)
            self.new_figure()
            self.update_queue()
        
        
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
                game.go_space()
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