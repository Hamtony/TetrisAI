import pygame
import random
from figure import Figure
import time
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
        self.figure = Figure(3, 0, -1)
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
        
        for i in range(height):
            new_line = []
            for j in range(width):
                new_line.append(0)
            self.field.append(new_line)
        self.update_queue()
        self.new_figure()
        self.update_queue()
            
    def get_simplified_field(self):
        f = list()
        for i in range(self.height):
            new_line = []
            for j in range(self.width):
                if self.field[i][j] > 0:
                    new_line.append(1)
                else:
                    new_line.append(0)
            f.append(new_line)
        return f

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
            gained_score = 3
        if lines == 2:
            gained_score = 2
        if lines == 1:
            gained_score = 1
        if self.all_clear():
            gained_score = gained_score + 10
        self.score = self.score+gained_score
        return gained_score*5
            
        
            
            
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

    def drop(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            if self.no_auto_freeze < 0 and self.no_auto_freeze < 0: 
                pass
                #self.freeze()
            else:
                self.limit_no_freeze = self.limit_no_freeze -1
                self.no_auto_freeze = self.no_auto_freeze - 1
        else:
            self.just_rotate = False

    def freeze(self):
        t_field = self.field.copy()
        for i in range(len(self.figure.image())):
            for j in range(len(self.figure.image())):
                if self.figure.image()[i][j] != 0:
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.break_lines(t_field)
        self.new_figure()
        if self.intersects():
            self.state = "gameover"
        self.no_auto_freeze = 5
        self.limit_no_freeze = 20
        self.update_queue()
        self.hold_avaible = True

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
        if self.hold_avaible:
            if self.hold_piece.type != -1:
                aux = self.hold_piece
                self.hold_piece = Figure(3, 0, self.figure.type)
                self.figure = aux
            else:
                self.hold_piece = Figure(3, 0, self.figure.type)
                self.new_figure()
                self.update_queue()
            self.hold_avaible = False