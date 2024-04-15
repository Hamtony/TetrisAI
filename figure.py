from pieces import pieces
import random
from kickdata import wallkickdata

class Figure:
    x = 0
    y = 0

    figures = pieces

    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type
        self.color = type+1
        self.rotation = 0
        self.kickdata = wallkickdata()
        self.ps = pieces
        self.temp_field = []

    def image(self):
        return self.figures[self.type][self.rotation]
    
    def prinPiece(self):
        print(self.x)
        print(self.y)
        print(self.ps[self.type][self.rotation])
        
    def rotate(self, rot, field, height, width):
        #print(rot, height, width)
        #self.prinPiece()
        #self.rotation = (self.rotation + rot) % len(self.figures[self.type])
        
        self.temp_field = field
        """for i in range(len(self.image())):
            for j in range(len(self.image())):
                if self.image()[i][j] != 0:
                    self.temp_field[i + self.y][j + self.x] = self.color"""
                    
        self.removePiece(self.type, self.rotation, self.x, self.y)
        if rot > 0:
            i = 0
            index = self.rotation
        else:
            i = 1
            index = self.rotation + rot
        print(index)
        if self.type == 0:
            for kick_elm in self.kickdata.i[index][i]:
                if self.placePiece(self.type, (self.rotation + rot) % 4, self.x + kick_elm[0], self.y + kick_elm[1], self.temp_field, height, width):
                    
                    #self.place(self.type, (self.rotation + rot) % 4, self.x + kick_elm[0], self.y + kick_elm[1], self.temp_field, height, width)
                    self.rotation = (self.rotation + rot) % 4
                    self.x += kick_elm[0]
                    self.y += kick_elm[1]
                    print("roto I")
                    return True
                else:
                    pass
            return False
        else:
            for kick_elm in self.kickdata.therest[index][i]:
                if self.placePiece(self.type, (self.rotation + rot) % 4, self.x + kick_elm[0], self.y + kick_elm[1], self.temp_field, height, width):
                    #self.place(self.type, (self.rotation + rot) % 4, self.x + kick_elm[0], self.y + kick_elm[1], self.temp_field, height, width)
                    self.rotation = (self.rotation + rot) % 4
                    self.x += kick_elm[0]
                    self.y += kick_elm[1]
                    print("roto no I")
                    return True
                else:
                    pass
            return False

    def placePiece(self, n, r, x, y, field, height, width):
        piece = self.ps[n][r]
        return self.check(piece, x, y, field, height, width)

    def removePiece(self, n, r, x, y):
        p = self.ps[n][r]
        l = len(p)
        l2 = len(p[0])
        for ys in range(y, y + l):
            for xs in range(x, x + l2):
                if p[ys - y][xs - x] != 0:
                    self.temp_field[ys][xs] = 0
                    
    def check(self, p, x, y, field, height, width):
        h = len(p)
        w = len(p[0])
        for ys in range(y, y + h):
            for xs in range(x, x + w):
                p1 = ys - y
                p2 = xs - x
                if (p[p1][p2] != 0) and (ys < height and ys > 0 and xs < width and xs > 0) and self.temp_field[ys][xs] == 0:
                    pass
                elif p[p1][p2] != 0:
                    return False
                else:
                    pass
        return True
    