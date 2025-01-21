import numpy as np

BLACK = 0
BLUE = 0.1
RED = 0.2
GREEN = 0.3
YELLOW = 0.4
GRAY = 0.5
MAGENTA = 0.6
ORANGE = 0.7
SKY = 0.8
BROWN = 0.9
WHITE = 1

class World:
    def __init__(self, build_zone, font_color=0.0):
        self.world = {}
        self.placed = set()
        self.build_zone = build_zone
        self.font_color = font_color
        self.initialized = False

    def deinit(self):
        for block in list(self.placed):
            self.remove_block(block)
        self.initialized = False
        for block in list(self.world.keys()):
            self.remove_block(block)
        self.world = {}
        self.placed = set()
        
    def initialize(self):
        """ Initialize the world by placing all the blocks."""
        for x in range(self.build_zone[0]):
            for y in range(self.build_zone[1]):
                self.add_block((x, y), self.font_color)
        self.initialized = True

    def add_block(self, position:tuple, color:float):
        """ Add a block with the given `texture` and `position` to the world.
        Parameters
        ----------
        position : tuple of len 2
            The (x, y) position of the block to add.
        texture : int
            The color of the block.
        """
        self.world[position] = color
        if self.initialized:
            self.placed.add(position)

    def remove_block(self, position):
        """ Remove the block at the given `position`.
        Parameters
        ----------
        position : tuple of len 2
            The (x, y) position of the block to remove.
        """
        del self.world[position]
        if self.initialized:
            self.placed.remove(position)

    def movement(self, agent, strafe: list):
        x = agent.position[0] + strafe[0]
        y = agent.position[1] + strafe[1]
        if (x, y) not in self.forbidden_cells and x in range(0, self.build_zone[0]) and y in range(0, self.build_zone[1]):
            agent.position = (x, y)
            agent.encoded_position = agent.encode_position()
        else:
            return

    def parse_action(self, action):
        # 0 left; 1 right; 2 up; 3 down; 4 place black block 5 place blue block
        # 6 place red block 7 place green block 8 place yellow block 9 place gray block
        # 10 place magenta block 11 place orange block 12 place sky block 13 place brown block
        strafe = [0, 0]
        add = -1
        if action == 0:
            strafe[0] += -1
        elif action == 1:
            strafe[0] += 1
        elif action == 2:
            strafe[1] += 1
        elif action == 3:
            strafe[1] += -1
        elif action == 4:
            add = BLACK
        elif action == 5:
            add = BLUE
        elif action == 6:
            add = RED
        elif action == 7:
            add = GREEN
        elif action == 8:
            add = YELLOW
        elif action == 9:
            add = GRAY
        elif action == 10:
            add = MAGENTA
        elif action == 11:
            add = ORANGE
        elif action == 12:
            add = SKY
        elif action == 13:
            add = BROWN
        return strafe, add

    def place_block(self, agent, color:int):
        if color != -1:
            self.add_block(agent.position, color)
    
    def step(self, agent, strafe, add):
        self.movement(agent, strafe=strafe)
        self.place_block(agent, color=add)
        
class Agent:
    def __init__(self, position:tuple=(14, 14)) -> None:
        self.position = position
        self.encoded_position = self.encode_position()
    
    def encode_position(self):
        norm_pos = (self.position[0]/30, self.position[1]/30)
        return norm_pos