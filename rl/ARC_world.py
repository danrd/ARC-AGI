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

BUILD_ZONE_SIZE_X = 30
BUILD_ZONE_SIZE_Y = 30
BUILD_ZONE_SIZE = 30, 30

class World:
    def __init__(self):
        self.world = {}
        self.placed = set()
        self.build_zone = (30, 30)
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
        for x in range(30):
            for y in range(30):
                self.add_block((x, y), 1)
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
        if (x, y) not in self.forbidden_cells:
            agent.position = (x,y)
            agent.encoded_position = agent.encode_position()
        else:
            return

    def parse_action(self, action):
        # 0 noop; 1 left; 2 right; 3 up; 4 down; 5 place black block 6 place blue block
        # 7 place red block 8 place green block 9 place yellow block 10 place gray block
        # 11 place magenta block 12 place orange block 13 place sky block 14 place brown block
        strafe = [0, 0]
        add = -1
        if action == 1:
            strafe[0] += -1
        elif action == 2:
            strafe[0] += 1
        elif action == 3:
            strafe[1] += 1
        elif action == 4:
            strafe[1] += -1
        elif action == 5:
            add = BLACK
        elif action == 6:
            add = BLUE
        elif action == 7:
            add = RED
        elif action == 8:
            add = GREEN
        elif action == 9:
            add = YELLOW
        elif action == 10:
            add = GRAY
        elif action == 11:
            add = MAGENTA
        elif action == 12:
            add = ORANGE
        elif action == 13:
            add = SKY
        elif action == 14:
            add = BROWN
        return strafe, add

    def place_block(self, agent, color:int):
        if color in range(10):
            self.add_block(agent.position, color)
    
    def step(self, agent, action):
        strafe, add = self.parse_action(action)
        self.movement(agent, strafe=strafe)
        self.place_block(agent, color=add)
        
class Agent:
    def __init__(self, position:tuple=(14,14)) -> None:
        self.position = position
        self.encoded_position = self.encode_position()
    
    def encode_position(self):
        grid = np.zeros((30,30))
        position = self.position
        grid[position] = 1
        return grid