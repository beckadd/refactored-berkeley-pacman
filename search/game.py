# game.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# game.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from util import *
import time, os
import traceback
import sys

#######################
# Parts worth reading #
#######################
class DIRECTION(Enum):
    '''
    The `DIRECTION` class defines the cardinal directions for game board. It is called as
    an Enum, e.g. `North` is `DIRECTION.NORTH`, `East` is `DIRECTION.EAST`, etc.


    The cardinal directions are encoded as follows:
    ```
                     NORTH [0, 1]
                         |
                         |
    WEST [-1, 0] ---  STOP [0, 0] --- EAST [1, 0]
                         |
                         |
                     SOUTH [0,-1]
    ```
    There is an ample opportunity to add more than two dimensions to this class.
    However, it's important to note that the rotate method is 
    '''
    NORTH = [0, 1]
    SOUTH = [0, -1]
    EAST = [1, 0]
    WEST = [-1, 0]
    STOP = [0, 0]


class Direction:
    """
    Direction
    ---

    This class is used to define the direction of a game `Agent`. It
    comes with a few useful classes to provide relative directions (such
    as `left` or `right`) to make it easier to make actions.

    The class comes with the "private" attribute `_current_direction`. 
    This attribute should not be accessed directly - accessing it directly
    will make any resultant program harder to troubleshoot and possibly 
    broken.

    Instead, access it through `Direction.current`.
    """

    __current_direction: np.ndarray

    def __init__(self, direction: DIRECTION):
        self.__current_direction = np.array(direction.value)

    @property
    def current(self):
        return DIRECTION(list(self._current_direction))


    def __rotate(self, x: np.ndarray, theta: float) -> np.ndarray:
        '''
        Rotate
        ---

        Private method for rotating a given direction. Currently,
        this is implemented in 2D, given that PacMan is played in 2D.

        Rotates a directional array a given amount in radians.

        :x: the ndarray to rotate. This is (currently) constricted to 2-element arrays.

        :theta: the amount (in radians) to rotate the vector by. Positive values rotate the vector counterclockwise, whereas negative values rotate the vector clockwise.
        '''
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]
            ]
        ) // 1
        return rotation_matrix @ x

    @property
    def left(self) -> Direction:
        rot = self.__rotate(self.__current_direction, theta=np.deg2rad(90))
        return Direction(DIRECTION(list(rot)))

    @property
    def right(self) -> Direction:
        rot = self.__rotate(self.__current_direction, theta=np.deg2rad(-90))
        return Direction(DIRECTION(list(rot)))
    
    def __str__(self) -> str:
        return str(DIRECTION(list(self.__current_direction)))


@dataclass
class Grid:
    """
    A 2-dimensional array of objects backed by a list of lists.  Data is accessed
    via grid[x][y] where (x,y) are positions on a Pacman map with x horizontal,
    y vertical and the origin (0,0) in the bottom left corner.

    The __str__ method constructs an output that is oriented like a pacman board.
    """
    _grid: np.array

    @property
    def width(self) -> int:
        self._grid.shape[0]

    def height(self) -> int:
        self._grid.shape[1]

    def 

    def __init__(self, width, height, initialValue=False, bitRepresentation=None):
        if initialValue not in [False, True]: raise Exception('Grids can only contain booleans')
        self.CELLS_PER_INT = 30

        self.width = width
        self.height = height
        self.data = [[initialValue for y in range(height)] for x in range(width)]
        if bitRepresentation:
            self._unpackBits(bitRepresentation)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __str__(self):
        out = [[str(self.data[x][y])[0] for x in range(self.width)] for y in range(self.height)]
        out.reverse()
        return '\n'.join([''.join(x) for x in out])

    def __eq__(self, other):
        if other == None: return False
        return self.data == other.data

    def __hash__(self):
        # return hash(str(self))
        base = 1
        h = 0
        for l in self.data:
            for i in l:
                if i:
                    h += base
                base *= 2
        return hash(h)

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def count(self, item =True ):
        return sum([x.count(item) for x in self.data])

    def asList(self, key = True):
        list = []
        for x in range(self.width):
            for y in range(self.height):
                if self[x][y] == key: list.append( (x,y) )
        return list

    def packBits(self):
        """
        Returns an efficient int list representation

        (width, height, bitPackedInts...)
        """
        bits = [self.width, self.height]
        currentInt = 0
        for i in range(self.height * self.width):
            bit = self.CELLS_PER_INT - (i % self.CELLS_PER_INT) - 1
            x, y = self._cellIndexToPosition(i)
            if self[x][y]:
                currentInt += 2 ** bit
            if (i + 1) % self.CELLS_PER_INT == 0:
                bits.append(currentInt)
                currentInt = 0
        bits.append(currentInt)
        return tuple(bits)

    def _cellIndexToPosition(self, index):
        x = index // self.height
        y = index % self.height
        return x, y

    def _unpackBits(self, bits):
        """
        Fills in data from a bit-level representation
        """
        cell = 0
        for packed in bits:
            for bit in self._unpackInt(packed, self.CELLS_PER_INT):
                if cell == self.width * self.height: break
                x, y = self._cellIndexToPosition(cell)
                self[x][y] = bit
                cell += 1

    def _unpackInt(self, packed, size):
        bools = []
        if packed < 0: raise ValueError("must be a positive integer")
        for i in range(size):
            n = 2 ** (self.CELLS_PER_INT - i - 1)
            if packed >= n:
                bools.append(True)
                packed -= n
            else:
                bools.append(False)
        return bools

def reconstituteGrid(bitRep):
    if type(bitRep) is not type((1,2)):
        return bitRep
    width, height = bitRep[:2]
    return Grid(width, height, bitRepresentation= bitRep[2:])

####################################
# Parts you shouldn't have to read #
####################################

class Direction(Enum):
    """
    Direction
    ---

    ### For Students:

    The `Direction` class defines the cardinal directions for game board. It is called as
    an Enum, e.g. `North` is `Direction.NORTH`, `East` is `Direction.EAST`, etc.

    The `Direction` class encodes absolute orientation on the board. An `Agent` keeps track of its
    own orientation with respect to the `GameBoard`.   

    ### For developers:

    `Agents` use the direction class to determine the direction of an `AgentAction`. 

    The cardinal directions are encoded as follows:
    ```
                     NORTH [0, 1]
                         |
                         |
    WEST [-1, 0] ---  STOP [0, 0] --- EAST [1, 0]
                         |
                         |
                     SOUTH [0,-1]
    ```
    
    There is an ample opportunity to add more than two dimensions to this class. simply increase the dimensionality of the arrays.
    """

    NORTH = (0, 1)
    SOUTH = (0, -1)
    EAST = (1, 0)
    WEST = (-1, 0)
    STOP = (0, 0)

class Path:
    '''
    Path
    ---
    The `Path` class is simply a list of directions that can be taken by
    an Agent. It provides convenient display functions for troubleshooting code
    with a text output.
    '''
    
    ### Base game board representation:

    __EMPTY_SPACE_BLOCK = '⬛'
    __TRAVELED_SPACE_BLOCK = '🔹' 
    __CURRENT_SPACE_BLOCK = ''

    


    def __str__():
        """
        Generate a simple path object for this class.
        """
        ...


    def pprint(gameBoard):
        """
        `pprint()` calls the `__map_repr()` method of every `Agent` on the `GameBoard`.
        If no `__map_repr()` is given, then `Path` will randomly select a representative
        emoji based on a hash of the object.
        """        
        ...

class Actions:
    """
    A collection of static methods for manipulating move actions.
    """
    # # Directions
    # _directions = {Directions.NORTH: (0, 1),
    #                Directions.SOUTH: (0, -1),
    #                Directions.EAST:  (1, 0),
    #                Directions.WEST:  (-1, 0),
    #                Directions.STOP:  (0, 0)}

    # _directionsAsList = _directions.items()

    # TOLERANCE = .001

    # def reverseDirection(action):
    #     if action == Directions.NORTH:
    #         return Directions.SOUTH
    #     if action == Directions.SOUTH:
    #         return Directions.NORTH
    #     if action == Directions.EAST:
    #         return Directions.WEST
    #     if action == Directions.WEST:
    #         return Directions.EAST
    #     return action
    # reverseDirection = staticmethod(reverseDirection)

    def vectorToDirection(vector):
        dx, dy = vector
        if dy > 0:
            return Directions.NORTH
        if dy < 0:
            return Directions.SOUTH
        if dx < 0:
            return Directions.WEST
        if dx > 0:
            return Directions.EAST
        return Directions.STOP
    vectorToDirection = staticmethod(vectorToDirection)

    def directionToVector(direction, speed = 1.0):
        dx, dy =  Actions._directions[direction]
        return (dx * speed, dy * speed)
    directionToVector = staticmethod(directionToVector)

    def getPossibleActions(config, walls):
        possible = []
        x, y = config.pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # In between grid points, all agents must continue straight
        if (abs(x - x_int) + abs(y - y_int)  > Actions.TOLERANCE):
            return [config.getDirection()]

        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if not walls[next_x][next_y]: possible.append(dir)

        return possible

    getPossibleActions = staticmethod(getPossibleActions)

    def getLegalNeighbors(position, walls):
        x,y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        neighbors = []
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            if next_x < 0 or next_x == walls.width: continue
            next_y = y_int + dy
            if next_y < 0 or next_y == walls.height: continue
            if not walls[next_x][next_y]: neighbors.append((next_x, next_y))
        return neighbors
    getLegalNeighbors = staticmethod(getLegalNeighbors)

    def getSuccessor(position, action):
        dx, dy = Actions.directionToVector(action)
        x, y = position
        return (x + dx, y + dy)
    getSuccessor = staticmethod(getSuccessor)

class GameStateData:
    """

    """
    def __init__( self, prevState = None ):
        """
        Generates a new data packet by copying information from its predecessor.
        """
        if prevState != None:
            self.food = prevState.food.shallowCopy()
            self.capsules = prevState.capsules[:]
            self.agentStates = self.copyAgentStates( prevState.agentStates )
            self.layout = prevState.layout
            self._eaten = prevState._eaten
            self.score = prevState.score

        self._foodEaten = None
        self._foodAdded = None
        self._capsuleEaten = None
        self._agentMoved = None
        self._lose = False
        self._win = False
        self.scoreChange = 0

    def deepCopy( self ):
        state = GameStateData( self )
        state.food = self.food.deepCopy()
        state.layout = self.layout.deepCopy()
        state._agentMoved = self._agentMoved
        state._foodEaten = self._foodEaten
        state._foodAdded = self._foodAdded
        state._capsuleEaten = self._capsuleEaten
        return state

    def copyAgentStates( self, agentStates ):
        copiedStates = []
        for agentState in agentStates:
            copiedStates.append( agentState.copy() )
        return copiedStates

    def __eq__( self, other ):
        """
        Allows two states to be compared.
        """
        if other == None: return False
        # TODO Check for type of other
        if not self.agentStates == other.agentStates: return False
        if not self.food == other.food: return False
        if not self.capsules == other.capsules: return False
        if not self.score == other.score: return False
        return True

    def __hash__( self ):
        """
        Allows states to be keys of dictionaries.
        """
        for i, state in enumerate( self.agentStates ):
            try:
                int(hash(state))
            except TypeError as e:
                print(e)
                #hash(state)
        return int((hash(tuple(self.agentStates)) + 13*hash(self.food) + 113* hash(tuple(self.capsules)) + 7 * hash(self.score)) % 1048575 )

    def __str__( self ):
        width, height = self.layout.width, self.layout.height
        map = Grid(width, height)
        if type(self.food) == type((1,2)):
            self.food = reconstituteGrid(self.food)
        for x in range(width):
            for y in range(height):
                food, walls = self.food, self.layout.walls
                map[x][y] = self._foodWallStr(food[x][y], walls[x][y])

        for agentState in self.agentStates:
            if agentState == None: continue
            if agentState.configuration == None: continue
            x,y = [int( i ) for i in nearestPoint( agentState.configuration.pos )]
            agent_dir = agentState.configuration.direction
            if agentState.isPacman:
                map[x][y] = self._pacStr( agent_dir )
            else:
                map[x][y] = self._ghostStr( agent_dir )

        for x, y in self.capsules:
            map[x][y] = 'o'

        return str(map) + ("\nScore: %d\n" % self.score)

    def _foodWallStr( self, hasFood, hasWall ):
        if hasFood:
            return '.'
        elif hasWall:
            return '%'
        else:
            return ' '

    def _pacStr( self, dir ):
        if dir == Directions.NORTH:
            return 'v'
        if dir == Directions.SOUTH:
            return '^'
        if dir == Directions.WEST:
            return '>'
        return '<'

    def _ghostStr( self, dir ):
        return 'G'
        if dir == Directions.NORTH:
            return 'M'
        if dir == Directions.SOUTH:
            return 'W'
        if dir == Directions.WEST:
            return '3'
        return 'E'

    def initialize( self, layout, numGhostAgents ):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.food = layout.food.copy()
        #self.capsules = []
        self.capsules = layout.capsules[:]
        self.layout = layout
        self.score = 0
        self.scoreChange = 0

        self.agentStates = []
        numGhosts = 0
        for isPacman, pos in layout.agentPositions:
            if not isPacman:
                if numGhosts == numGhostAgents: continue # Max ghosts reached already
                else: numGhosts += 1
            self.agentStates.append( AgentState( Configuration( pos, Directions.STOP), isPacman) )
        self._eaten = [False for a in self.agentStates]

class GameObject(Enum):
    WALL = auto()
    PATHWAY = auto()
    NEWLINE = auto()
    FOOD = auto()
    CAPSULE = auto()
    PACMAN = auto()
    GHOST = auto()

@dataclass
class GameBoard:
    """
    GameBoard
    ---
    The `GameBoard` holds all information related to the location of objects on the board,
    as well as the shape of the grid.
    """

    __grid: np.ndarray

    @property
    def walls(self):
        return self.__grid == GameObject.WALL

    @property
    def ghosts(self):
        return self.__grid == GameObject.GHOST

    @property
    def pacman(self):
        return self.__grid == GameObject.PACMAN

    @property
    def food(self):
        return self.__grid == GameObject.FOOD

    @property
    def capsules(self):
        return self.__grid == GameObject.CAPSULE

    # def __str__(self): ...
    # TODO needs to represent a pretty little command line pacman game

    # importing functions

    def from_layout(path: str, **fmt_options) -> GameBoard:
        """
        Generates a gameboard from a `.lay` file. This is the layout system that
        the original Berkey Pacman project used, and acts as a way to bridge the
        refactored project with older projects that may have used the old layout system.

        :f: The '.layout' file to be converted to the gameboard.

        :fmt_options: A dictionary of options describing the characters used to format the `.lay` file. 
        By default, '%' represents a wall, a space represents a pathway, and a newline character represents
        the border of the walls.
        """
        fmt = {
            GameObject.WALL: "%",
            GameObject.PATHWAY: " ",
            GameObject.NEWLINE: "\n",
            GameObject.FOOD: ".",
            GameObject.CAPSULE: "o",
            GameObject.PACMAN: "P",
            GameObject.GHOST: "G",
        }
        fmt.update(fmt_options)

        # OPTIMIZE: need to figure out a numpy native way to convert this string
        with open(path) as f:
            array = [list(row.strip()) for row in f.readlines()]
            raw = np.array(array)
        grid = GameBoard.__translate(raw, fmt)
        return GameBoard(grid)

    def from_csv(path: str, delimiter=",", **fmt_options) -> GameBoard:
        fmt = {enum: enum.value for enum in GameObject}
        fmt.update(fmt_options)
        raw = np.loadtxt(path, delimiter)
        grid = GameBoard.__translate(raw, fmt)
        return GameBoard(grid)

    def __translate(raw: np.ndarray, fmt) -> np.ndarray:
        """
        helper method to translate an input grid
        using provided format.
        """
        # nasty way of doing this...
        # going through each of the enums and checking where
        grid = np.zeros_like(raw, dtype="object")
        for key, val in fmt.items():
            grid[raw == val] = key
        return grid

    # exporting functions

    def to_json(writepath: str) -> None:
        ...

    def to_csv(writepath: str) -> None:
        ...

    def to_txt(writepath: str) -> None:
        ...


class GameStatus(Enum):
    WIN = auto(),
    PLAYING = auto(),
    LOSS = auto(),
    CRASHED = auto()
    

@dataclass
class Game:
    """
    Game
    ---
    The `Game` object is the host of the Pacman game. It handles the main gameplay loop, executing actions with every frame.
    The Game manages the control flow, soliciting actions from agents.
    The `Game` is implemented with a command pattern, where commands are dispatched by `Agent`s and
    received by the `GameBoard`. delta encoded, meaning that, similar to a git repository, the latest
    version of the board is kept in its entirety, but only the changes that led up to that
    most recent state are stored apart from that final state. This way, the original state 
    of the board can be determined by "walking" the states backwards in time. 
    """

    board: GameBoard
    status: GameStatus

    def __init__( self, agents, display, rules, startingIndex=0, muteAgents=False, catchExceptions=False ):
        self.agentCrashed = False
        self.agents = agents
        self.display = display
        self.rules = rules
        self.startingIndex = startingIndex
        self.gameOver = False
        self.muteAgents = muteAgents
        self.catchExceptions = catchExceptions
        self.moveHistory = []
        self.totalAgentTimes = [0 for agent in agents]
        self.totalAgentTimeWarnings = [0 for agent in agents]
        self.agentTimeout = False
        import io
        self.agentOutput = [io.StringIO() for agent in agents]

    def getProgress(self):
        if self.gameOver:
            return 1.0
        else:
            return self.rules.getProgress(self)

    def _agentCrash( self, agentIndex, quiet=False):
        "Helper method for handling agent crashes"
        if not quiet: traceback.print_exc()
        self.gameOver = True
        self.agentCrashed = True
        self.rules.agentCrash(self, agentIndex)

    OLD_STDOUT = None
    OLD_STDERR = None

    def mute(self, agentIndex):
        if not self.muteAgents: return
        global OLD_STDOUT, OLD_STDERR
        import io
        OLD_STDOUT = sys.stdout
        OLD_STDERR = sys.stderr
        sys.stdout = self.agentOutput[agentIndex]
        sys.stderr = self.agentOutput[agentIndex]

    def unmute(self):
        if not self.muteAgents: return
        global OLD_STDOUT, OLD_STDERR
        # Revert stdout/stderr to originals
        sys.stdout = OLD_STDOUT
        sys.stderr = OLD_STDERR


    def run( self ):
        # FIXME: This is such a disgusting function, it's not even funny.

        """
        Main control loop for game play.
        """
        self.display.initialize(self.state.data)
        self.numMoves = 0

        ###self.display.initialize(self.state.makeObservation(1).data)
        # inform learning agents of the game start
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if not agent:
                self.mute(i)
                # this is a null agent, meaning it failed to load
                # the other team wins
                print("Agent %d failed to load" % i, file=sys.stderr)
                self.unmute()
                self._agentCrash(i, quiet=True)
                return
            if ("registerInitialState" in dir(agent)):
                self.mute(i)
                if self.catchExceptions:
                    try:
                        timed_func = TimeoutFunction(agent.registerInitialState, int(self.rules.getMaxStartupTime(i)))
                        try:
                            start_time = time.time()
                            timed_func(self.state.deepCopy())
                            time_taken = time.time() - start_time
                            self.totalAgentTimes[i] += time_taken
                        except TimeoutFunctionException:
                            print("Agent %d ran out of time on startup!" % i, file=sys.stderr)
                            self.unmute()
                            self.agentTimeout = True
                            self._agentCrash(i, quiet=True)
                            return
                    except Exception as data:
                        self._agentCrash(i, quiet=False)
                        self.unmute()
                        return
                else:
                    agent.registerInitialState(self.state.deepCopy())
                ## TODO: could this exceed the total time
                self.unmute()

        agentIndex = self.startingIndex
        numAgents = len( self.agents )

        while not self.gameOver:
            # Fetch the next agent
            agent = self.agents[agentIndex]
            move_time = 0
            skip_action = False
            # Generate an observation of the state
            if 'observationFunction' in dir( agent ):
                self.mute(agentIndex)
                if self.catchExceptions:
                    try:
                        timed_func = TimeoutFunction(agent.observationFunction, int(self.rules.getMoveTimeout(agentIndex)))
                        try:
                            start_time = time.time()
                            observation = timed_func(self.state.deepCopy())
                        except TimeoutFunctionException:
                            skip_action = True
                        move_time += time.time() - start_time
                        self.unmute()
                    except Exception as data:
                        self._agentCrash(agentIndex, quiet=False)
                        self.unmute()
                        return
                else:
                    observation = agent.observationFunction(self.state.deepCopy())
                self.unmute()
            else:
                observation = self.state.deepCopy()

            # Solicit an action
            action = None
            self.mute(agentIndex)
            if self.catchExceptions:
                try:
                    timed_func = TimeoutFunction(agent.getAction, int(self.rules.getMoveTimeout(agentIndex)) - int(move_time))
                    try:
                        start_time = time.time()
                        if skip_action:
                            raise TimeoutFunctionException()
                        action = timed_func( observation )
                    except TimeoutFunctionException:
                        print("Agent %d timed out on a single move!" % agentIndex, file=sys.stderr)
                        self.agentTimeout = True
                        self._agentCrash(agentIndex, quiet=True)
                        self.unmute()
                        return

                    move_time += time.time() - start_time

                    if move_time > self.rules.getMoveWarningTime(agentIndex):
                        self.totalAgentTimeWarnings[agentIndex] += 1
                        print("Agent %d took too long to make a move! This is warning %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex]), file=sys.stderr)
                        if self.totalAgentTimeWarnings[agentIndex] > self.rules.getMaxTimeWarnings(agentIndex):
                            print("Agent %d exceeded the maximum number of warnings: %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex]), file=sys.stderr)
                            self.agentTimeout = True
                            self._agentCrash(agentIndex, quiet=True)
                            self.unmute()
                            return

                    self.totalAgentTimes[agentIndex] += move_time
                    #print("Agent: %d, time: %f, total: %f" % (agentIndex, move_time, self.totalAgentTimes[agentIndex]))
                    if self.totalAgentTimes[agentIndex] > self.rules.getMaxTotalTime(agentIndex):
                        print("Agent %d ran out of time! (time: %1.2f)" % (agentIndex, self.totalAgentTimes[agentIndex]), file=sys.stderr)
                        self.agentTimeout = True
                        self._agentCrash(agentIndex, quiet=True)
                        self.unmute()
                        return
                    self.unmute()
                except Exception as data:
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
            else:
                action = agent.getAction(observation)
            self.unmute()

            # Execute the action
            self.moveHistory.append( (agentIndex, action) )
            if self.catchExceptions:
                try:
                    self.state = self.state.generateSuccessor( agentIndex, action )
                except Exception as data:
                    self.mute(agentIndex)
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
            else:
                self.state = self.state.generateSuccessor( agentIndex, action )

            # Change the display
            self.display.update( self.state.data )
            ###idx = agentIndex - agentIndex % 2 + 1
            ###self.display.update( self.state.makeObservation(idx).data )

            # Allow for game specific conditions (winning, losing, etc.)
            self.rules.process(self.state, self)
            # Track progress
            if agentIndex == numAgents + 1: self.numMoves += 1
            # Next agent
            agentIndex = ( agentIndex + 1 ) % numAgents

            if _BOINC_ENABLED:
                boinc.set_fraction_done(self.getProgress())

        # inform a learning agent of the game result
        for agentIndex, agent in enumerate(self.agents):
            if "final" in dir( agent ) :
                try:
                    self.mute(agentIndex)
                    agent.final( self.state )
                    self.unmute()
                except Exception as data:
                    if not self.catchExceptions: raise data
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
        self.display.finish()
