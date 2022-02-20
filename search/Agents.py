'''
Agents.py
--------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
----
This code was reorganized by Beck Addison as part of an attempt to refactor the code.
Imports were cleaned up and algorithms were improved for speed.
'''

from dataclasses import dataclass
from typing import Protocol, Tuple
from enum import Enum
import numpy as np


from game import Actions
from search.game import AgentState, Direction
from test import DIRECTION
from util import manhattanDistance
import util

@dataclass
class Agent(Protocol):
    """
    Agent
    ---
    The `Agent` class is an abstract protocol for any agent used in the game.
    
    An `Agent` is any object that exists on the GameBoard. At its most basic level,
    an `Agent` needs to have a position and map representation.

    It contains the state for the Agent at any given time, as well as key methods.

    Importantly, the `Agent` differentiates itself from the 
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    """
    position: Tuple(int, int)

    def __map_repr(self) -> str:
        """
        This method should return the single-char representation of the object.
        If no unicode representation is used, then the method returns a hash of the object.
        """
        ...

@dataclass
class MobileAgent:
    """
    MobileAgent
    ---
    The `MobileAgent` is an abstract class that implements the `Agent` Protocol, and 
    extends it to include direction and movement via the `getAction` method.
    """
    
    position: Tuple(int, int)
    direction: Direction = DIRECTION.STOP

    def getAction(self) -> Direction:
        ...
    
    def __map_repr(self) -> str:
        return chr(self.__hash__())

    @property
    def left(self) -> Direction:
        return self._rotate(self.direction, theta=np.deg2rad(90))

    @property
    def right(self) -> Direction:
        return self._rotate(self.direction, theta=np.deg2rad(-90))

@dataclass
class Pacman(MobileAgent):
    

    def getAction(self) -> Direction:
        ...

    def __map_repr() -> str:
        return '😙'

@dataclass
class Ghost(MobileAgent):
    """
    Ghost
    ---

    An instance of the `Agent` [Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol) defining a Ghost character.

    Contains all the state information for the Ghost.
        
    """
    
    scared_timer: int = 0

    @property
    def is_scared(self) -> bool:
        return self.scared_timer != 0
    
    def getAction(self):
        ...
    
    def __map_repr(self):
        if self.is_scared:
            return '🪦' # lol, this isn't the best way to represent a scared ghost but oh well
        else:
            return '👻'

class StaticAgent:
    
    position: Tuple(int, int)
    
    def __map_repr(self) -> str:
        ...

class Food(StaticAgent):
    
    def __map_repr(self) -> str:
        return '▫'

class Capsule(StaticAgent):
    
    def __map_repr(self) -> str:
        return '◽'

#### OLD CODE ###### NEED TO ENSURE THIS CAN BE DELETED

class GhostAgent( Agent ):
    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution( dist )

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()

class RandomGhost( GhostAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist

class DirectionalGhost( GhostAgent ):
    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution( self, state ):
        # Read variables from state
        ghostState = state.getGhostState( self.index )
        legalActions = state.getLegalActions( self.index )
        pos = state.getGhostPosition( self.index )
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
        if isScared:
            bestScore = max( distancesToPacman )
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min( distancesToPacman )
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
        dist.normalize()
        return dist