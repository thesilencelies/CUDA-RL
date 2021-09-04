# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:24:05 2021
Mancala agent base class and simple agents
@author: stephen
"""
import MancalaEnv as menv
import random
from abc import ABC, abstractmethod

class MancalaAgent(ABC):
  def __init__(self):
    self._env = None
    self._player = 0
    self._opp = 1
  
  
  def set_env(self, environment : menv.MancalaEnv):
    self._env = environment
  
  def set_player(self, player : int):
    self._player = player
    self._opp = (player + 1) %2
    
  @abstractmethod
  def decide(self, own_pots: [int], opp_pots: [int]) -> int:
    pass
  
  @abstractmethod
  def train(self, result : bool, learning_rate : float):
    pass
  
  @abstractmethod
  def save(self):
    pass
  
  @abstractmethod
  def load(self):
    pass
  
  def take_move(self):
    has_turn = True
    while has_turn:
      if self._env.is_game_over():
        return
      own_pots = self._env.get_pots(self._player)
      opp_pots = self._env.get_pots(self._opp)
      
      action = self.decide(own_pots, opp_pots)
      
      has_turn = self._env.take_turn(action, self._player)
      
      

#simple agent that only takes random moves
class RandomAgent(MancalaAgent):
  def decide(self, own_pots: [int], opp_pots: [int]) -> int:
    while True:
      rval = random.randrange(0,len(own_pots))
      if own_pots[rval] > 0:
        return rval
  def train(self, result : bool, learning_rate : float):
    pass
  
  def save(self):
    pass
  
  def load(self):
    pass
  
  
#agent that plays according to a simple heuristic
class HeuristicAgent(MancalaAgent):
  def decide(self, own_pots: [int], opp_pots: [int]) -> int:
    #check if any of your moves would give you another turn
    for i in range(len(own_pots)):
      if own_pots[i] + i == len(own_pots):
        return i
      
    #else return the rightmost non-empty pot
    for i in range(len(own_pots) -1, -1, -1):
      if own_pots[i] > 0:
        return i
  def train(self, result : bool, learning_rate : float):
    pass
  
  def save(self):
    pass
  
  def load(self):
    pass
    
