# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:23:05 2021
  Mancala play environment
@author: stephen
"""


#actions are an int indicating which pot to select
#player is an int - 0 is first player, 1 is second

class MancalaEnv():
  #npots is then number of pots per player
  #nbeads is the starting number of beads per pot
  def __init__(self, npots : int, nbeads : int) -> None:
    self._round_running = False
    self._nbeads = nbeads
    self._npots = npots
    self._pots = [0 for i in range(npots*2 + 2)] 
    self._turn_record = []
  
  def start_round(self) -> None: 
    self._turn_record = []
    for i in range(len(self._pots)):
      self._pots[i] = self._nbeads if (i + 1) % (self._npots + 1) != 0 else 0
    self._round_running = True
  
  def is_game_over(self) -> bool:
    return self._round_running
  
  def get_n_pots(self) -> int:
    return self._npots
  
  #returns if you get another turn or not
  def take_turn(self, action: int, player: int) -> bool:
    assert(action < self._npots)
    assert(action >= 0)
    assert(player < 2 and player >= 0)
    self._turn_record.append((action, player))
    rval = False
    start_index = player*(self._npots +1) + action
    pool_index = (player + 1) *(self._npots + 1) - 1
    beads = self._pots[start_index]
    self._pots[start_index] = 0
    index = start_index
    for i in range(beads):
      index += 1
      if index >= len(self._pots):
        index = 0
      self._pots[index] += 1
    #handle the bonus turn
    if index == pool_index:
      rval = True
    
    #handle hitting an empty pot
    if self._pots[index] == 1 and \
        index >= player*(self._npots + 1) and \
         index < pool_index:
      opp_index = (index + self._npots + 1) % len(self._pots)
      self._pots[pool_index] += self._pots[opp_index]
      self._pots[opp_index] = 0
      self._pots[index] += self._pots[index]
      self._pots[index] = 0
    
    #check if the game is over
    player1empty = True
    player2empty = True
    for i in range(0, self._npots):
      if i > 0:
        player1empty = False
        break;
    
    for i in range(self._npots + 1, self.npots*2 + 1):
      if i > 0:
        player2empty = False
        break;
    
    if player1empty or player2empty:
      self._round_running = False
    
    return rval
    
  def get_pots(self, player: int) -> list(int):
    return [self._pots[player*(self._npots +1) + i] for i in range(self._npots)]
  
  def get_pool(self, player: int) -> int:
           return self._pots[(player + 1) *(self._npots + 1) - 1]
  
  def get_score(self, player: int) -> int:
    score = 0
    for i in range(player*(self._npots +1), (player + 1) * (self._npots + 1)):
      score += self._pots[i]
      
    return score
  

