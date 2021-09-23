# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:11:28 2021
Mancala RL agent
trained with SARSA
@author: stephen
"""

import MancalaAgent as magent
import numpy as np
import protobuf.TabularRLData_pb2 as rlProto
import os
import random


#format of the Qspace is empty, own, pool, opp, opp pool

class RLAgent(magent.MancalaAgent):
  def __init__(self, save_path, npits, nseeds, epsilon):
    super().__init__()
    self.epsilon = epsilon
    self.nseeds = nseeds
    self.save_path = save_path
    self.lr = 0.001
    self.gamma = 0.99
    self.set_npits(npits)
    self.move_history = []

  def save(self):
    outBuffer = rlProto.QAgent()
    outBuffer.npits = self.npits
    outBuffer.Q[:] = self.Qspace.reshape(-1).tolist()
    outBuffer.nseeds = self.nseeds
    
    with open(self.save_path, "wb") as f:
      f.write(outBuffer.SerializeToString())

  def load(self):
    if os.path.exists(self.save_path):
      with open(self.save_path, "rb") as f:
        inBuffer = rlProto.QAgent()
        inBuffer.ParseFromString(f.read())
        self.npits = inBuffer.npits
        self.pit_state_count = self.npits*2 + 2
        self.nseeds = inBuffer.nseeds
        self.nseeds_total = self.nseeds*self.npits*2
        self.Qspace = np.array(inBuffer.Q)
        self.Qspace = self.Qspace.reshape([-1, self.npits])

  def set_npits(self, npits):
    self.npits = npits
    self.nseeds_total = self.nseeds*npits*2
    self.pit_state_count = self.npits*2 + 2
    # all zeros bar the final pit is the last entry
    self.Qspace = np.ones([self.get_board_index([0]*self.pit_state_count) + 1, self.npits])
    
    
  #data is stored according to a tree starting with own pot left totally full  
  def get_board_index(self, state : [int]) -> int:
    index = 0
    remSeeds = self.nseeds_total
    for i in range(self.pit_state_count -2, -1, -1):
        remSeeds = remSeeds - state[i]
        maxAvoid = 1
        for j in range(i):
            maxAvoid = (maxAvoid*(remSeeds+j))/(j+1)
        index += maxAvoid
    return int(index)


  def decide(self, own_pots: [int], opp_pots: [int], own_pool : int) -> int:
    #get the right part of the (absurdly large) table
    #this large table is not efficient but it allows us to see the value of GPU bound work maybe
    index = self.get_board_index(own_pots + [own_pool] +  opp_pots)
    q_space = self.Qspace[index]
    move = [index]
      
    #current q space is the npits options
    rval = -1
    
    #handle the random choice
    if random.random() < self.epsilon:
      while True:
        rval = random.randrange(0,len(own_pots))
        if own_pots[rval] > 0:
          break
    else:        
      rq = -10000000 #very low value we shouldn't hit
      for i in range(self.npits):
        if own_pots[i] > 0 and q_space[i] > rq:
          rq = q_space[i]
          rval = i
    
    #record the action for the later training
    move.append(rval)
    #when to train - how do we get this to not collide while parallel?
    self.move_history.append(move)
    
    return rval
  
  def train(self, result : bool, learning_rate : float):
    #update on the final action (which has a reward but optimal Q is assumed to be 0
    indexer = tuple(self.move_history[-1])
    self.Qspace[indexer] = self.Qspace[indexer] +\
      learning_rate * ((1 if result else 0) -self.Qspace[indexer])
      
    #update the other states
    for i in range(1, len(self.move_history)):
      curent_ind = tuple(self.move_history[-i-1])
      next_ind = tuple(self.move_history[-i])
      self.Qspace[curent_ind] = self.Qspace[curent_ind] + \
        learning_rate * (self.gamma*self.Qspace[next_ind] -self.Qspace[curent_ind])

    self.move_history.clear()
      