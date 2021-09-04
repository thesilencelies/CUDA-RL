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
  def init(self, save_path, npits, epsilon):
    super().init()
    self.epsilon = epsilon
    self.save_path = save_path
    self.lr = 0.001
    self.gamma = 0.99
    self.set_npits(npits)
    self.move_history = []

  def save(self):
    outBuffer = rlProto.QAgent()
    outBuffer.npits = self.npits
    outBuffer.Q[:] = self.Qspace.reshape(-1).tolist()
    
    with open(self.save_path, "w") as f:
      f.write(outBuffer.SerializeToString())

  def load(self):
    if os.path.exists(self.save_path):
      with open(self.save_path, "r") as f:
        inBuffer = rlProto.QAgent()
        inBuffer.ParseFromString(f.read())
        self.npits = inBuffer.npits
        self.pit_state_count = self.npits*2 + 3
        self.Qspace = np.array(inBuffer.Q)
        self.Qspace = self.Qspace.reshape([self.pit_state_count for i in range(self.npits*2)] + [self.npits])

  def set_npits(self, npits):
    self.npits = npits
    self.pit_state_count = self.npits*2 + 3 # both players pits, pools and empty
    self.Qspace = np.ones([self.pit_state_count for i in range(self.npits*2)] + [self.npits])
    

  def decide(self, own_pots: [int], opp_pots: [int]) -> int:
    #get the right part of the (absurdly large) table
    #this large table is not efficient but it allows us to see the value of GPU bound work maybe
    q_space = self.Qspace
    move = []
    for i in range(len(own_pots)):
      #work out where the point goes
      #handle the empty space case
      dest = 0 if own_pots[i] == 0 else (i + own_pots[i]) % ((self.npits + 1)*2) + 1
      q_space = q_space[dest]
      move.append(dest)
    
    for i in range(len(opp_pots)):
      dest = 0 if opp_pots[i] == 0 else (self.npits + 1 + i + opp_pots[i]) % ((self.npits + 1)*2) + 1
      q_space = q_space[dest]
      move.append(dest)
      
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
    #update on the final action (which has a reward but optimal Q is assumed to be 0)
    self.q_space[self.move_history[-1]] = self.q_space[self.move_history[-1]] + \
      learning_rate * ((1 if result else 0) -self.q_space[self.move_history[-1]])
      
    #update the other states
    for i in range(2, len(self.move_history)):
      self.q_space[self.move_history[-i-1]] = self.q_space[self.move_history[-i-1]] + \
        learning_rate * (self.q_space[self.move_history[-i]] -self.q_space[self.move_history[-i-1]])

    self.move_history.clear()
      