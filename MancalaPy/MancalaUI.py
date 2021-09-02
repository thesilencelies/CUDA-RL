# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:26:15 2021
Mancala UI handling and base run script

@author: stephen
"""

import tkinter as tk
import MancalaEnv as menv
import MancalaAgent as magent

#UI constants
pit_width = 60
margin = 10
pit_height = 80
seed_radius = 3

seed_row = 4



class Pit():
  def __init__(self, canvas : tk.Canvas, index: int, player: int):
    self.canvas = canvas
    self.x1 = (pit_width * 3)/2 + index * (pit_width + margin)
    self.y1 = (pit_height )/2 + player * (pit_height + margin)
    self.x2 = (pit_width * 5)/2 + index * (pit_width + margin)
    self.y2 = (pit_height * 3)/2 + player * (pit_height + margin)
    
    self.index = index
    
    self.box = canvas.create_rectangle( self.x1, self.y1, self.x2, self.y2,
                                        fill="grey")
    self.seeds = []
  
  def draw_seeds(self, nseeds : int):
    if nseeds < len(self.seeds):
      #delete excess
      for i in range(nseeds, len(self.seeds)):
        self.canvas.delete(self.seeds[i])
      del self.seeds[nseeds:]
    else:
      for i in range(len(self.seeds), nseeds):
        x = (((i % seed_row) * 2) + 1) * (seed_radius + 1) + self.x1
        y = (((i / seed_row) * 2) + 1) * (seed_radius + 1) + self.y1
        
        self.seeds.append(self.canvas.create_oval(x - seed_radius, y - seed_radius,
                                  x + seed_radius, y + seed_radius,
                                  fill='black'))

  def delete(self):
    for s in self.seeds:
      self.canvas.delete(s)
    self.canvas.delete(self.box)

class Pool(Pit):
  def __init__(self, canvas : tk.Canvas, player : int, npits : int):
    super().__init__(canvas, 0, player)
    if player == 0:
      # on the left
      self.x1 = pit_width/4 
      self.x2 = (pit_width * 5)/4
    else:
      #on the right
      self.x1 = (pit_width * 7)/4 + npits * (pit_width + margin)
      self.x2 = (pit_width * 11)/4 + npits * (pit_width + margin)
      
    self.y1 = (pit_height * 3)/4
    self.y2 = (pit_height * 9)/4
    canvas.coords(self.box, self.x1, self.y1, self.x2, self.y2)


class Game(tk.Frame):
    def __init__(self, master):
      super(Game, self).__init__(master)
      self.width = 610
      self.height = 400
      self.canvas = tk.Canvas(self, bg='#bdc999',
                              width=self.width,
                              height=self.height)
      self.canvas.pack()
      self.pack()
      self.env = None
      self.pits = []
      self.player1_agent = None
      self.player2_agent = None
      self.player1_turn = True
      self.score = None
        
    def create_board(self, npits : int, nseeds : int):
      self.env =  menv.MancalaEnv(npits, nseeds)
      for p in self.pits:
        p.delete()
      self.pits.clear()
      #player 1
      for i in range(npits):
        self.pits.append(Pit(self.canvas, (npits - i -1), 0))
      self.pits.append(Pool(self.canvas, 0, npits))
      #player 2
      for i in range(npits):
        self.pits.append(Pit(self.canvas, i, 1))
      self.pits.append(Pool(self.canvas, 1, npits))      
        
    def setPlayer1Agent(self, agent : magent.MancalaAgent):
      self.player1_agent = agent
        
    def setPlayer2Agent(self, agent : magent.MancalaAgent):
      self.player2_agent = agent
      
    def play_turn(self):
      if self.env.is_game_over(): 
        return
      if self.player1_turn:
        #print("player 1 turn")
        self.player1_agent.take_move()
        self.player1_turn = False
      else:
        #print("player 2 turn")
        self.player2_agent.take_move()
        self.player1_turn = True
      self.draw_seeds()
    
    def draw_seeds(self):
      seed_counts = self.env._pots
      for p, s in zip(self.pits, seed_counts):
        p.draw_seeds(s)
        
    def setup(self):      
      self.text = self.draw_text(300, 300, 'Press Space to start')
      self.canvas.focus_set()
      self.canvas.bind('<space>', lambda _: self.start_game())
      #self.canvas.bind('<space>', lambda _: self.play_turn())

    def draw_text(self, x, y, text, size='16'):
      font = ('Helvetica', size)
      return self.canvas.create_text(x, y, text=text, font=font)
    
    def start_game(self):
      if self.env is None: return
      if self.player1_agent is None: return
      if self.player2_agent is None: return
      if self.score is not None : self.canvas.delete(self.score)
      if not self.env.is_game_over(): return
      self.player1_agent.set_env(self.env)
      self.player1_agent.set_player(0)
      self.player2_agent.set_env(self.env)
      self.player2_agent.set_player(1)
      self.env.start_game()
      self.draw_seeds()
      self.game_loop()
    
    def game_loop(self):
      if self.env.is_game_over():
        player1score = self.env.get_score(0)
        player2score = self.env.get_score(1)
        print(f"game over\n score was {player1score} {player2score}")
        self.score = self.draw_text(300, 340, f"Player 1 {player1score}\nPlayer 2 {player2score}")
        return
      self.play_turn()
      self.after(250, self.game_loop)

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Mancala')
    game = Game(root)
    game.create_board(4, 4)
    player1 = magent.RandomAgent()
    player2 = magent.HeuristicAgent()
    game.setPlayer1Agent(player1)
    game.setPlayer2Agent(player2)
    game.setup()
    game.mainloop()