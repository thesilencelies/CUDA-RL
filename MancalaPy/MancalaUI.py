# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:26:15 2021
Mancala UI handling and base run script

@author: stephen
"""

import tkinter as tk
import MancalaEnv as menv
import MancalaAgent as magent
import MancalaRL as mrl

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
      self.master = master
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
      self.learning_rate = 0.001
      self.num_games = 1000
      self._game_count = 0
      self.save_period = 100
      self.accumulated_results = [0,0,0]
      self.repeated_mode = False
      self.interval = 100
        
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
    
    def _toggle_repeated_mode(self):
      self.repeated_mode =  not self.repeated_mode
      self.interval = 5 if self.repeated_mode else 100
      self.r_m_toggle.config(text = "repeat on" if self.repeated_mode else "repeat off")
    
    def setup(self):      
      self.text = self.draw_text(300, 300, 'Press Space to start')
      self.canvas.focus_set()
      self.canvas.bind('<space>', lambda _: self.start_game())
      self.r_m_toggle = tk.Button(self.master, text ="repeat off",
                                  command = lambda: self._toggle_repeated_mode())
      self.r_m_toggle.pack(pady=5)
      
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
      self._game_count = 0
      self.player1_agent.set_env(self.env)
      self.player1_agent.set_player(0)
      self.player2_agent.set_env(self.env)
      self.player2_agent.set_player(1)
      self.env.start_game()
      self.draw_seeds()
      self.game_loop()
    
    def game_loop(self):
      if self.score is not None : self.canvas.delete(self.score)
      if self.env.is_game_over():
        player1score = self.env.get_score(0)
        player2score = self.env.get_score(1)
        
        if player1score > player2score:
          result = "player 1 wins" 
          self.accumulated_results[0] += 1
        elif player2score > player1score:
          result = "player 2 wins" 
          self.accumulated_results[1] += 1
        else:
          result = "draw"
          self.accumulated_results[2] += 1
        
        print(f"game over - {result} \n score was {player1score} {player2score}")
        self.score = self.draw_text(300, 340, f"{result}\n{player1score} - {player2score}")
        
        self.player1_agent.train(player1score > player2score, self.learning_rate)
        self.player2_agent.train(player2score > player1score, self.learning_rate)
        if self.repeated_mode:
          if self._game_count < self.num_games:
            self._game_count += 1
            #alternate the first turn
            self.player1_turn = self._game_count % 2 == 0
            self.env.start_game()
            self.after(100, self.game_loop)
          else:
            print(f"final results {self.accumulated_results}")
          if self._game_count % self.save_period == 0:
            self.score = self.draw_text(300, 340, f"saving")
            self.player1_agent.save()
            self.player2_agent.save()
        return
          
      self.play_turn()
      self.after(self.interval, self.game_loop)

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Mancala')
    game = Game(root)
    #boards bigger than 3 make for too large a table right now
    npits = 3
    
    game.create_board(npits, 3)
    #player1 = magent.RandomAgent()
    player1 = mrl.RLAgent(f"../AIData/Tabular{npits}_1.pb", npits, 0.1)
    player1.load()
    #player2 = magent.HeuristicAgent()
    #player2 = magent.RandomAgent()
    player2 = mrl.RLAgent(f"../AIData/Tabular{npits}_2.pb", npits, 0.1)
    player2.load()
    game.setPlayer1Agent(player1)
    game.setPlayer2Agent(player2)
    game.setup()
    game.mainloop()