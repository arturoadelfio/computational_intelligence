import random
import time
import pickle
import numpy as np
from tqdm.auto import tqdm
from game import Game, Move, Player
from Montecarlo import MontecarloAgent
from QLearning import QAgent
from MinMax import MinmaxPlayer

class RandomPlayer(Player):
    """class defining a player that chooses his moves randomly"""
    def __init__(self,symbol:None) -> None:
        super().__init__()
        self.symbol=symbol

    def make_move(self, game=None, state=None) -> tuple[tuple[int, int], Move]:

        #random.seed(time.time())
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move



class MyGame(Game):

    def set_board(self,board):

        self._board=board

    def print(self):
        '''Prints the board. -1 are neutral pieces, 0 are pieces of player 0, 1 pieces of player 1'''
        for riga in self.get_board():
            for elemento in riga:
                if elemento == 0:
                    print("❌", end=" ")  # Simbolo per 0
                elif elemento == 1:
                    print("⭕️", end=" ")  # Simbolo per 1
                elif elemento == -1:
                    print("➖", end=" ")  # Simbolo per -1
            print("\n")

def train_montecarlo(agent: MontecarloAgent):

    """  with open('minmax_trained.pkl', 'rb') as file:
        agent.q_table=pickle.load(file) """
    agent.train(RandomPlayer(1-agent.symbol))
    # Save dictionary to a file
    with open('my_dict.pkl', 'wb') as file:
        pickle.dump(agent.q_table, file) 

def test_montecarlo(agent: MontecarloAgent):
    
     
    print("Montecarlo is testing...")
      
    with open('my_dict.pkl', 'rb') as file:
        agent.q_table=pickle.load(file)
    #agent.print_q_table()

    opponent=RandomPlayer(1-agent.symbol)

    agent.test(opponent)

def train_qagent(agent: QAgent):

    """  with open('minmax_trained.pkl', 'rb') as file:
        agent.q_table=pickle.load(file) """
    agent.train(RandomPlayer(1-agent.symbol))
    # Save dictionary to a file
    with open('qtable.pkl', 'wb') as file:
        pickle.dump(agent.q_table, file) 

def test_qagent(agent: QAgent):

     
    print("Q Table is testing...")
       
    print("Loading Q table")

    with open('qtable.pkl', 'rb') as file:
        agent.q_table=pickle.load(file)
    #agent.print_q_table()

    opponent=RandomPlayer(1-agent.symbol)

    agent.test(opponent)

def minmax_simulation():
    
    ITERATIONS=50
    count=0


    SYMBOL_MYAGENT=0
    SYMBOL_OPPONENT=1-SYMBOL_MYAGENT
    DEPTH_MINMAX=2
    
 
    print("MINMAX DEPTH ", DEPTH_MINMAX) 
    print("Playing as first")
    for i in tqdm(range(ITERATIONS)):

        player1 = MinmaxPlayer(DEPTH_MINMAX,SYMBOL_MYAGENT)
        player2 = RandomPlayer(SYMBOL_OPPONENT) 
        #player2 = MinmaxPlayer(3,SYMBOL_OPPONENT)
        g = MyGame()        
      
        winner = g.play( player1, player2)
        
        if winner==SYMBOL_MYAGENT:
            count+=1

    print("My player won ", count/ITERATIONS) 
    
    ITERATIONS=50
    count=0


    SYMBOL_MYAGENT=1
    SYMBOL_OPPONENT=1-SYMBOL_MYAGENT
    DEPTH_MINMAX=2
    
 
    print("MINMAX DEPTH ", DEPTH_MINMAX)
    print("Playing as second")
    for i in tqdm(range(ITERATIONS)):

        player1 = MinmaxPlayer(DEPTH_MINMAX,SYMBOL_MYAGENT)
        player2 = RandomPlayer(SYMBOL_OPPONENT) 
        #player2 = MinmaxPlayer(3,SYMBOL_OPPONENT)
        g = MyGame()        
      
        winner = g.play( player2, player1)
        
        if winner==SYMBOL_MYAGENT:
            count+=1

    print("My player won ", count/ITERATIONS) 
    

   


if __name__ == '__main__':

    #minmax_simulation()
    agent=MontecarloAgent(1)
    
    agent=QAgent(0)

    #train_montecarlo(agent)

    #print("Playing as first")
    #train_qagent(agent)
    #agent=QAgent(0)
    #agent=MontecarloAgent(1)
    #agent.print_q_table()
    #test_montecarlo(agent)
    test_qagent(agent)
    #agent.test(RandomPlayer(0))

 