from random import random,choice
from copy import deepcopy
from game import Player,Move, Game
from tqdm.auto import tqdm
from Montecarlo import MontecarloAgent



class QAgent(MontecarloAgent):
    
    def __init__(self, symbol):
        self.alpha=0.7
        self.exploration_rate=0.3
        self.gamma=0.9

        super().__init__(symbol)

    def make_move(self, game, state) -> tuple[tuple[int, int], Move]:

        """ if self.alpha>0.1:
            self.alpha*=0.99 """

        """ if self.gamma<0.99:
            self.gamma*=1.01 """

        action=None
        
        board = game.get_board()
        state_key,board,is_in_qtable,rotation_type=self.transform_state(state, game.get_board())

        available_moves=list(self.get_legal_moves(board))
  
        if random() < self.exploration_rate and self.is_train:
            # sometimes make random moves
            action = choice(available_moves)
            
            if not is_in_qtable:
                self.q_table[state_key] = dict.fromkeys([action], 0)
 
        else:
            if not is_in_qtable and self.is_train:
                self.q_table[state_key] = dict.fromkeys(available_moves, 0)
                is_in_qtable=True
            #choose the action based on the q table
            if is_in_qtable: 
                action = max(self.q_table[state_key], key=self.q_table[state_key].get)
                
                #If the best action has a negative value, all the possible moves are added 
                if self.q_table[state_key][action] < 0 and len(self.q_table[state_key])==1:
                    for move in available_moves:
                        if move not in self.q_table[state_key].keys():
 
                            self.q_table[state_key][move]=0

                    action = max(self.q_table[state_key], key=self.q_table[state_key].get)
                     
                         
            if action is None or action not in available_moves:

                action=choice(list(available_moves))

                if self.is_train:
                    self.q_table[state_key] = dict.fromkeys([action], 0) 


        if self.is_train:            
            count_zeros=0
            count_ones=0
            for x in range(game.get_board().shape[0]):  
                for y in range(game.get_board().shape[1]):
                    
                    el=game.get_board()[x][y]
                             
                    if el==0:
                        count_zeros+=1
                    elif el==1:
                        count_ones+=1
           
            if self.symbol==0:
                reward=count_zeros - count_ones
            else:
                reward=count_ones - count_zeros 
            
            self.update_q_table(action, available_moves, state_key,reward)
        new_action= self.transform_action(rotation_type,action)
        return new_action

    def train(self,opponent):
        self._winning_games=0
        self.changing_symbol=False
        self.is_train=True
        num_iterations=50_000
        
        print("Q Agent is training...")
        for i in tqdm(range(num_iterations)):

            #if i>(num_iterations/3)*2 and self.exploration_rate>0.1:
                #self.exploration_rate=((1-self.exploration_rate)*10)/i 
                #self.exploration_rate*=0.99
            game=QGame()
            
            if self.symbol==0:
                trajectory,final_reward,winner=game.play(self, opponent)  
            else:
                trajectory,final_reward,winner=game.play(opponent, self)
    
            if final_reward == -1:
            
                  s=trajectory[-2][0]
                  a=trajectory[-2][1]

            else:  
                  s=trajectory[-1][0]
                  a=trajectory[-1][1] 

            s,_,_,rotation_type=self.transform_state(s)
            a=self.rotate_action(rotation_type, a)                  

            self.update_q_table(a,(),(frozenset(s[0]),frozenset(s[1])),final_reward)
            
      
            if i==num_iterations/2:
            #if random()<0.5:
                self.symbol=1-self.symbol
                opponent.symbol=1-self.symbol
                
                
            if winner==self.symbol:
                self.add_winning()
        
        print("My player won ", self._winning_games/num_iterations)
        print(self.exploration_rate)

    def test(self, opponent):

        self._winning_games=0
        self.is_train=False
        self.changing_symbol=False
        #self.symbol=0
        num_iterations=100

        self.exploration_rate=0
        
        for i in tqdm(range(num_iterations)):
            game=QGame()
            
            if self.symbol==0:
                _,_,winner=game.play(self, opponent)
            else:
                _,_,winner=game.play(opponent, self)
                
            """ if i==num_iterations/2:
                self.symbol=1-self.symbol
                opponent.symbol=1-self.symbol   """
            
            if winner==self.symbol:
                self.add_winning()
        print("My player won ", self._winning_games/num_iterations*100, "% of the times")

    def update_q_table(self, action, available_moves,state_key, reward=None):

        """ update the q table """
        if state_key not in self.q_table:
            self.q_table[state_key] = dict.fromkeys([action], 0)

        state=(set(state_key[0]),set(state_key[1]))
        new_state = deepcopy(state)
        new_state[0].add(action)
        next_state_key = (frozenset(new_state[0]), frozenset(new_state[1]))

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = dict.fromkeys(available_moves, 1)
        
        self.q_table[state_key][action] = (1 - self.alpha) * self.q_table[state_key].get(action, 0) + self.alpha * (reward + self.gamma * (max(self.q_table[next_state_key].values(), default=0)))
  
    
    

class QGame(Game):

    def __move(self, from_pos, slide, index):
        return super()._Game__move(from_pos, slide, index)


    def play(self, player1: Player, player2: Player) -> int:

        trajectory=list()
        state=(set(), set())
        
        players=[player1,player2]
        #print(players)
        index=0
        while True:

            ok = False
            current_player=players[index]
            for x in range(super().get_board().shape[0]): 
                for y in range(super().get_board().shape[1]):
                    if super().get_board()[x][y]==0:
                        state[0].add((x,y))
                    elif super().get_board()[x][y]==1:
                        state[1].add((x,y))
            while not ok:
                from_pos, slide = current_player.make_move(self,state)

                ok = self.__move(from_pos, slide, current_player.symbol)
            
            move=(from_pos,slide)
            
            if(super().check_winner()!=-1):
                trajectory.append((deepcopy(state),move))
                break

            index=1-index

            trajectory.append((deepcopy(state),move))
            state=(set(), set())            

        if isinstance(player1, QAgent):
            final_reward, winner= (5, 0) if super().check_winner()==player1.symbol else (-5,1)


        elif isinstance(player2, QAgent):
            final_reward, winner= (5, 1) if super().check_winner()==player2.symbol else (-5,0)

        return trajectory, final_reward, winner
