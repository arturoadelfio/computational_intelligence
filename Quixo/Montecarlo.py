from random import random,choice
from copy import deepcopy
from game import Player,Move, Game
from tqdm.auto import tqdm
import numpy as np
from rotations import *

class MontecarloAgent(Player):
    """
    agent that is trained with the montecarlo algorithm of reinforcement learning
    """
    def __init__(self, symbol):
        self.q_table = {}
        self.symbol=symbol
        self._winning_games=0
        self._drawn_games=0
        self.exploration_rate=0.7
        self.rewards=[]
        self.gamma=0.9
        self.changing_symbol=False
        self.is_train=True



    def transform_state(self,state, board=np.empty((0,))):

        state_key = (frozenset(state[0]), frozenset(state[1]))
        original_board=board
        
        is_in_qtable = False
        rotation_type = ""


        if state_key in self.q_table:
            is_in_qtable = True
        else:
            state_key = (frozenset(set(rotate_90_clockwise(frozenset(state[0])))), frozenset(set(rotate_90_clockwise(frozenset(state[1])))))

            if state_key in self.q_table:
                if board.size!=0:
                    board = rotate_board_90_clockwise(board)
                is_in_qtable = True
                rotation_type = "clockwise"
            else:
                state_key = (frozenset(set(rotate_90_anticlockwise(frozenset(state[0])))), frozenset(set(rotate_90_anticlockwise(frozenset(state[1])))))
                if state_key in self.q_table:
                    if board.size!=0:
                        board = rotate_board_90_anticlockwise(board)
                    is_in_qtable = True
                    rotation_type = "anticlockwise"
                else:
                    state_key = (frozenset(set(mirror_points(frozenset(state[0])))), frozenset(set(mirror_points(frozenset(state[1])))))
                    if state_key in self.q_table:
                        if board.size!=0:
                            board = mirror_board(board)
                        is_in_qtable = True
                        rotation_type = "mirrored" 
                    else:
                        return (frozenset(state[0]), frozenset(state[1])),original_board,False, ""
        return state_key,board,is_in_qtable,rotation_type

    def transform_action(self,rotation_type,action)-> tuple[tuple[int, int], Move]:
        
        """ rotate the action according to opposite of the rotation_type"""
        #decode the action calculating the effective point (x,y) and the correct direction
        if rotation_type == "" :#board hasn't been rotated
            return action #return the action as it was calculated
        elif rotation_type == "clockwise":
            initial_point = action[0]
            direction = action[1]
            
            new_point = rotate_90_anticlockwise([(initial_point[1],initial_point[0])])[0]
            
            if direction == Move.TOP:
                direction = Move.LEFT
            elif direction == Move.LEFT:
                direction = Move.BOTTOM
            elif direction == Move.BOTTOM:
                direction = Move.RIGHT
            else: 
                direction = Move.TOP
    
            return ((new_point[1], new_point[0]), direction)
        
        elif rotation_type == "anticlockwise":
            initial_point = action[0]
            direction = action[1]
            
            new_point = rotate_90_clockwise([(initial_point[1],initial_point[0])])[0]
            
            if direction == Move.TOP:
                direction = Move.RIGHT
            elif direction == Move.LEFT:
                direction = Move.TOP
            elif direction == Move.BOTTOM:
                direction = Move.LEFT
            else: 
                direction = Move.BOTTOM

            return ((new_point[1], new_point[0]), direction)
        else: #mirrored
            initial_point = action[0]
            direction = action[1]
            
            new_point = mirror_points([(initial_point[1],initial_point[0])])[0]
            
            if direction == Move.RIGHT:
                direction = Move.LEFT
            elif direction == Move.LEFT:
                direction = Move.RIGHT
            return ((new_point[1], new_point[0]), direction)
        
    def rotate_action(self, rotation_type, action) -> tuple[tuple[int, int], Move]:

        """ rotate the action according to the rotation_type"""
           #decode the action calculating the effective point (x,y) and the correct direction
        if rotation_type == "" :#board hasn't been rotated
            return action #return the action as it was calculated
        elif rotation_type == "anticlockwise":
            initial_point = action[0]
            direction = action[1]
            
            new_point = rotate_90_anticlockwise([(initial_point[1],initial_point[0])])[0]
            
            if direction == Move.TOP:
                direction = Move.LEFT
            elif direction == Move.LEFT:
                direction = Move.BOTTOM
            elif direction == Move.BOTTOM:
                direction = Move.RIGHT
            else: 
                direction = Move.TOP
            return ((new_point[1], new_point[0]), direction)
        
        elif rotation_type == "clockwise":
            initial_point = action[0]
            direction = action[1]
            
            new_point = rotate_90_clockwise([(initial_point[1],initial_point[0])])[0]
            
            if direction == Move.TOP:
                direction = Move.RIGHT
            elif direction == Move.LEFT:
                direction = Move.TOP
            elif direction == Move.BOTTOM:
                direction = Move.LEFT
            else: 
                direction = Move.BOTTOM

            return ((new_point[1], new_point[0]), direction)
        else: #mirrored
            
            initial_point = action[0]
            direction = action[1]
            
            new_point = mirror_points([(initial_point[1],initial_point[0])])[0]
            
            if direction == Move.RIGHT:
                direction = Move.LEFT
            elif direction == Move.LEFT:
                direction = Move.RIGHT
            return ((new_point[1], new_point[0]), direction)
     

    def make_move(self,game, state)-> tuple[tuple[int, int], Move]:
        
        """ function that returns a move for the montecarlo agent """
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
                self.rewards.append(count_zeros - count_ones)
            else:
                self.rewards.append(count_ones - count_zeros)
        
        new_action= self.transform_action(rotation_type,action)
        return new_action
        
    def add_winning(self)->None:
        """
        increase the number of winnings
        """
        self._winning_games+=1
       
    def get_legal_moves(self, board):
        """
        Return the legal moves
        """
        legal_moves = []

        rows, cols = board.shape
        # Top border indices
        top_indices = [(0, i) for i in range(cols)]

        # Bottom border indices
        bottom_indices = [(rows - 1, i) for i in range(cols)]

        # Left border indices (excluding corners)
        left_indices = [(i, 0) for i in range(1, rows - 1)]

        # Right border indices (excluding corners)
        right_indices = [(i, cols - 1) for i in range(1, rows - 1)]

        indices=top_indices+bottom_indices+left_indices+right_indices

        for x,y in indices:
            for direction in Move:
                if self.is_move_playable(board, (y,x), direction):
                                #print("back da playable")
                                legal_moves.append(((x, y), direction))
        #game.print()
        #print(legal_moves)
        return legal_moves

    def is_move_playable(self, board, position, direction):
        """
        check wheter the proposed move is applicable
        """
        x, y = position

        acceptable: bool = (
            # check if it is in the first row
            (x == 0 and y < 5)
            # check if it is in the last row
            or (x == 4 and y< 5)
            # check if it is in the first column
            or (x <5 and y ==0)
            # check if it is in the last column
            or (x <5 and y == 4)
            # and check if the piece can be moved by the current player
        ) 
        if acceptable is False:
            return False
        # Check if the move is within the bounds of the board
        if not (0 <= x < board.shape[0] and 0 <= y < board.shape[1]):
            return False

        if board[x, y] == 1-self.symbol:
            return False
        # Check if the move is towards an empty cell
        if direction == Move.TOP and x==0:
            #print("STEP 1")
            return False
        elif direction == Move.BOTTOM and x==4:
            #print("STEP 2")
            return False
        elif direction == Move.LEFT and y == 0:
            #print("STEP 3")
            return False
        elif direction == Move.RIGHT and y == 4:
            #print("STEP 4")
            return False

        return True

    def print_q_table(self):

        """
        print the q table
        """

        print("Printing last 5 rows...")
        for chiave, valore in list(self.q_table.items())[:5]:
            print(f'{chiave}: {valore} \n')

    def update_q_table(self, trajectory):
        """
        update the values in the q table
        """
        counter=0
        for state,action in trajectory:
            counter+=1

            new_state,_,_,rotation_type=self.transform_state(state)
            new_action=self.rotate_action(rotation_type, action)
            
            if (new_state) in self.q_table and new_action in self.q_table[new_state]:
               
                self.q_table[new_state][new_action]+=0.001* (sum(self.rewards) -  self.q_table[new_state][new_action])
                #self.q_table[new_state][new_action]+=self.gamma * (sum(self.rewards)-  self.q_table[new_state][new_action])
                
                #self.q_table[new_state][new_action]+= (sum(self.rewards) -  self.q_table[new_state][new_action])/1000

    def train(self,opponent):
        self._winning_games=0

        self.is_train=True
        num_iterations=5_000
        
        print("Montecarlo is training...")
        for i in tqdm(range(num_iterations)):

            # if i>(num_iterations/4)*3 and self.exploration_rate>0.1:
            #     #self.exploration_rate=((1-self.exploration_rate)*10)/i 
            #     self.exploration_rate*=0.99
            game=MontecarloGame()
            
            if self.symbol==0:
                _,winner=game.play(self, opponent)  
            else:
                _,winner=game.play(opponent, self)
    
            if i==num_iterations/2: 
            #if random()<0.5:
                self.symbol=1-self.symbol
                opponent.symbol=1-self.symbol

            if winner==self.symbol:
                self.add_winning()
        
        print("My player won ", self._winning_games/num_iterations)


    def test(self, opponent):
        self._winning_games=0
        self.is_train=False
        self.changing_symbol=False
        #self.symbol=0
        num_iterations=100

        self.exploration_rate=0
        
        for i in tqdm(range(num_iterations)):
            game=MontecarloGame()
            
            if self.symbol==0:
                _,winner=game.play(self, opponent)
            else:
                _,winner=game.play(opponent, self)
                
            """  if i==num_iterations/2:
                self.symbol=1-self.symbol
                opponent.symbol=1-self.symbol  """  
            
            if winner==self.symbol:
                self.add_winning()
        print("My player won ", self._winning_games/num_iterations*100, "% of the times")

class MontecarloGame(Game):
    """
    a subclass of the game class that changes the play method to adapt to the montecarlo agent
    """
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

        if isinstance(player1, MontecarloAgent):
            final_reward, winner= (5, 0) if super().check_winner()==player1.symbol else (-5,1)

            player1.rewards.append(final_reward)
            if player1.is_train:
                player1.update_q_table(trajectory)

            player1.rewards=[]

        elif isinstance(player2, MontecarloAgent):
            final_reward, winner= (5, 1) if super().check_winner()==player2.symbol else (-5,0)
            player2.rewards.append(final_reward)
            
            if player2.is_train:
                player2.update_q_table(trajectory)
            player2.rewards=[]

        return trajectory, winner

    def set_board(self,board):
        self._board=board

