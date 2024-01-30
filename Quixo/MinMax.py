from game import Game, Move, Player
from copy import deepcopy

class MinmaxPlayer(Player):
    """
    player that plays according to the minmax algorithm
    """
    def __init__(self, depth, symbol):
        self.depth = depth
        self.symbol=symbol
        self.maximizer=True

    def make_move(self,game,state=None):
        #print("MINMAX MOVING")
        _, best_move = self.minimax(game, self.depth, True, float('-inf'), float('inf'))
        return best_move

    def minimax(self, game, depth, maximizing_player, alpha, beta):
        """
        minmax algorithm
        """
        self.maximizer=maximizing_player    
            
        if depth == 0 or game.check_winner() != -1:
            return self.evaluate(game), None

        legal_moves = self.get_legal_moves(game)
        next_states = self.calculate_next_states(game, legal_moves)
        
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for i, new_game in enumerate(next_states):
               
                eval, _ = self.minimax(new_game, depth - 1, False, alpha, beta)
                #print("MAX",eval)
                #game.print()
                if eval > max_eval:
                    max_eval = eval
                    best_move = legal_moves[i]
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    #print("PRUNED")
                    break
            #self.maximizer=False
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for i, new_game in enumerate(next_states):

                eval, _ = self.minimax(new_game, depth - 1, True, alpha, beta)
                #print("MIN", eval)
                if eval < min_eval:
                    min_eval = eval
                    best_move = legal_moves[i]                
                    beta = min(beta, min_eval)
                if beta <= alpha:
                    #print("PRUNED")
                    break
            #print("best_move, ", best_move)
            #self.maximizer=True  
            
            #best_move=(best_move[1],best_move[0])
  
            return min_eval, best_move
        
    def calculate_next_states(self, game, legal_moves):
        next_states = []

        for move in legal_moves:
            position, direction = move
            new_game = deepcopy(game)
            new_game=self.apply_move(new_game, position, direction)
            next_states.append(new_game)

        return next_states

    def apply_move(self, game, pos, direction):
        
        board=deepcopy(game.get_board())
        #position=pos
        position=(pos[1],pos[0])
        #print("ORIGINAL BOARD", game.get_board())
        #print("POSITION", position)
        if self.maximizer is True:
            board[position] = self.symbol 
        else:
            board[position] =(1-self.symbol)

        piece = board[position]
        #print("PIECE", piece)
        # if the player wants to slide it to the left
        if direction == Move.LEFT:
            # for each column starting from the column of the piece and moving to the left
            for i in range(position[1], 0, -1):
                # copy the value contained in the same row and the previous column
                board[(position[0], i)] = board[(
                    position[0], i - 1)]
            # move the piece to the left
            board[(position[0], 0)] = piece
        # if the player wants to slide it to the right
        elif direction == Move.RIGHT:
            # for each column starting from the column of the piece and moving to the right
            for i in range(position[1], board.shape[1] - 1, 1):
                # copy the value contained in the same row and the following column
                board[(position[0], i)] = board[(
                    position[0], i + 1)]
            # move the piece to the right
            board[(position[0], board.shape[1] - 1)] = piece
        # if the player wants to slide it upward
        elif direction == Move.TOP:

            # for each row starting from the row of the piece and going upward
            for i in range(position[0], 0, -1):
                # copy the value contained in the same column and the previous row
                board[(i, position[1])] = board[(i - 1, position[1])]
            # move the piece up
            board[(0, position[1])] = piece

        # if the player wants to slide it downward
        elif direction == Move.BOTTOM:
            # for each row starting from the row of the piece and going downward
            for i in range(position[0], board.shape[0] - 1, 1):
                # copy the value contained in the same column and the following row
                board[(i, position[1])] = board[(
                    i + 1, position[1])]
            # move the piece down
            board[(board.shape[0] - 1, position[1])] = piece

        #print(board)
        game.set_board(board)
        return game
      
    def evaluate(self, game:'Game'):
        """
            function that evaluates the leaves
        """

        winner = game.check_winner()

        if winner == self.symbol:
            return 10
           
        elif winner == 1-self.symbol:
            return -10

        else:
            # return the difference between the 0 and the ones 
            count_zeros=0
            count_ones=0
            for x in range(game.get_board().shape[0]):  
                for y in range(game.get_board().shape[1]):
                    
                    el=game.get_board()[x][y]
                    #print(el)                
                    if el==0:
                        count_zeros+=1
                    elif el==1:
                        count_ones+=1

            if self.symbol==0:
                return count_zeros - count_ones
            else:
                return count_ones - count_zeros
          
    def get_legal_moves(self, game:Game):
        """
        Return the legal moves
        """
        legal_moves = []

        rows, cols = game.get_board().shape
        # Top border indices
        top_indices = [(0, i) for i in range(cols)]

        # Bottom border indices
        bottom_indices = [(rows - 1, i) for i in range(cols)]

        # Left border indices (excluding corners)
        left_indices = [(i, 0) for i in range(1, rows - 1)]

        # Right border indices (excluding corners)
        right_indices = [(i, cols - 1) for i in range(1, rows - 1)]

        indices=top_indices+ bottom_indices+left_indices+right_indices
        #print(indices)

        for x,y in indices:
            for direction in Move:
                if self.is_move_playable(game, (y,x), direction):
                                legal_moves.append(((x, y), direction))

        #game.print()
        #print(legal_moves)
        return legal_moves

    def is_move_playable(self, game, position, direction):
        x,y= position

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
        if not (0 <= x < game.get_board().shape[0] and 0 <= y < game.get_board().shape[1]):
            return False

        if self.maximizer:
            if game.get_board()[x, y] == 1-self.symbol:
                return False
        else:
            if game.get_board()[x, y] == self.symbol:
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

